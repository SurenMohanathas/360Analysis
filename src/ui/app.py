"""
Main UI application for the 360Analysis tool.
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
import webbrowser

import customtkinter as ctk
from PIL import Image, ImageTk
from ttkthemes import ThemedTk

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.database.connection import (
    get_restaurants, search_restaurants, get_restaurant_by_id,
    get_reviews_for_restaurant, get_rating_distribution, get_reviews_by_platform,
    get_review_counts_by_month
)
from src.analysis.sentiment import (
    analyze_reviews_batch, get_sentiment_summary, get_sentiment_by_platform,
    get_sentiment_trends
)
from src.analysis.llm import LLMFactory
from src.report.generator import ReportGenerator


class RestaurantAnalysisApp:
    """Main application class for the restaurant analysis tool."""
    
    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.root.title("360Analysis - Restaurant Review Analyzer")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Set theme and style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Configure style
        self.style.configure('TLabel', font=('Helvetica', 11))
        self.style.configure('TButton', font=('Helvetica', 11))
        self.style.configure('TEntry', font=('Helvetica', 11))
        self.style.configure('TCombobox', font=('Helvetica', 11))
        
        # Initialize variables
        self.search_var = tk.StringVar()
        self.restaurant_id_var = tk.StringVar()
        self.restaurant_name_var = tk.StringVar()
        self.start_date_var = tk.StringVar()
        self.end_date_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.selected_restaurant = None
        
        # Configure main frame
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create and place widgets
        self._create_widgets()
        
        # Load restaurants
        self._load_restaurants()
    
    def _create_widgets(self):
        """Create and place UI widgets."""
        # Configure grid layout
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(3, weight=1)
        
        # Create header
        header_frame = ttk.Frame(self.main_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        ttk.Label(
            header_frame, 
            text="360Analysis", 
            font=('Helvetica', 24, 'bold')
        ).pack(side=tk.LEFT)
        
        ttk.Label(
            header_frame,
            text="Restaurant Review Analysis Tool",
            font=('Helvetica', 14)
        ).pack(side=tk.LEFT, padx=20)
        
        # Create search frame
        search_frame = ttk.LabelFrame(self.main_frame, text="Search Restaurants", padding=10)
        search_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        search_frame.columnconfigure(1, weight=1)
        
        ttk.Label(search_frame, text="Restaurant Name:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        search_entry.bind("<Return>", lambda e: self._search_restaurants())
        
        ttk.Button(
            search_frame, 
            text="Search", 
            command=self._search_restaurants
        ).grid(row=0, column=2, sticky="e")
        
        # Create results frame
        results_frame = ttk.LabelFrame(self.main_frame, text="Search Results", padding=10)
        results_frame.grid(row=2, column=0, sticky="ew", pady=(0, 20))
        results_frame.columnconfigure(0, weight=1)
        
        # Create Treeview for restaurants
        self.restaurant_tree = ttk.Treeview(
            results_frame,
            columns=("name", "address", "cuisine", "rating"),
            show="headings",
            height=5
        )
        
        # Define columns
        self.restaurant_tree.heading("name", text="Restaurant Name")
        self.restaurant_tree.heading("address", text="Address")
        self.restaurant_tree.heading("cuisine", text="Cuisine Type")
        self.restaurant_tree.heading("rating", text="Rating")
        
        self.restaurant_tree.column("name", width=200, minwidth=150)
        self.restaurant_tree.column("address", width=250, minwidth=150)
        self.restaurant_tree.column("cuisine", width=150, minwidth=100)
        self.restaurant_tree.column("rating", width=80, minwidth=50)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.restaurant_tree.yview)
        self.restaurant_tree.configure(yscroll=scrollbar.set)
        
        # Place treeview and scrollbar
        self.restaurant_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Bind selection event
        self.restaurant_tree.bind("<<TreeviewSelect>>", self._on_restaurant_select)
        
        # Create analysis frame
        analysis_frame = ttk.LabelFrame(self.main_frame, text="Generate Analysis", padding=10)
        analysis_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 20))
        analysis_frame.columnconfigure(1, weight=1)
        analysis_frame.rowconfigure(4, weight=1)
        
        # Selected restaurant
        ttk.Label(analysis_frame, text="Selected Restaurant:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Label(analysis_frame, textvariable=self.restaurant_name_var, font=('Helvetica', 11, 'bold')).grid(row=0, column=1, sticky="w", pady=5)
        
        # Date range
        date_frame = ttk.Frame(analysis_frame)
        date_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(date_frame, text="Date Range:").pack(side=tk.LEFT, padx=(0, 10))
        
        date_presets = ttk.Frame(date_frame)
        date_presets.pack(side=tk.LEFT)
        
        ttk.Button(
            date_presets, 
            text="Last 30 Days",
            command=lambda: self._set_date_range(30)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            date_presets, 
            text="Last 90 Days",
            command=lambda: self._set_date_range(90)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            date_presets, 
            text="Last Year",
            command=lambda: self._set_date_range(365)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            date_presets, 
            text="All Time",
            command=lambda: self._set_date_range(0)
        ).pack(side=tk.LEFT, padx=5)
        
        # Custom date range
        custom_date_frame = ttk.Frame(analysis_frame)
        custom_date_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(custom_date_frame, text="From:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(custom_date_frame, textvariable=self.start_date_var, width=12).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(custom_date_frame, text="To:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(custom_date_frame, textvariable=self.end_date_var, width=12).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(custom_date_frame, text="(YYYY-MM-DD format)").pack(side=tk.LEFT)
        
        # Generate button
        ttk.Button(
            analysis_frame,
            text="Generate Analysis Report",
            command=self._generate_analysis,
            style='Accent.TButton'
        ).grid(row=3, column=0, columnspan=2, pady=20)
        
        # Progress and status
        status_frame = ttk.Frame(self.main_frame)
        status_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(
            self.main_frame,
            orient=tk.HORIZONTAL,
            length=100,
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.grid(row=5, column=0, sticky="ew")
    
    def _load_restaurants(self):
        """Load all restaurants from the database."""
        try:
            restaurants = get_restaurants()
            self._populate_restaurant_tree(restaurants)
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to load restaurants: {str(e)}")
    
    def _search_restaurants(self):
        """Search for restaurants by name."""
        search_text = self.search_var.get().strip()
        
        if not search_text:
            self._load_restaurants()
            return
        
        try:
            restaurants = search_restaurants(search_text)
            self._populate_restaurant_tree(restaurants)
        except Exception as e:
            messagebox.showerror("Search Error", f"Failed to search restaurants: {str(e)}")
    
    def _populate_restaurant_tree(self, restaurants):
        """Populate the restaurant treeview with search results."""
        # Clear existing items
        self.restaurant_tree.delete(*self.restaurant_tree.get_children())
        
        # Add restaurants to treeview
        for restaurant in restaurants:
            self.restaurant_tree.insert(
                "",
                "end",
                values=(
                    restaurant.get("name", ""),
                    restaurant.get("address", ""),
                    restaurant.get("cuisine_type", ""),
                    f"{restaurant.get('average_rating', 0):.1f}"
                ),
                tags=(str(restaurant.get("id")),)
            )
    
    def _on_restaurant_select(self, event):
        """Handle restaurant selection from the treeview."""
        selected_items = self.restaurant_tree.selection()
        
        if not selected_items:
            return
            
        item = selected_items[0]
        restaurant_id = self.restaurant_tree.item(item, "tags")[0]
        
        try:
            restaurant = get_restaurant_by_id(int(restaurant_id))
            self.selected_restaurant = restaurant
            self.restaurant_id_var.set(str(restaurant.get("id")))
            self.restaurant_name_var.set(restaurant.get("name", ""))
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to load restaurant details: {str(e)}")
    
    def _set_date_range(self, days):
        """Set date range based on preset buttons."""
        if days == 0:
            # All time
            self.start_date_var.set("")
            self.end_date_var.set("")
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            self.start_date_var.set(start_date.strftime("%Y-%m-%d"))
            self.end_date_var.set(end_date.strftime("%Y-%m-%d"))
    
    def _generate_analysis(self):
        """Generate restaurant analysis report."""
        if not self.selected_restaurant:
            messagebox.showwarning("No Restaurant", "Please select a restaurant first.")
            return
        
        # Parse date range
        start_date = None
        end_date = None
        
        try:
            if self.start_date_var.get().strip():
                start_date = datetime.strptime(self.start_date_var.get().strip(), "%Y-%m-%d")
                
            if self.end_date_var.get().strip():
                end_date = datetime.strptime(self.end_date_var.get().strip(), "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid Date", "Please use YYYY-MM-DD format for dates.")
            return
        
        # Start analysis thread
        self.status_var.set("Starting analysis...")
        self.progress_var.set(0)
        
        analysis_thread = threading.Thread(
            target=self._run_analysis,
            args=(self.selected_restaurant, start_date, end_date)
        )
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def _run_analysis(self, restaurant, start_date, end_date):
        """Run the full analysis process in a separate thread."""
        try:
            # Step 1: Fetch reviews
            self.status_var.set("Fetching reviews...")
            self.progress_var.set(10)
            reviews = get_reviews_for_restaurant(restaurant["id"], start_date, end_date)
            
            if not reviews:
                self.root.after(0, lambda: messagebox.showinfo(
                    "No Reviews", 
                    "No reviews found for the selected date range."
                ))
                self.status_var.set("Ready")
                self.progress_var.set(0)
                return
            
            # Step 2: Analyze sentiment
            self.status_var.set("Analyzing sentiment...")
            self.progress_var.set(30)
            analyzed_reviews = analyze_reviews_batch(reviews)
            
            # Step 3: Generate summaries
            self.status_var.set("Generating summaries...")
            self.progress_var.set(50)
            sentiment_summary = get_sentiment_summary(analyzed_reviews)
            sentiment_trends = get_sentiment_trends(analyzed_reviews)
            
            # Step 4: LLM analysis
            self.status_var.set("Generating LLM analysis...")
            self.progress_var.set(70)
            llm = LLMFactory.get_instance().get_llm()
            llm_analysis = llm.generate_restaurant_analysis(
                restaurant, analyzed_reviews, sentiment_summary, sentiment_trends
            )
            
            # Step 5: Generate report
            self.status_var.set("Generating PDF report...")
            self.progress_var.set(90)
            report_generator = ReportGenerator()
            report_path = report_generator.generate_report(
                restaurant, analyzed_reviews, sentiment_summary, 
                sentiment_trends, llm_analysis, start_date, end_date
            )
            
            # Done
            self.status_var.set("Analysis complete!")
            self.progress_var.set(100)
            
            # Show success message and open report
            self.root.after(0, lambda: self._on_analysis_complete(report_path))
            
        except Exception as e:
            # Show error message
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.status_var.set("Analysis failed")
            self.progress_var.set(0)
    
    def _on_analysis_complete(self, report_path):
        """Handle analysis completion."""
        result = messagebox.askquestion(
            "Analysis Complete",
            f"Analysis complete! Report saved to:\n{report_path}\n\nDo you want to open the report now?"
        )
        
        if result == 'yes':
            try:
                # Open the PDF file with the default PDF viewer
                if sys.platform == 'win32':
                    os.startfile(report_path)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f"open '{report_path}'")
                else:  # Linux
                    os.system(f"xdg-open '{report_path}'")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open the report: {str(e)}")
        
        # Reset UI
        self.status_var.set("Ready")
        self.progress_var.set(0)


def main():
    """Main entry point for the application."""
    root = ThemedTk(theme="arc")
    app = RestaurantAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()