"""
PDF report generator for restaurant analysis.
"""
import os
import datetime
from typing import Dict, List, Any, Optional, Tuple
import tempfile

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch


class ReportGenerator:
    """Class for generating PDF reports."""
    
    def __init__(self, output_dir: str = "."):
        """Initialize report generator."""
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles."""
        # Add custom styles
        self.styles.add(
            ParagraphStyle(
                name='Heading1',
                parent=self.styles['Heading1'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.darkblue
            )
        )
        
        self.styles.add(
            ParagraphStyle(
                name='Heading2',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceAfter=10,
                textColor=colors.darkblue
            )
        )
        
        self.styles.add(
            ParagraphStyle(
                name='Normal',
                parent=self.styles['Normal'],
                fontSize=10,
                spaceAfter=8
            )
        )
        
        self.styles.add(
            ParagraphStyle(
                name='Bullet',
                parent=self.styles['Normal'],
                fontSize=10,
                leftIndent=20,
                firstLineIndent=-15,
                spaceAfter=5
            )
        )
    
    def generate_report(
        self,
        restaurant_info: Dict[str, Any],
        reviews: List[Dict[str, Any]],
        sentiment_summary: Dict[str, Any],
        sentiment_trends: Dict[str, Any],
        llm_analysis: Dict[str, str],
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> str:
        """Generate a PDF report with restaurant analysis."""
        # Generate filename
        report_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        restaurant_name = restaurant_info.get("name", "restaurant").replace(" ", "_")
        filename = f"{restaurant_name}_analysis_{report_date}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create document
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        
        # Build document content
        elements = []
        
        # Add title
        elements.append(Paragraph(f"Restaurant Analysis Report: {restaurant_info.get('name')}", self.styles["Title"]))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add date range
        date_range = "All Time"
        if start_date and end_date:
            date_range = f"{start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"
        elif start_date:
            date_range = f"From {start_date.strftime('%B %d, %Y')}"
        elif end_date:
            date_range = f"Until {end_date.strftime('%B %d, %Y')}"
            
        elements.append(Paragraph(f"Date Range: {date_range}", self.styles["Normal"]))
        elements.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y, %I:%M %p')}", self.styles["Normal"]))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add restaurant information
        elements.append(Paragraph("Restaurant Information", self.styles["Heading1"]))
        elements.append(Spacer(1, 0.1*inch))
        
        # Create a table for restaurant info
        restaurant_data = [
            ["Name", restaurant_info.get("name", "N/A")],
            ["Address", f"{restaurant_info.get('address', '')}, {restaurant_info.get('city', '')}, {restaurant_info.get('state', '')} {restaurant_info.get('postal_code', '')}"],
            ["Cuisine Type", restaurant_info.get("cuisine_type", "N/A")],
            ["Price Range", restaurant_info.get("price_range", "N/A")],
            ["Phone", restaurant_info.get("phone", "N/A")],
            ["Website", restaurant_info.get("website", "N/A")],
            ["Average Rating", f"{restaurant_info.get('average_rating', 0):.1f}/5.0"]
        ]
        
        restaurant_table = Table(restaurant_data, colWidths=[1.5*inch, 5*inch])
        restaurant_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        elements.append(restaurant_table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add executive summary
        elements.append(Paragraph("Executive Summary", self.styles["Heading1"]))
        elements.append(Paragraph(llm_analysis.get("executive_summary", "Not available"), self.styles["Normal"]))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add sentiment overview
        elements.append(Paragraph("Sentiment Overview", self.styles["Heading1"]))
        
        # Create sentiment charts
        sentiment_charts = self._create_sentiment_charts(sentiment_summary, sentiment_trends)
        for chart in sentiment_charts:
            elements.append(Image(chart, width=6.5*inch, height=3*inch))
            elements.append(Spacer(1, 0.1*inch))
        
        # Add sentiment summary text
        sentiment_text = f"""
        Total Reviews: {sentiment_summary.get('total_reviews')}
        Positive Reviews: {sentiment_summary.get('positive_count')} ({sentiment_summary.get('positive_percentage', 0):.1f}%)
        Negative Reviews: {sentiment_summary.get('negative_count')} ({sentiment_summary.get('negative_percentage', 0):.1f}%)
        Neutral Reviews: {sentiment_summary.get('neutral_count')} ({sentiment_summary.get('neutral_percentage', 0):.1f}%)
        Overall Sentiment Trend: {sentiment_trends.get('overall_trend', 'stable').capitalize()}
        """
        elements.append(Paragraph(sentiment_text, self.styles["Normal"]))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add strengths
        elements.append(Paragraph("Strengths", self.styles["Heading1"]))
        elements.append(Paragraph(llm_analysis.get("strengths", "Not available"), self.styles["Normal"]))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add areas for improvement
        elements.append(Paragraph("Areas for Improvement", self.styles["Heading1"]))
        elements.append(Paragraph(llm_analysis.get("areas_for_improvement", "Not available"), self.styles["Normal"]))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add customer experience
        elements.append(Paragraph("Customer Experience", self.styles["Heading1"]))
        elements.append(Paragraph(llm_analysis.get("customer_experience", "Not available"), self.styles["Normal"]))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add food quality
        elements.append(Paragraph("Food Quality", self.styles["Heading1"]))
        elements.append(Paragraph(llm_analysis.get("food_quality", "Not available"), self.styles["Normal"]))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add recommendations
        elements.append(Paragraph("Recommendations", self.styles["Heading1"]))
        elements.append(Paragraph(llm_analysis.get("recommendations", "Not available"), self.styles["Normal"]))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add review examples
        elements.append(Paragraph("Sample Reviews", self.styles["Heading1"]))
        elements.append(Spacer(1, 0.1*inch))
        
        # Add positive reviews
        elements.append(Paragraph("Positive Reviews", self.styles["Heading2"]))
        positive_reviews = [r for r in reviews if r.get("sentiment", {}).get("combined_label") == "positive"]
        for i, review in enumerate(positive_reviews[:3]):
            elements.append(Paragraph(f"<b>Rating:</b> {review.get('rating')}/5", self.styles["Normal"]))
            elements.append(Paragraph(f"<b>Review:</b> {review.get('review_text')}", self.styles["Normal"]))
            elements.append(Spacer(1, 0.1*inch))
            
        # Add negative reviews
        elements.append(Paragraph("Critical Reviews", self.styles["Heading2"]))
        negative_reviews = [r for r in reviews if r.get("sentiment", {}).get("combined_label") == "negative"]
        for i, review in enumerate(negative_reviews[:3]):
            elements.append(Paragraph(f"<b>Rating:</b> {review.get('rating')}/5", self.styles["Normal"]))
            elements.append(Paragraph(f"<b>Review:</b> {review.get('review_text')}", self.styles["Normal"]))
            elements.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(elements)
        
        return filepath
    
    def _create_sentiment_charts(
        self,
        sentiment_summary: Dict[str, Any],
        sentiment_trends: Dict[str, Any]
    ) -> List[str]:
        """Create charts for sentiment analysis."""
        chart_files = []
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["font.family"] = "sans-serif"
        
        # Create sentiment distribution pie chart
        pie_chart_file = self._create_sentiment_pie_chart(sentiment_summary)
        if pie_chart_file:
            chart_files.append(pie_chart_file)
        
        # Create sentiment trend chart
        trend_chart_file = self._create_sentiment_trend_chart(sentiment_trends)
        if trend_chart_file:
            chart_files.append(trend_chart_file)
        
        return chart_files
    
    def _create_sentiment_pie_chart(self, sentiment_summary: Dict[str, Any]) -> Optional[str]:
        """Create a pie chart for sentiment distribution."""
        try:
            # Create data
            labels = ['Positive', 'Neutral', 'Negative']
            sizes = [
                sentiment_summary.get('positive_percentage', 0),
                sentiment_summary.get('neutral_percentage', 0),
                sentiment_summary.get('negative_percentage', 0)
            ]
            colors = ['#4CAF50', '#FFC107', '#F44336']
            explode = (0.1, 0, 0)  # explode the 1st slice (Positive)
            
            # Create pie chart
            plt.figure(figsize=(8, 6))
            plt.pie(
                sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140
            )
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Sentiment Distribution')
            
            # Save chart to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
        except Exception as e:
            print(f"Error creating pie chart: {str(e)}")
            return None
    
    def _create_sentiment_trend_chart(self, sentiment_trends: Dict[str, Any]) -> Optional[str]:
        """Create a line chart for sentiment trends over time."""
        try:
            monthly_data = sentiment_trends.get('monthly_sentiment', [])
            
            if not monthly_data:
                return None
                
            # Create DataFrame from monthly data
            df = pd.DataFrame(monthly_data)
            
            # Convert month strings to datetime for better x-axis formatting
            df['month'] = pd.to_datetime(df['month'] + '-01')
            
            # Create line chart
            plt.figure(figsize=(10, 6))
            
            # Plot sentiment trend
            plt.subplot(2, 1, 1)
            plt.plot(df['month'], df['average_compound'], marker='o', color='#2196F3', linewidth=2)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            plt.title('Sentiment Trend Over Time')
            plt.ylabel('Sentiment Score')
            plt.ylim(-1, 1)
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Plot review count
            plt.subplot(2, 1, 2)
            plt.bar(df['month'], df['review_count'], color='#4CAF50', alpha=0.7)
            plt.title('Review Count by Month')
            plt.ylabel('Number of Reviews')
            plt.grid(True, axis='y', alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            
            # Save chart to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
        except Exception as e:
            print(f"Error creating trend chart: {str(e)}")
            return None