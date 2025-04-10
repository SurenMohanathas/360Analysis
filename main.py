#!/usr/bin/env python3
"""
Main entry point for the 360Analysis application.
"""
import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("360analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("360analysis")

# Load environment variables
load_dotenv("config.env", override=True)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="360Analysis - Restaurant Review Analysis Tool")
    parser.add_argument(
        "--no-gui", 
        action="store_true",
        help="Run in command-line mode instead of GUI"
    )
    parser.add_argument(
        "--restaurant", 
        type=str,
        help="Restaurant name to analyze (for command-line mode)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default=".",
        help="Output directory for reports (default: current directory)"
    )
    parser.add_argument(
        "--start-date", 
        type=str,
        help="Start date for analysis (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--end-date", 
        type=str,
        help="End date for analysis (YYYY-MM-DD format)"
    )
    
    args = parser.parse_args()
    
    if args.no_gui:
        # Run in command-line mode
        from datetime import datetime
        from src.database.connection import search_restaurants, get_restaurant_by_id, get_reviews_for_restaurant
        from src.analysis.sentiment import analyze_reviews_batch, get_sentiment_summary, get_sentiment_trends
        from src.analysis.llm import LLMFactory
        from src.report.generator import ReportGenerator
        
        if not args.restaurant:
            logger.error("Restaurant name is required in command-line mode")
            parser.print_help()
            sys.exit(1)
        
        try:
            # Parse dates
            start_date = None
            end_date = None
            
            if args.start_date:
                start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
                
            if args.end_date:
                end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
            
            # Search for restaurant
            logger.info(f"Searching for restaurant: {args.restaurant}")
            restaurants = search_restaurants(args.restaurant)
            
            if not restaurants:
                logger.error(f"No restaurants found matching '{args.restaurant}'")
                sys.exit(1)
            
            # Use the first result
            restaurant = get_restaurant_by_id(restaurants[0]["id"])
            logger.info(f"Selected restaurant: {restaurant['name']}")
            
            # Fetch reviews
            logger.info("Fetching reviews...")
            reviews = get_reviews_for_restaurant(restaurant["id"], start_date, end_date)
            
            if not reviews:
                logger.error("No reviews found for the selected date range")
                sys.exit(1)
                
            logger.info(f"Found {len(reviews)} reviews")
            
            # Analyze sentiment
            logger.info("Analyzing sentiment...")
            analyzed_reviews = analyze_reviews_batch(reviews)
            
            # Generate summaries
            logger.info("Generating summaries...")
            sentiment_summary = get_sentiment_summary(analyzed_reviews)
            sentiment_trends = get_sentiment_trends(analyzed_reviews)
            
            # LLM analysis
            logger.info("Generating LLM analysis...")
            llm = LLMFactory.get_instance().get_llm()
            llm_analysis = llm.generate_restaurant_analysis(
                restaurant, analyzed_reviews, sentiment_summary, sentiment_trends
            )
            
            # Generate report
            logger.info("Generating PDF report...")
            report_generator = ReportGenerator(output_dir=args.output)
            report_path = report_generator.generate_report(
                restaurant, analyzed_reviews, sentiment_summary,
                sentiment_trends, llm_analysis, start_date, end_date
            )
            
            logger.info(f"Analysis complete! Report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
            sys.exit(1)
    else:
        # Run in GUI mode
        from src.ui.app import main as run_gui
        run_gui()


if __name__ == "__main__":
    main()