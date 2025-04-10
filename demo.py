#!/usr/bin/env python3
"""
Simplified demo script to demonstrate the integration between crawler and 360Analysis.
This version doesn't require UI components or LLM integration.
"""
import os
import sys
import sqlite3
import json
from datetime import datetime, timedelta
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("360Analysis-Demo")

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "demo_output")
os.makedirs(output_dir, exist_ok=True)

# Connect to the SQLite database created by the crawler
db_path = os.path.join(os.path.dirname(__file__), "..", "crawler", "restaurant_reviews.db")
if not os.path.exists(db_path):
    logger.error(f"Database file not found: {db_path}")
    logger.error("Please run the crawler first to populate the database.")
    sys.exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row


def get_restaurants():
    """Get list of restaurants from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, source_platform, average_rating FROM restaurants")
    return [dict(row) for row in cursor.fetchall()]


def get_reviews(restaurant_id):
    """Get reviews for a specific restaurant."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM reviews WHERE restaurant_id = ?",
        (restaurant_id,)
    )
    return [dict(row) for row in cursor.fetchall()]


def analyze_sentiment(text):
    """Simple sentiment analysis using TextBlob."""
    if not text:
        return {"polarity": 0, "sentiment": "neutral"}
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        
    return {
        "polarity": polarity,
        "sentiment": sentiment
    }


def analyze_reviews(reviews):
    """Analyze sentiment of reviews."""
    for review in reviews:
        review["sentiment"] = analyze_sentiment(review["review_text"])
    
    return reviews


def generate_report(restaurant, reviews):
    """Generate a simple report for the restaurant."""
    analyzed_reviews = analyze_reviews(reviews)
    
    # Count sentiments
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    for review in analyzed_reviews:
        sentiments[review["sentiment"]["sentiment"]] += 1
    
    total_reviews = len(analyzed_reviews)
    if total_reviews > 0:
        positive_pct = (sentiments["positive"] / total_reviews) * 100
        neutral_pct = (sentiments["neutral"] / total_reviews) * 100
        negative_pct = (sentiments["negative"] / total_reviews) * 100
    else:
        positive_pct = neutral_pct = negative_pct = 0
    
    # Create a summary
    summary = {
        "restaurant_name": restaurant["name"],
        "source_platform": restaurant["source_platform"],
        "average_rating": restaurant["average_rating"],
        "total_reviews": total_reviews,
        "sentiment_counts": sentiments,
        "sentiment_percentages": {
            "positive": positive_pct,
            "neutral": neutral_pct,
            "negative": negative_pct
        }
    }
    
    # Generate visualization
    if total_reviews > 0:
        generate_sentiment_chart(restaurant["name"], sentiments, output_dir)
    
    return summary


def generate_sentiment_chart(restaurant_name, sentiments, output_dir):
    """Generate a pie chart of sentiment distribution."""
    plt.figure(figsize=(8, 6))
    
    # Create pie chart
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [sentiments["positive"], sentiments["neutral"], sentiments["negative"]]
    colors = ['#4CAF50', '#FFC107', '#F44336']
    explode = (0.1, 0, 0)  # Explode the first slice (Positive)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio
    plt.title(f'Sentiment Distribution for {restaurant_name}')
    
    # Save the chart
    chart_path = os.path.join(output_dir, f"{restaurant_name.replace(' ', '_')}_sentiment.png")
    plt.savefig(chart_path)
    plt.close()
    
    logger.info(f"Sentiment chart saved to {chart_path}")
    return chart_path


def main():
    """Main function to run the demo."""
    logger.info("Starting 360Analysis Demo")
    
    # Get restaurants
    restaurants = get_restaurants()
    if not restaurants:
        logger.error("No restaurants found in the database")
        return
    
    logger.info(f"Found {len(restaurants)} restaurants in the database")
    
    # Analyze each restaurant
    all_reports = []
    for restaurant in restaurants:
        logger.info(f"Analyzing restaurant: {restaurant['name']} (Platform: {restaurant['source_platform']})")
        
        # Get reviews
        reviews = get_reviews(restaurant["id"])
        logger.info(f"Found {len(reviews)} reviews")
        
        # Generate report
        report = generate_report(restaurant, reviews)
        all_reports.append(report)
        
        # Save report to JSON
        report_path = os.path.join(output_dir, f"{restaurant['name'].replace(' ', '_')}_{restaurant['source_platform']}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
    
    # Create a combined report
    combined_report = {
        "total_restaurants": len(all_reports),
        "restaurants": all_reports
    }
    
    combined_report_path = os.path.join(output_dir, "combined_report.json")
    with open(combined_report_path, 'w') as f:
        json.dump(combined_report, f, indent=2)
    
    logger.info(f"Combined report saved to {combined_report_path}")
    
    # Generate comparison visualization
    if all_reports:
        generate_comparison_chart(all_reports, output_dir)
    
    logger.info("Demo completed successfully")


def generate_comparison_chart(reports, output_dir):
    """Generate a bar chart comparing sentiment across restaurants."""
    # Create dataframe for plotting
    data = []
    for report in reports:
        data.append({
            'restaurant': f"{report['restaurant_name']} ({report['source_platform']})",
            'positive': report['sentiment_percentages']['positive'],
            'neutral': report['sentiment_percentages']['neutral'],
            'negative': report['sentiment_percentages']['negative']
        })
    
    df = pd.DataFrame(data)
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    
    # Reshape data for stacked bar chart
    df_plot = pd.DataFrame({
        'Restaurant': df['restaurant'],
        'Positive': df['positive'],
        'Neutral': df['neutral'],
        'Negative': df['negative']
    })
    
    df_plot.set_index('Restaurant', inplace=True)
    
    # Create stacked bar chart
    ax = df_plot.plot(kind='bar', stacked=True, figsize=(12, 6), 
                     color=['#4CAF50', '#FFC107', '#F44336'])
    
    plt.title('Sentiment Comparison Across Restaurants')
    plt.xlabel('Restaurant')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title='Sentiment')
    
    # Save the chart
    chart_path = os.path.join(output_dir, "restaurant_comparison.png")
    plt.savefig(chart_path)
    plt.close()
    
    logger.info(f"Comparison chart saved to {chart_path}")
    return chart_path


if __name__ == "__main__":
    main()
    conn.close()