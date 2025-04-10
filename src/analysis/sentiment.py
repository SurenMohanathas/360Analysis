"""
Sentiment analysis module for restaurant reviews.
"""
import os
from typing import Dict, List, Any, Optional, Tuple
import statistics
from datetime import datetime

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download NLTK resources if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


def get_textblob_sentiment(text: str) -> Dict[str, float]:
    """Get sentiment analysis using TextBlob."""
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    # Convert to a normalized score between -1 and 1
    polarity = sentiment.polarity
    
    # Classify as positive, negative, or neutral
    if polarity > 0.05:
        label = "positive"
    elif polarity < -0.05:
        label = "negative"
    else:
        label = "neutral"
    
    return {
        "polarity": polarity,
        "subjectivity": sentiment.subjectivity,
        "label": label
    }


def get_vader_sentiment(text: str) -> Dict[str, float]:
    """Get sentiment analysis using VADER."""
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    
    # Classify as positive, negative, or neutral
    if sentiment["compound"] > 0.05:
        label = "positive"
    elif sentiment["compound"] < -0.05:
        label = "negative"
    else:
        label = "neutral"
    
    return {
        "negative": sentiment["neg"],
        "neutral": sentiment["neu"],
        "positive": sentiment["pos"],
        "compound": sentiment["compound"],
        "label": label
    }


def analyze_review_sentiment(review_text: str) -> Dict[str, Any]:
    """Analyze sentiment of a single review."""
    if not review_text or len(review_text.strip()) == 0:
        return {
            "textblob": {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "label": "neutral"
            },
            "vader": {
                "negative": 0.0,
                "neutral": 1.0,
                "positive": 0.0,
                "compound": 0.0,
                "label": "neutral"
            },
            "combined_label": "neutral"
        }
    
    # Get sentiment from multiple analyzers
    textblob_sentiment = get_textblob_sentiment(review_text)
    vader_sentiment = get_vader_sentiment(review_text)
    
    # Combine sentiment labels (simple majority)
    labels = [textblob_sentiment["label"], vader_sentiment["label"]]
    positive_count = labels.count("positive")
    negative_count = labels.count("negative")
    neutral_count = labels.count("neutral")
    
    if positive_count > negative_count and positive_count > neutral_count:
        combined_label = "positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        combined_label = "negative"
    else:
        combined_label = "neutral"
    
    return {
        "textblob": textblob_sentiment,
        "vader": vader_sentiment,
        "combined_label": combined_label
    }


def analyze_reviews_batch(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze sentiment for a batch of reviews."""
    analyzed_reviews = []
    
    for review in reviews:
        sentiment = analyze_review_sentiment(review["review_text"])
        
        # Add sentiment analysis to the review
        review_with_sentiment = {
            **review,
            "sentiment": sentiment
        }
        
        analyzed_reviews.append(review_with_sentiment)
        
    return analyzed_reviews


def get_sentiment_summary(analyzed_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get summary statistics for sentiment analysis."""
    if not analyzed_reviews:
        return {
            "total_reviews": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "positive_percentage": 0,
            "negative_percentage": 0,
            "neutral_percentage": 0,
            "average_polarity": 0,
            "average_subjectivity": 0,
            "average_compound": 0
        }
    
    # Count sentiment labels
    positive_count = sum(1 for r in analyzed_reviews if r["sentiment"]["combined_label"] == "positive")
    negative_count = sum(1 for r in analyzed_reviews if r["sentiment"]["combined_label"] == "negative")
    neutral_count = sum(1 for r in analyzed_reviews if r["sentiment"]["combined_label"] == "neutral")
    
    total_reviews = len(analyzed_reviews)
    
    # Calculate percentages
    positive_percentage = (positive_count / total_reviews) * 100 if total_reviews > 0 else 0
    negative_percentage = (negative_count / total_reviews) * 100 if total_reviews > 0 else 0
    neutral_percentage = (neutral_count / total_reviews) * 100 if total_reviews > 0 else 0
    
    # Calculate averages
    try:
        average_polarity = statistics.mean([r["sentiment"]["textblob"]["polarity"] for r in analyzed_reviews])
    except statistics.StatisticsError:
        average_polarity = 0
        
    try:
        average_subjectivity = statistics.mean([r["sentiment"]["textblob"]["subjectivity"] for r in analyzed_reviews])
    except statistics.StatisticsError:
        average_subjectivity = 0
        
    try:
        average_compound = statistics.mean([r["sentiment"]["vader"]["compound"] for r in analyzed_reviews])
    except statistics.StatisticsError:
        average_compound = 0
    
    return {
        "total_reviews": total_reviews,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "positive_percentage": positive_percentage,
        "negative_percentage": negative_percentage,
        "neutral_percentage": neutral_percentage,
        "average_polarity": average_polarity,
        "average_subjectivity": average_subjectivity,
        "average_compound": average_compound
    }


def get_sentiment_by_platform(analyzed_reviews: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Get sentiment summary grouped by platform."""
    # Group reviews by platform
    reviews_by_platform = {}
    
    for review in analyzed_reviews:
        platform = review["source_platform"]
        if platform not in reviews_by_platform:
            reviews_by_platform[platform] = []
        
        reviews_by_platform[platform].append(review)
    
    # Get sentiment summary for each platform
    sentiment_by_platform = {}
    
    for platform, reviews in reviews_by_platform.items():
        sentiment_by_platform[platform] = get_sentiment_summary(reviews)
    
    return sentiment_by_platform


def get_sentiment_trends(
    analyzed_reviews: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Get sentiment trends over time."""
    if not analyzed_reviews:
        return {
            "monthly_sentiment": [],
            "overall_trend": "stable"
        }
    
    # Sort reviews by date
    sorted_reviews = sorted(
        [r for r in analyzed_reviews if r.get("review_date")],
        key=lambda x: x["review_date"]
    )
    
    if not sorted_reviews:
        return {
            "monthly_sentiment": [],
            "overall_trend": "stable"
        }
    
    # Group reviews by month
    reviews_by_month = {}
    
    for review in sorted_reviews:
        date = review["review_date"]
        if not date:
            continue
            
        month_key = date.strftime("%Y-%m")
        
        if month_key not in reviews_by_month:
            reviews_by_month[month_key] = []
            
        reviews_by_month[month_key].append(review)
    
    # Calculate sentiment for each month
    monthly_sentiment = []
    
    for month, reviews in sorted(reviews_by_month.items()):
        sentiment_summary = get_sentiment_summary(reviews)
        monthly_sentiment.append({
            "month": month,
            "review_count": len(reviews),
            "average_rating": sum(r["rating"] for r in reviews) / len(reviews) if reviews else 0,
            "positive_percentage": sentiment_summary["positive_percentage"],
            "negative_percentage": sentiment_summary["negative_percentage"],
            "average_compound": sentiment_summary["average_compound"]
        })
    
    # Determine overall trend
    if len(monthly_sentiment) <= 1:
        overall_trend = "stable"
    else:
        # Compare first and last month for a simple trend
        first_month = monthly_sentiment[0]
        last_month = monthly_sentiment[-1]
        
        if last_month["average_compound"] > first_month["average_compound"] + 0.1:
            overall_trend = "improving"
        elif last_month["average_compound"] < first_month["average_compound"] - 0.1:
            overall_trend = "declining"
        else:
            overall_trend = "stable"
    
    return {
        "monthly_sentiment": monthly_sentiment,
        "overall_trend": overall_trend
    }