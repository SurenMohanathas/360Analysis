"""
Database connection module for interacting with the restaurant reviews database.
"""
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import text

# Load environment variables
load_dotenv("config.env")

# Database connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "restaurant_reviews")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define ORM models (matching the crawler repo schema)
class Restaurant(Base):
    __tablename__ = "restaurants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    address = Column(String(255))
    city = Column(String(100))
    state = Column(String(50))
    postal_code = Column(String(20))
    phone = Column(String(50))
    website = Column(String(255))
    cuisine_type = Column(String(100))
    price_range = Column(String(10))
    average_rating = Column(Float)
    source_url = Column(String(255), unique=True)
    source_id = Column(String(100))
    source_platform = Column(String(50))
    last_updated = Column(DateTime)

    # Relationship with reviews
    reviews = relationship("Review", back_populates="restaurant")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"))
    rating = Column(Float, nullable=False)
    review_text = Column(Text)
    review_date = Column(DateTime)
    reviewer_name = Column(String(255))
    reviewer_id = Column(String(100))
    helpful_count = Column(Integer, default=0)
    source_url = Column(String(255))
    source_id = Column(String(100), unique=True)
    source_platform = Column(String(50))
    crawl_date = Column(DateTime)

    # Relationship with restaurant
    restaurant = relationship("Restaurant", back_populates="reviews")


@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_restaurants() -> List[Dict[str, Any]]:
    """Get a list of all restaurants in the database."""
    with get_db_session() as session:
        restaurants = session.query(Restaurant).all()
        return [
            {
                "id": restaurant.id,
                "name": restaurant.name,
                "address": restaurant.address,
                "city": restaurant.city,
                "cuisine_type": restaurant.cuisine_type,
                "price_range": restaurant.price_range,
                "average_rating": restaurant.average_rating,
                "source_platform": restaurant.source_platform,
            }
            for restaurant in restaurants
        ]


def search_restaurants(name: str) -> List[Dict[str, Any]]:
    """Search for restaurants by name."""
    with get_db_session() as session:
        query = session.query(Restaurant).filter(
            Restaurant.name.ilike(f"%{name}%")
        )
        restaurants = query.all()
        return [
            {
                "id": restaurant.id,
                "name": restaurant.name,
                "address": restaurant.address,
                "city": restaurant.city,
                "cuisine_type": restaurant.cuisine_type,
                "price_range": restaurant.price_range,
                "average_rating": restaurant.average_rating,
                "source_platform": restaurant.source_platform,
            }
            for restaurant in restaurants
        ]


def get_restaurant_by_id(restaurant_id: int) -> Optional[Dict[str, Any]]:
    """Get a restaurant by its ID."""
    with get_db_session() as session:
        restaurant = session.query(Restaurant).filter(Restaurant.id == restaurant_id).first()
        if not restaurant:
            return None
        
        return {
            "id": restaurant.id,
            "name": restaurant.name,
            "address": restaurant.address,
            "city": restaurant.city,
            "state": restaurant.state,
            "postal_code": restaurant.postal_code,
            "phone": restaurant.phone,
            "website": restaurant.website,
            "cuisine_type": restaurant.cuisine_type,
            "price_range": restaurant.price_range,
            "average_rating": restaurant.average_rating,
            "source_platform": restaurant.source_platform,
            "last_updated": restaurant.last_updated,
        }


def get_reviews_for_restaurant(
    restaurant_id: int, 
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Get reviews for a specific restaurant with optional date filtering."""
    with get_db_session() as session:
        query = session.query(Review).filter(Review.restaurant_id == restaurant_id)
        
        if start_date:
            query = query.filter(Review.review_date >= start_date)
            
        if end_date:
            query = query.filter(Review.review_date <= end_date)
            
        reviews = query.order_by(Review.review_date.desc()).all()
        
        return [
            {
                "id": review.id,
                "rating": review.rating,
                "review_text": review.review_text,
                "review_date": review.review_date,
                "reviewer_name": review.reviewer_name,
                "helpful_count": review.helpful_count,
                "source_platform": review.source_platform,
            }
            for review in reviews
        ]


def get_rating_distribution(restaurant_id: int) -> Dict[float, int]:
    """Get rating distribution for a restaurant."""
    with get_db_session() as session:
        reviews = session.query(Review.rating).filter(
            Review.restaurant_id == restaurant_id
        ).all()
        
        # Count occurrences of each rating
        distribution = {}
        for review in reviews:
            rating = review[0]
            if rating in distribution:
                distribution[rating] += 1
            else:
                distribution[rating] = 1
                
        return distribution


def get_reviews_by_platform(restaurant_id: int) -> Dict[str, List[Dict[str, Any]]]:
    """Get reviews for a restaurant grouped by platform."""
    with get_db_session() as session:
        reviews = session.query(Review).filter(
            Review.restaurant_id == restaurant_id
        ).all()
        
        result = {}
        for review in reviews:
            platform = review.source_platform
            if platform not in result:
                result[platform] = []
                
            result[platform].append({
                "id": review.id,
                "rating": review.rating,
                "review_text": review.review_text,
                "review_date": review.review_date,
                "reviewer_name": review.reviewer_name,
                "helpful_count": review.helpful_count,
            })
            
        return result


def get_review_counts_by_month(
    restaurant_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, int]:
    """Get review counts by month for a restaurant."""
    with get_db_session() as session:
        # SQL query to extract month/year and count reviews
        sql = text("""
            SELECT 
                TO_CHAR(review_date, 'YYYY-MM') AS month,
                COUNT(*) as review_count
            FROM 
                reviews
            WHERE 
                restaurant_id = :restaurant_id
                AND (:start_date IS NULL OR review_date >= :start_date)
                AND (:end_date IS NULL OR review_date <= :end_date)
            GROUP BY 
                TO_CHAR(review_date, 'YYYY-MM')
            ORDER BY 
                month
        """)
        
        result = session.execute(
            sql,
            {
                "restaurant_id": restaurant_id,
                "start_date": start_date,
                "end_date": end_date
            }
        )
        
        return {row[0]: row[1] for row in result}