from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
import uvicorn
import logging
from bson import ObjectId
from datetime import datetime, timedelta
from pymongo import DESCENDING

# Import the core search logic and the new company matcher
from on_demand_api import search_with_tiered_threshold # Use the new on-demand logic

import sys
import os

# Add project root to Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.utils.company_matcher import build_matcher
from sentence_transformers import SentenceTransformer

# Define the embedding model to match the indexing script
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'

# --- Pydantic Models for Request and Response Validation ---
class SearchRequest(BaseModel):
    query: str
    limit: int = Field(-1, ge=-1, description="Max articles to return. Set to -1 to get all results above the threshold.")
    sort_by: Literal["similarity", "date"] = Field("similarity", description="Sort results by 'similarity' score or 'date'.")
    content_format: str = Field("full", description="Options: summary, full, both")
    with_link: bool = False
    category: str = "all"
    company: Optional[str] = Field("all", description="Filter by a specific company before semantic search.")
    start_date: Optional[str] = Field(None, description="Start date for the search in YYYY-MM-DD format.")
    duration_days: Optional[int] = Field(365, description="Number of days to search. Used with start_date or counted back from today.")
    custom_threshold: float = Field(-1.0, description="Custom similarity threshold. Must be between 0.0-1.0, or -1.0 to use default logic.")

    @validator('custom_threshold')
    def validate_threshold(cls, v):
        if v == -1.0:
            return v
        if 0.0 <= v <= 1.0:
            return v
        raise ValueError('custom_threshold must be between 0.0 and 1.0, or exactly -1.0')

# --- FastAPI Application ---
app = FastAPI(
    title="Semantic News Search API",
    description="An API for performing semantic search on news articles, with company-first filtering.",
    version="1.2.0"
)

# --- Database, Milvus, and Matcher Connection Management ---
@app.on_event("startup")
async def startup_event():
    from pymilvus import connections, utility, Collection
    from pymongo import MongoClient
    from FYPII_Project.api.on_demand_api import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME

    try:
        # Load the Sentence Transformer model
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        app.state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logging.info("Embedding model loaded successfully.")

        # Build Company Matcher
        app.state.master_matcher, app.state.var_map = build_matcher()
        logging.info("Company matcher built and loaded successfully.")

        # Connect to Milvus
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        if not utility.has_collection(COLLECTION_NAME):
            raise RuntimeError(f"Milvus collection '{COLLECTION_NAME}' not found.")
        
        milvus_collection = Collection(COLLECTION_NAME)
        milvus_collection.load()
        app.state.milvus_collection = milvus_collection
        logging.info("Successfully connected to Milvus and loaded collection.")

        # Connect to MongoDB
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client[MONGO_DB_NAME]
        app.state.mongo_collection = db[MONGO_COLLECTION_NAME]
        app.state.mongo_client = mongo_client
        logging.info("Successfully connected to MongoDB.")

    except Exception as e:
        logging.critical(f"Failed to initialize resources on startup: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
def shutdown_event():
    from pymilvus import connections
    if "default" in connections.list_connections():
        connections.disconnect("default")
    if hasattr(app.state, "mongo_client"):
        app.state.mongo_client.close()
    logging.info("Disconnected from databases.")


# --- API Endpoints ---
@app.post("/search")
async def search(request: SearchRequest):
    """
    Perform a semantic search for news articles based on a query and filters.
    Supports company-first filtering.
    """
    try:
        results = search_with_tiered_threshold(
            milvus_collection=app.state.milvus_collection,
            mongo_collection=app.state.mongo_collection,
            model=app.state.embedding_model, # Pass the loaded model
            query=request.query,
            limit=request.limit,
            sort_by=request.sort_by, # Pass the new sorting parameter
            content_format=request.content_format,
            with_link=request.with_link,
            category=request.category,
            company=request.company,
            start_date=request.start_date,
            duration_days=request.duration_days,
            custom_threshold=request.custom_threshold,
            company_matcher=app.state.master_matcher,
            company_variation_map=app.state.var_map
        )
        return results
    except Exception as e:
        logging.error(f"An error occurred during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during the search process.")

# Helper to convert ObjectId to string and ensure all desired fields are present
def mongo_to_dict(item):
    if "_id" in item and isinstance(item["_id"], ObjectId):
        item["_id"] = str(item["_id"])
    # Ensure company fields are included if they exist, otherwise default to empty lists
    item['companies'] = item.get('companies', [])
    item['tickers'] = item.get('tickers', [])
    return item

@app.get("/all_news")
async def get_all_news(
    skip: int = Query(0, ge=0, description="Number of articles to skip for pagination."),
    limit: Optional[int] = Query(None, ge=1, description="Number of articles to return. If not provided, ALL articles will be returned."),
    company: Optional[str] = Query("all", description="Filter by a specific company name."),
    content_format: Optional[str] = Query("summary", description="Content to return: summary, full, or both."),
    duration_days: Optional[int] = Query(365, description="Filter articles from the last N days."),
    category: Optional[str] = Query("all", description="Filter by a specific category.")
):
    """
    Retrieve news articles from the database with optional filtering.
    """
    try:
        mongo_collection = app.state.mongo_collection
        mongo_filter = {}

        # Category Filter
        if category and category.lower() != 'all':
            mongo_filter['category'] = category

        # Date Filter
        if duration_days is not None and duration_days > 0:
            start_dt = datetime.now() - timedelta(days=duration_days)
            mongo_filter["published_time"] = {"$gte": start_dt.strftime('%Y-%m-%d')}

        # Company Filter
        if company and company.lower() != 'all':
            matcher = app.state.master_matcher
            var_map = app.state.var_map
            match = matcher.search(company.upper())
            if match:
                matched_variation = match.group(0)
                company_data = var_map.get(matched_variation)
                if company_data:
                    mongo_filter['tickers'] = company_data['ticker']
                else:
                    return {"status": f"Company '{company}' could not be resolved.", "total_articles": 0, "data": []}
            else:
                return {"status": f"Company '{company}' not found.", "total_articles": 0, "data": []}

        # Content Projection
        projection = None
        if content_format == "summary":
            projection = {"content": 0}
        elif content_format == "full":
            projection = {"summary": 0}
        
        # Sort by published_time descending (latest first) before skipping and limiting
        cursor = mongo_collection.find(mongo_filter, projection).sort("published_time", DESCENDING).skip(skip)
        if limit is not None:
            cursor = cursor.limit(limit)
        
        results = [mongo_to_dict(doc) for doc in cursor]
        total_count = mongo_collection.count_documents(mongo_filter)

        if not results:
            return {"status": "No articles found for the given filters.", "total_articles": 0, "data": []}
            
        return {"total_articles": total_count, "count": len(results), "data": results}
        
    except Exception as e:
        logging.error(f"An error occurred while fetching all news: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching news articles.")


# --- Root Endpoint for Health Check ---
@app.get("/")
def read_root():
    return {"status": "API is running"}

if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
