from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Optional, Literal, List, Dict, Any
import uvicorn
import logging
import asyncio
import aiohttp
from bson import ObjectId
from datetime import datetime, timedelta
from pymongo import DESCENDING
import uuid
from collections import defaultdict
import time

# Import the core search logic from api_demo.py
from on_demand_api import search_with_tiered_threshold

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
# classify model: "google/gemma-3-27b-it:free"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models for Request and Response Validation ---

class SearchRequest(BaseModel):
    """Base search request model used by both search and subscription"""
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


class SubscriptionRequest(SearchRequest):
    """Extended model for subscription requests"""
    callback_url: HttpUrl = Field(..., description="URL to send matching news updates")
    delivery_interval_seconds: int = Field(30, ge=1, description="How often to push updates in seconds")
    initial_full_dump: bool = Field(True, description="Whether to push all matching historical news right after subscribing")
    active: bool = Field(True, description="Whether the subscription is active")
    
    # Additional fields for subscription management
    subscription_name: Optional[str] = Field(None, description="Optional friendly name for the subscription")
    max_deliveries: Optional[int] = Field(None, ge=1, description="Maximum number of deliveries before auto-unsubscribe")
    expire_at: Optional[datetime] = Field(None, description="When the subscription should automatically expire")


class SubscriptionResponse(BaseModel):
    """Response model for subscription operations"""
    subscription_id: str
    status: str
    message: str
    subscription_details: Optional[Dict[str, Any]] = None


class DeliveryStatus(BaseModel):
    """Track delivery status for each subscription"""
    last_delivered: Optional[datetime] = None
    total_deliveries: int = 0
    last_article_id: Optional[str] = None
    pending_articles: List[Dict] = Field(default_factory=list)


# --- FastAPI Application ---
app = FastAPI(
    title="Semantic News Search API with Subscription System",
    description="An API for performing semantic search on news articles with live subscription-based delivery.",
    version="2.0.0"
)

# --- In-memory subscription management (will be backed by MongoDB) ---
class SubscriptionManager:
    def __init__(self, mongo_collection):
        self.mongo_collection = mongo_collection
        self.delivery_status = defaultdict(DeliveryStatus)
        self.delivery_tasks = {}
        
    async def add_subscription(self, subscription: SubscriptionRequest) -> str:
        """Add a new subscription to MongoDB"""
        sub_id = str(uuid.uuid4())
        sub_dict = subscription.dict()
        sub_dict['subscription_id'] = sub_id
        sub_dict['created_at'] = datetime.utcnow()
        sub_dict['updated_at'] = datetime.utcnow()
        sub_dict['delivery_count'] = 0
        
        # Convert HttpUrl to string for MongoDB storage
        sub_dict['callback_url'] = str(subscription.callback_url)
        
        await asyncio.get_event_loop().run_in_executor(
            None, self.mongo_collection.insert_one, sub_dict
        )
        
        logger.info(f"✅ Subscription added: {sub_id} for query: '{subscription.query}'")
        return sub_id
    
    async def remove_subscription(self, subscription_id: str = None, callback_url: str = None) -> bool:
        """Remove subscription from MongoDB"""
        filter_query = {}
        if subscription_id:
            filter_query['subscription_id'] = subscription_id
        elif callback_url:
            filter_query['callback_url'] = callback_url
        else:
            return False
        
        result = await asyncio.get_event_loop().run_in_executor(
            None, self.mongo_collection.delete_one, filter_query
        )
        
        if result.deleted_count > 0:
            # Cancel any running delivery tasks
            if subscription_id in self.delivery_tasks:
                self.delivery_tasks[subscription_id].cancel()
                del self.delivery_tasks[subscription_id]
            
            logger.info(f"🗑️ Subscription removed: {subscription_id or callback_url}")
            return True
        return False
    
    async def get_active_subscriptions(self) -> List[Dict]:
        """Get all active subscriptions from MongoDB"""
        cursor = self.mongo_collection.find({'active': True})
        subscriptions = await asyncio.get_event_loop().run_in_executor(
            None, lambda: list(cursor)
        )
        
        # Convert ObjectId to string and clean up for response
        for sub in subscriptions:
            if '_id' in sub:
                sub['_id'] = str(sub['_id'])
        
        return subscriptions
    
    async def update_subscription_status(self, subscription_id: str, updates: Dict):
        """Update subscription status in MongoDB"""
        updates['updated_at'] = datetime.utcnow()
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.mongo_collection.update_one(
                {'subscription_id': subscription_id},
                {'$set': updates}
            )
        )


# --- Background delivery system ---
async def deliver_news_batch(subscription: Dict, articles: List[Dict], app_state):
    """Deliver a batch of articles to a subscription's callback URL"""
    try:
        payload = {
            'subscription_id': subscription['subscription_id'],
            'timestamp': datetime.utcnow().isoformat(),
            'article_count': len(articles),
            'articles': articles
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                subscription['callback_url'],
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    logger.info(f"📤 Delivered {len(articles)} articles to {subscription['callback_url']}")
                    
                    # Update delivery count
                    app_state.subscription_manager.mongo_collection.update_one(
                        {'subscription_id': subscription['subscription_id']},
                        {
                            '$inc': {'delivery_count': 1},
                            '$set': {'last_delivery': datetime.utcnow()}
                        }
                    )
                    
                    # Check if max deliveries reached
                    if subscription.get('max_deliveries'):
                        current_count = subscription.get('delivery_count', 0) + 1
                        if current_count >= subscription['max_deliveries']:
                            await app_state.subscription_manager.update_subscription_status(
                                subscription['subscription_id'],
                                {'active': False, 'deactivation_reason': 'max_deliveries_reached'}
                            )
                            logger.info(f"🛑 Subscription {subscription['subscription_id']} auto-deactivated: max deliveries reached")
                else:
                    logger.error(f"❌ Delivery failed with status {response.status} for {subscription['callback_url']}")
                    
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Delivery timeout for {subscription['callback_url']}")
    except Exception as e:
        logger.error(f"❌ Delivery error for {subscription['callback_url']}: {e}")


async def check_and_deliver_news(app_state):
    """Background task to check for new news and deliver to subscribers"""
    while True:
        try:
            # Get all active subscriptions
            subscriptions = await app_state.subscription_manager.get_active_subscriptions()
            
            for subscription in subscriptions:
                try:
                    # Check if subscription has expired
                    if subscription.get('expire_at') and datetime.utcnow() > subscription['expire_at']:
                        await app_state.subscription_manager.update_subscription_status(
                            subscription['subscription_id'],
                            {'active': False, 'deactivation_reason': 'expired'}
                        )
                        logger.info(f"⏰ Subscription {subscription['subscription_id']} expired")
                        continue
                    
                    # Get delivery status
                    sub_id = subscription['subscription_id']
                    status = app_state.subscription_manager.delivery_status[sub_id]
                    
                    # Check if it's time to deliver
                    if status.last_delivered:
                        time_since_last = (datetime.utcnow() - status.last_delivered).total_seconds()
                        if time_since_last < subscription['delivery_interval_seconds']:
                            # Even if not time for regular delivery, check for pending articles
                            if not status.pending_articles:
                                continue
                    
                    articles_to_deliver = []
                    
                    # First, check if there are pending articles from the webhook
                    if status.pending_articles:
                        articles_to_deliver.extend(status.pending_articles)
                        status.pending_articles = []  # Clear pending articles
                        logger.info(f"📦 Found {len(articles_to_deliver)} pending articles for {sub_id}")
                    
                    # If it's time for regular check or initial dump, search for articles
                    should_search = (
                        not status.last_delivered or  # First delivery
                        (datetime.utcnow() - status.last_delivered).total_seconds() >= subscription['delivery_interval_seconds'] or
                        subscription.get('initial_full_dump', False)
                    )
                    
                    if should_search:
                        # Search for matching articles using full semantic search
                        search_params = {k: v for k, v in subscription.items() 
                                       if k in SearchRequest.__fields__}
                        
                        # If not initial dump, only get new articles since last delivery
                        if status.last_delivered and not subscription.get('initial_full_dump'):
                            search_params['start_date'] = status.last_delivered.strftime('%Y-%m-%d')
                        
                        results = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: search_with_tiered_threshold(
                                milvus_collection=app_state.milvus_collection,
                                mongo_collection=app_state.mongo_collection,
                                model=app_state.embedding_model,
                                company_matcher=app_state.master_matcher,
                                company_variation_map=app_state.var_map,
                                **search_params
                            )
                        )
                        
                        if results and results.get('results'):  # Note: changed from 'data' to 'results'
                            search_articles = results['results']
                            
                            # Filter out already delivered articles if tracking
                            if status.last_article_id:
                                new_articles = []
                                for article in search_articles:
                                    article_id = str(article.get('mongo_id', article.get('_id')))
                                    if article_id == status.last_article_id:
                                        break
                                    # Avoid duplicates with pending articles
                                    if not any(a.get('_id') == article_id or a.get('mongo_id') == article_id 
                                             for a in articles_to_deliver):
                                        new_articles.append(article)
                                articles_to_deliver.extend(new_articles)
                            else:
                                # Avoid duplicates when combining with pending articles
                                existing_ids = {a.get('_id', a.get('mongo_id')) for a in articles_to_deliver}
                                for article in search_articles:
                                    article_id = article.get('mongo_id', article.get('_id'))
                                    if article_id not in existing_ids:
                                        articles_to_deliver.append(article)
                    
                    # Deliver if we have articles
                    if articles_to_deliver:
                        # Apply limit if specified
                        if subscription.get('limit', -1) != -1:
                            articles_to_deliver = articles_to_deliver[:subscription['limit']]
                        
                        # Deliver the news batch
                        await deliver_news_batch(subscription, articles_to_deliver, app_state)
                        
                        # Update delivery status
                        status.last_delivered = datetime.utcnow()
                        status.total_deliveries += 1
                        if articles_to_deliver:
                            # Track the most recent article ID
                            first_article = articles_to_deliver[0]
                            status.last_article_id = str(first_article.get('mongo_id', first_article.get('_id')))
                        
                        # Reset initial_full_dump flag after first delivery
                        if subscription.get('initial_full_dump'):
                            await app_state.subscription_manager.update_subscription_status(
                                sub_id, {'initial_full_dump': False}
                            )
                    
                except Exception as e:
                    logger.error(f"Error processing subscription {subscription['subscription_id']}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in check_and_deliver_news: {e}")
        
        # Sleep before next check cycle
        await asyncio.sleep(5)  # Check every 5 seconds


# --- Database, Milvus, and Matcher Connection Management ---
@app.on_event("startup")
async def startup_event():
    from pymilvus import connections, utility, Collection
    from pymongo import MongoClient
    from FYPII_Project.api.on_demand_api import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME

    try:
        # Load the Sentence Transformer model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        app.state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")

        # Build Company Matcher
        app.state.master_matcher, app.state.var_map = build_matcher()
        logger.info("Company matcher built and loaded successfully.")

        # Connect to Milvus
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        if not utility.has_collection(COLLECTION_NAME):
            raise RuntimeError(f"Milvus collection '{COLLECTION_NAME}' not found.")
        
        milvus_collection = Collection(COLLECTION_NAME)
        milvus_collection.load()
        app.state.milvus_collection = milvus_collection
        logger.info("Successfully connected to Milvus and loaded collection.")

        # Connect to MongoDB
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client[MONGO_DB_NAME]
        app.state.mongo_collection = db[MONGO_COLLECTION_NAME]
        app.state.mongo_client = mongo_client
        
        # Create subscriptions collection
        app.state.subscriptions_collection = db['subscriptions']
        # Create index on subscription_id for faster lookups
        app.state.subscriptions_collection.create_index('subscription_id', unique=True)
        app.state.subscriptions_collection.create_index('callback_url')
        app.state.subscriptions_collection.create_index('active')
        
        logger.info("Successfully connected to MongoDB.")
        
        # Initialize subscription manager
        app.state.subscription_manager = SubscriptionManager(app.state.subscriptions_collection)
        
        # Start background task for checking and delivering news
        app.state.delivery_task = asyncio.create_task(check_and_deliver_news(app.state))
        logger.info("Background delivery task started.")

    except Exception as e:
        logger.critical(f"Failed to initialize resources on startup: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    from pymilvus import connections
    
    # Cancel background task
    if hasattr(app.state, 'delivery_task'):
        app.state.delivery_task.cancel()
        try:
            await app.state.delivery_task
        except asyncio.CancelledError:
            pass
    
    if "default" in connections.list_connections():
        connections.disconnect("default")
    if hasattr(app.state, "mongo_client"):
        app.state.mongo_client.close()
    logger.info("Disconnected from databases and stopped background tasks.")


# --- API Endpoints ---

@app.post("/search")
async def search(request: SearchRequest):
    """
    Perform a semantic search for news articles based on a query and filters.
    This is the main user-facing search endpoint.
    Supports company-first filtering and returns articles ranked by semantic similarity.
    """
    try:
        # Use the imported search function from api_demo.py
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_with_tiered_threshold(
                milvus_collection=app.state.milvus_collection,
                mongo_collection=app.state.mongo_collection,
                model=app.state.embedding_model,
                query=request.query,
                limit=request.limit,
                sort_by=request.sort_by,
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
        )
        
        # Transform the response to match expected format
        if results and 'results' in results:
            return {
                "status": "success",
                "search_metadata": results.get('search_metadata', {}),
                "count": len(results['results']),
                "data": results['results']
            }
        else:
            return {
                "status": "no_results",
                "search_metadata": results.get('search_metadata', {}),
                "count": 0,
                "data": []
            }
            
    except Exception as e:
        logger.error(f"An error occurred during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during the search process.")


@app.post("/subscribe", response_model=SubscriptionResponse)
async def subscribe(request: SubscriptionRequest, background_tasks: BackgroundTasks):
    """
    Register a new subscription for live news delivery.
    Will immediately deliver all matching historical news if initial_full_dump is True.
    """
    try:
        # Add subscription to database
        subscription_id = await app.state.subscription_manager.add_subscription(request)
        
        # If initial_full_dump is True, immediately search and deliver existing matches
        if request.initial_full_dump:
            logger.info(f"📦 Performing initial full dump for subscription {subscription_id}")
            
            # Perform immediate search
            search_params = {k: v for k, v in request.dict().items() 
                           if k in SearchRequest.__fields__}
            
            results = search_with_tiered_threshold(
                milvus_collection=app.state.milvus_collection,
                mongo_collection=app.state.mongo_collection,
                model=app.state.embedding_model,
                company_matcher=app.state.master_matcher,
                company_variation_map=app.state.var_map,
                **search_params
            )
            
            if results and results.get('data'):
                # Schedule immediate delivery in background
                subscription_dict = request.dict()
                subscription_dict['subscription_id'] = subscription_id
                subscription_dict['callback_url'] = str(request.callback_url)
                subscription_dict['delivery_count'] = 0
                
                background_tasks.add_task(
                    deliver_news_batch,
                    subscription_dict,
                    results['data'],
                    app.state
                )
                
                message = f"Subscription created with ID {subscription_id}. Initial dump of {len(results['data'])} articles scheduled."
            else:
                message = f"Subscription created with ID {subscription_id}. No matching articles found for initial dump."
        else:
            message = f"Subscription created with ID {subscription_id}. Will check for new articles every {request.delivery_interval_seconds} seconds."
        
        return SubscriptionResponse(
            subscription_id=subscription_id,
            status="success",
            message=message,
            subscription_details={
                "query": request.query,
                "callback_url": str(request.callback_url),
                "delivery_interval_seconds": request.delivery_interval_seconds,
                "filters": {
                    "company": request.company,
                    "category": request.category,
                    "duration_days": request.duration_days
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create subscription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create subscription: {str(e)}")


@app.post("/unsubscribe", response_model=SubscriptionResponse)
async def unsubscribe(
    subscription_id: Optional[str] = None,
    callback_url: Optional[str] = None
):
    """
    Remove an active subscription by ID or callback URL.
    At least one identifier must be provided.
    """
    if not subscription_id and not callback_url:
        raise HTTPException(
            status_code=400, 
            detail="Either subscription_id or callback_url must be provided"
        )
    
    success = await app.state.subscription_manager.remove_subscription(
        subscription_id=subscription_id,
        callback_url=callback_url
    )
    
    if success:
        return SubscriptionResponse(
            subscription_id=subscription_id or "N/A",
            status="success",
            message=f"Subscription removed successfully"
        )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Subscription not found with {'ID: ' + subscription_id if subscription_id else 'URL: ' + callback_url}"
        )


@app.get("/subscriptions")
async def list_subscriptions(
    active_only: bool = Query(True, description="Show only active subscriptions"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of subscriptions to return")
):
    """
    List current subscriptions with their details.
    """
    try:
        filter_query = {'active': True} if active_only else {}
        cursor = app.state.subscriptions_collection.find(filter_query).limit(limit)
        subscriptions = list(cursor)
        
        # Clean up response
        for sub in subscriptions:
            if '_id' in sub:
                del sub['_id']  # Remove MongoDB internal ID
            # Add delivery status if available
            if sub['subscription_id'] in app.state.subscription_manager.delivery_status:
                status = app.state.subscription_manager.delivery_status[sub['subscription_id']]
                sub['last_delivered'] = status.last_delivered.isoformat() if status.last_delivered else None
                sub['total_deliveries'] = status.total_deliveries
        
        return {
            "total_subscriptions": len(subscriptions),
            "subscriptions": subscriptions
        }
        
    except Exception as e:
        logger.error(f"Failed to list subscriptions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve subscriptions")


@app.put("/subscription/{subscription_id}/pause")
async def pause_subscription(subscription_id: str):
    """Pause an active subscription"""
    result = await app.state.subscription_manager.update_subscription_status(
        subscription_id, {'active': False, 'paused': True}
    )
    
    if result:
        return {"status": "success", "message": f"Subscription {subscription_id} paused"}
    else:
        raise HTTPException(status_code=404, detail="Subscription not found")


@app.put("/subscription/{subscription_id}/resume")
async def resume_subscription(subscription_id: str):
    """Resume a paused subscription"""
    result = await app.state.subscription_manager.update_subscription_status(
        subscription_id, {'active': True, 'paused': False}
    )
    
    if result:
        return {"status": "success", "message": f"Subscription {subscription_id} resumed"}
    else:
        raise HTTPException(status_code=404, detail="Subscription not found")


# Helper to convert ObjectId to string and ensure all desired fields are present
def mongo_to_dict(item):
    if "_id" in item and isinstance(item["_id"], ObjectId):
        item["_id"] = str(item["_id"])
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
    This is a user-facing endpoint for browsing all news without semantic search.
    Returns articles sorted by published_time (newest first).
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
                    return {
                        "status": f"Company '{company}' could not be resolved.",
                        "total_articles": 0,
                        "count": 0,
                        "data": []
                    }
            else:
                return {
                    "status": f"Company '{company}' not found.",
                    "total_articles": 0,
                    "count": 0,
                    "data": []
                }

        # Content Projection based on format
        projection = None
        if content_format == "summary":
            projection = {"content": 0}
        elif content_format == "full":
            projection = {"summary": 0}
        # If "both", no projection needed
        
        # Execute query with sorting by date (newest first)
        cursor = mongo_collection.find(mongo_filter, projection).sort("published_time", DESCENDING).skip(skip)
        if limit is not None:
            cursor = cursor.limit(limit)
        
        # Convert results to list with proper formatting
        results = [mongo_to_dict(doc) for doc in cursor]
        
        # Get total count for pagination info
        total_count = mongo_collection.count_documents(mongo_filter)

        if not results:
            return {
                "status": "No articles found for the given filters.",
                "total_articles": 0,
                "count": 0,
                "data": []
            }
            
        return {
            "status": "success",
            "total_articles": total_count,
            "count": len(results),
            "data": results,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "has_more": (skip + len(results)) < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"An error occurred while fetching all news: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching news articles.")


# --- Webhook endpoint for scraper integration ---
async def check_article_matches_subscription(article_id: str, subscription: Dict, app_state) -> Dict:
    """
    Check if a specific article matches a subscription using semantic search.
    Returns the article with similarity score if it matches, None otherwise.
    """
    try:
        # Build the search filter to only search this specific article
        id_filter = f'mongo_id == "{article_id}"'
        
        # Get subscription search parameters
        search_params = {k: v for k, v in subscription.items() 
                       if k in SearchRequest.__fields__}
        
        # Perform semantic search on just this article
        query_embedding = app_state.embedding_model.encode(subscription['query']).tolist()
        
        # Search parameters for single article check
        milvus_search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
        
        # Search for this specific article with the subscription's query
        results = app_state.milvus_collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=milvus_search_params,
            limit=10,  # Get multiple chunks from the same article
            expr=id_filter,
            output_fields=["mongo_id", "title", "chunk_text"]
        )
        
        if not results or not results[0]:
            return None
        
        # Get the best matching chunk's score
        best_score = max([hit.distance for hit in results[0]]) if results[0] else 0
        
        # Determine threshold (using same logic as search_with_tiered_threshold)
        final_threshold = subscription.get('custom_threshold', -1.0)
        if final_threshold == -1.0:
            if best_score >= 0.50:
                final_threshold = 0.50
            elif best_score >= 0.40:
                final_threshold = 0.45
            else:
                final_threshold = 0.38
        
        # Check if article meets the threshold
        if best_score < final_threshold:
            logger.debug(f"Article {article_id} score {best_score:.3f} below threshold {final_threshold:.3f} for subscription {subscription['subscription_id']}")
            return None
        
        # Fetch full article from MongoDB
        article = app_state.mongo_collection.find_one({'_id': ObjectId(article_id)})
        if not article:
            return None
        
        # Convert to dict
        article = mongo_to_dict(article)
        
        # Apply additional filters (company, category, date)
        # Company filter
        if subscription.get('company') and subscription['company'].lower() != 'all':
            matcher = app_state.master_matcher
            var_map = app_state.var_map
            match = matcher.search(subscription['company'].upper())
            if match:
                matched_variation = match.group(0)
                company_data = var_map.get(matched_variation)
                if company_data:
                    if company_data['ticker'] not in article.get('tickers', []):
                        logger.debug(f"Article {article_id} filtered out by company filter")
                        return None
        
        # Category filter
        if subscription.get('category') and subscription['category'].lower() != 'all':
            if subscription['category'] != article.get('category'):
                logger.debug(f"Article {article_id} filtered out by category filter")
                return None
        
        # Date filter
        if subscription.get('start_date') or subscription.get('duration_days'):
            from datetime import datetime, timedelta
            today = datetime.now()
            
            if subscription.get('start_date'):
                start_dt = datetime.strptime(subscription['start_date'], '%Y-%m-%d')
                duration_days = subscription.get('duration_days', 365)
                end_dt = start_dt + timedelta(days=duration_days)
            else:
                duration_days = subscription.get('duration_days', 365)
                end_dt = today
                start_dt = end_dt - timedelta(days=duration_days)
            
            article_date = datetime.strptime(article['published_time'], '%Y-%m-%d')
            if not (start_dt <= article_date <= min(end_dt, today)):
                logger.debug(f"Article {article_id} filtered out by date filter")
                return None
        
        # Add similarity score to article
        article['similarity_score'] = best_score
        
        # Format content based on subscription preferences
        content_format = subscription.get('content_format', 'full')
        if content_format == 'summary':
            article.pop('content', None)
        elif content_format == 'full':
            article.pop('summary', None)
        # If 'both', keep both fields
        
        if not subscription.get('with_link', False):
            article.pop('link', None)
        
        logger.info(f"✅ Article {article_id} matches subscription {subscription['subscription_id']} with score {best_score:.3f}")
        return article
        
    except Exception as e:
        logger.error(f"Error checking article {article_id} against subscription: {e}")
        return None


async def check_article_matches_basic(article_id: str, subscription: Dict, app_state) -> Dict:
    """
    Basic filtering check for article matching (without semantic search).
    Used for quick filtering based on metadata only.
    """
    try:
        # Fetch article from MongoDB
        article = app_state.mongo_collection.find_one({'_id': ObjectId(article_id)})
        if not article:
            return None
        
        # Convert to dict
        article = mongo_to_dict(article)
        
        # Apply filters (company, category, date)
        # Company filter
        if subscription.get('company') and subscription['company'].lower() != 'all':
            matcher = app_state.master_matcher
            var_map = app_state.var_map
            match = matcher.search(subscription['company'].upper())
            if match:
                matched_variation = match.group(0)
                company_data = var_map.get(matched_variation)
                if company_data:
                    if company_data['ticker'] not in article.get('tickers', []):
                        return None
        
        # Category filter
        if subscription.get('category') and subscription['category'].lower() != 'all':
            if subscription['category'] != article.get('category'):
                return None
        
        # Date filter
        if subscription.get('start_date') or subscription.get('duration_days'):
            from datetime import datetime, timedelta
            today = datetime.now()
            
            if subscription.get('start_date'):
                start_dt = datetime.strptime(subscription['start_date'], '%Y-%m-%d')
                duration_days = subscription.get('duration_days', 365)
                end_dt = start_dt + timedelta(days=duration_days)
            else:
                duration_days = subscription.get('duration_days', 365)
                end_dt = today
                start_dt = end_dt - timedelta(days=duration_days)
            
            article_date = datetime.strptime(article['published_time'], '%Y-%m-%d')
            if not (start_dt <= article_date <= min(end_dt, today)):
                return None
        
        # Format content based on subscription preferences
        content_format = subscription.get('content_format', 'full')
        if content_format == 'summary':
            article.pop('content', None)
        elif content_format == 'full':
            article.pop('summary', None)
        
        if not subscription.get('with_link', False):
            article.pop('link', None)
        
        return article
        
    except Exception as e:
        logger.error(f"Error in basic check for article {article_id}: {e}")
        return None


class WebhookRequest(BaseModel):
    """Request model for the news scraping webhook"""
    article_ids: List[str] = Field(..., description="List of newly scraped article IDs")
    immediate_delivery: bool = Field(False, description="Force immediate delivery instead of batching")
    use_semantic_matching: bool = Field(True, description="Use semantic search for matching (slower but more accurate)")


@app.post("/news_scraped_webhook")
async def news_scraped_webhook(request: WebhookRequest):
    """
    Webhook to be called by the scraper when new articles are added.
    Can use either semantic search or basic filtering for matching.
    
    Args:
        article_ids: List of MongoDB ObjectId strings for new articles
        immediate_delivery: If True, deliver matches immediately instead of queuing
        use_semantic_matching: If True, use full semantic search; if False, use basic metadata filtering only
    """
    try:
        logger.info(f"📰 Received notification of {len(request.article_ids)} new articles (semantic={request.use_semantic_matching})")
        
        # Get all active subscriptions
        subscriptions = await app.state.subscription_manager.get_active_subscriptions()
        
        if not subscriptions:
            logger.info("No active subscriptions to process")
            return {
                "status": "success",
                "processed_articles": len(request.article_ids),
                "total_matches": 0,
                "subscriptions_with_matches": 0,
                "immediate_delivery": request.immediate_delivery,
                "semantic_matching": request.use_semantic_matching
            }
        
        # Track matches per subscription
        subscription_matches = defaultdict(list)
        total_matches = 0
        
        # Process each subscription
        for subscription in subscriptions:
            logger.debug(f"Checking subscription {subscription['subscription_id']} with query: {subscription['query']}")
            
            # Check each new article against this subscription
            for article_id in request.article_ids:
                matching_article = None
                
                if request.use_semantic_matching and subscription.get('query'):
                    # Full semantic search matching
                    matching_article = await check_article_matches_subscription(
                        article_id, subscription, app.state
                    )
                else:
                    # Basic metadata filtering only (faster)
                    matching_article = await check_article_matches_basic(
                        article_id, subscription, app.state
                    )
                
                if matching_article:
                    subscription_matches[subscription['subscription_id']].append(matching_article)
                    total_matches += 1
        
        # Handle delivery based on immediate_delivery flag
        if request.immediate_delivery and subscription_matches:
            # Immediately deliver all matches
            delivery_tasks = []
            for sub_id, articles in subscription_matches.items():
                if articles:
                    subscription = next(s for s in subscriptions if s['subscription_id'] == sub_id)
                    task = deliver_news_batch(subscription, articles, app.state)
                    delivery_tasks.append(task)
                    logger.info(f"🚀 Scheduling immediate delivery of {len(articles)} articles to subscription {sub_id}")
            
            # Execute all deliveries concurrently
            if delivery_tasks:
                await asyncio.gather(*delivery_tasks, return_exceptions=True)
        else:
            # Add to pending deliveries for batched delivery
            for sub_id, articles in subscription_matches.items():
                if articles:
                    app.state.subscription_manager.delivery_status[sub_id].pending_articles.extend(articles)
                    logger.info(f"📋 Added {len(articles)} articles to pending delivery for {sub_id}")
        
        return {
            "status": "success",
            "processed_articles": len(request.article_ids),
            "total_matches": total_matches,
            "subscriptions_with_matches": len(subscription_matches),
            "immediate_delivery": request.immediate_delivery,
            "semantic_matching": request.use_semantic_matching
        }
        
    except Exception as e:
        logger.error(f"Error processing scraped news webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process scraped news notification")


# --- Root Endpoint for Health Check ---
@app.get("/")
def read_root():
    return {
        "status": "API is running",
        "version": "2.0.0",
        "features": ["semantic_search", "subscriptions", "live_delivery"]
    }


if __name__ == "__main__":
    uvicorn.run("subscription-api-v5:app", host="0.0.0.0", port=8000, reload=True)