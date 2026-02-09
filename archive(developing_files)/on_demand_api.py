import os
from pymilvus import connections, Collection, utility
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging
import io
import sys
import numpy as np
import json
import time
from datetime import datetime, timedelta

import sys
import os

# Add project root to Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the new on-demand classification utility
from core.utils.llm_utils import classify_with_model

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- API & DB Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "news_semantic_chunks"

MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "news_db"
MONGO_COLLECTION_NAME = "news_the_edge_market"

# --- Main Search Logic (DEMO VERSION with On-Demand Processing) ---
def search_with_tiered_threshold(
    milvus_collection: Collection, 
    mongo_collection,
    model, # Accept the loaded SentenceTransformer model
    query: str, 
    limit: int = -1, # Default to -1 for unlimited results over threshold
    sort_by: str = "similarity", # New sorting parameter
    content_format: str = "full",
    with_link: bool = False,
    category: str = "all",
    company: str = "all",
    start_date: str = None,
    duration_days: int = 365,
    custom_threshold: float = -1.0, # Use -1 as the default to signify auto-threshold
    company_matcher = None,
    company_variation_map = None
):
    start_time = time.time()

    # --- Smart Limit for On-Demand Processing ---
    is_on_demand_triggered = content_format in ["summary", "both"] or category.lower().strip() != 'all'
    if is_on_demand_triggered:
        if limit > 50:
            limit = 50
        elif limit == -1:
            logging.warning(f"On-demand processing is active. Forcing result limit to 10 (user requested {limit}).")
            limit = 10

    # --- Parameter Validation & Fallbacks ---
    valid_formats = ["summary", "full", "both"]
    if content_format not in valid_formats:
        logging.warning(f"Invalid content_format '{content_format}'. Falling back to 'summary'.")
        content_format = "summary"

    if not isinstance(with_link, bool):
        logging.warning(f"Invalid with_link value. Must be boolean. Falling back to False.")
        with_link = False

    try:
        query_embedding = model.encode(query).tolist()
    except Exception as e:
        logging.error(f"Failed to generate embedding: {e}")
        return {"error": "Could not generate query embedding."}

    # --- Build Filter Expression (excluding category for now) ---
    filters = []
    user_requested_category = category.lower().strip()

    # 1. Company-First Filtering
    if company and company.strip().lower() != 'all' and company_matcher and company_variation_map:
        logging.info(f"Performing company filter for: '{company}'")
        match = company_matcher.search(company.upper())
        if match:
            matched_variation = match.group(0)
            company_data = company_variation_map.get(matched_variation)
            if company_data:
                target_ticker = company_data['ticker']
                logging.info(f"Company query '{company}' matched to ticker: {target_ticker}")
                
                article_ids_cursor = mongo_collection.find({"tickers": target_ticker}, {"_id": 1})
                article_ids = [str(doc['_id']) for doc in article_ids_cursor]
                
                if not article_ids:
                    return {"search_metadata": {"query": query, "status": f"No articles found for company '{company_data['company']}'."}}

                id_filter_str = ", ".join([f'\"{_id}\"' for _id in article_ids])
                filters.append(f'mongo_id in [{id_filter_str}]')
            else:
                 return {"search_metadata": {"query": query, "status": f"Company '{company}' could not be resolved by matcher."}}
        else:
            return {"search_metadata": {"query": query, "status": f"Company '{company}' not found."}}

    # 2. Date Filter Logic
    try:
        today = datetime.now()
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = start_dt + timedelta(days=duration_days)
        else:
            end_dt = today
            start_dt = end_dt - timedelta(days=duration_days)

        if start_dt <= today:
            end_dt = min(end_dt, today)
            exclusive_end_dt = end_dt + timedelta(days=1)
            date_filter_str = f"(published_time >= '{start_dt.strftime('%Y-%m-%d')}' and published_time < '{exclusive_end_dt.strftime('%Y-%m-%d')}')"
            filters.append(date_filter_str)
    except (ValueError, TypeError):
        logging.warning(f"Invalid date format or value. No date filter applied.")

    expr = " and ".join(filters)
    logging.info(f"Executing Milvus search with expression: {expr or 'No filters'}")

    # --- Perform Broad Search --- 
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}

    pre_flight_results = milvus_collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=1, expr=expr, output_fields=["mongo_id"])
    if not pre_flight_results[0] or pre_flight_results[0][0].distance < 0.33:
        return {"search_metadata": {"query": query, "status": "No sufficiently relevant results found."}}

    # Dynamically determine how many chunks to retrieve
    if limit == -1:
        candidate_limit = 1000
    else:
        base_candidate_limit = 150
        multiplier = 20
        candidate_limit = base_candidate_limit + limit * multiplier
        candidate_limit = min(candidate_limit, 2000)

    logging.info(f"Fetching top {candidate_limit} candidate chunks from Milvus...")
    results = milvus_collection.search(
        data=[query_embedding], 
        anns_field="embedding", 
        param=search_params, 
        limit=candidate_limit, 
        expr=expr, 
        output_fields=["mongo_id", "title", "link", "published_time", "chunk_text"]
    )
    hits = results[0]
    if not hits:
        return {"search_metadata": {"query": query, "status": "No results found."}}

    # --- Process and Enrich Results --- 
    scores = [hit.distance for hit in hits]
    top_score = scores[0] if scores else 0

    final_threshold = 0.0
    if custom_threshold != -1.0:
        final_threshold = custom_threshold
    else:
        if top_score >= 0.50: final_threshold = 0.50
        elif top_score >= 0.40: final_threshold = 0.40
        else: final_threshold = 0.38

    final_hits = [hit for hit in hits if hit.distance >= final_threshold]
    if not final_hits:
        return {"search_metadata": {"query": query, "status": f"No chunks met the final threshold of {final_threshold:.4f}."}}

    all_processed_articles = []
    seen_article_ids = set()
    for hit in final_hits:
        article_id = hit.entity.get('mongo_id')
        if article_id not in seen_article_ids:
            article_doc = mongo_collection.find_one({"_id": ObjectId(article_id)})
            if not article_doc:
                continue

            # On-Demand Classification & Summarization
            needs_processing = not article_doc.get('summary') or article_doc.get('category') in ["unclassified", None, ""]
            if (content_format in ["summary", "both"] or user_requested_category != 'all') and needs_processing:
                logging.warning(f"Article {article_id} requires on-demand processing.")
                full_content = f"title: {article_doc.get('title', '')}\n{article_doc.get('content', '')}"
                new_summary, new_category = classify_with_model(full_content)
                if new_summary != "daily limit exceeded":
                    mongo_collection.update_one(
                        {"_id": ObjectId(article_id)},
                        {"$set": {"summary": new_summary, "category": new_category}}
                    )
                    logging.info(f"Successfully updated article {article_id} with new data.")
                    article_doc['summary'] = new_summary
                    article_doc['category'] = new_category
                else:
                    logging.error(f"Could not process article {article_id} due to API limits.")

            article_data = {
                "mongo_id": article_id,
                "title": hit.entity.get('title'),
                "published_time": hit.entity.get('published_time'),
                "category": article_doc.get('category'),
                "companies": article_doc.get('companies', []),
                "tickers": article_doc.get('tickers', []),
                "similarity_score": hit.distance,
                "content_format_returned": content_format,
                "summary": article_doc.get("summary", "N/A") if content_format in ["summary", "both"] else None,
                "content": article_doc.get("content", "N/A") if content_format in ["full", "both"] else None
            }
            if with_link:
                article_data["link"] = hit.entity.get('link')
            
            all_processed_articles.append(article_data)
            seen_article_ids.add(article_id)

            # --- Early Exit if Limit is Reached ---
            if limit != -1 and len(all_processed_articles) >= limit:
                logging.info(f"Sufficient number of articles ({limit}) collected. Stopping further processing.")
                break

    # --- In-Memory Filtering and Sorting ---
    # 1. Filter by category after enrichment
    if user_requested_category != 'all':
        final_articles = [p for p in all_processed_articles if p.get('category') == user_requested_category]
    else:
        final_articles = all_processed_articles

    # 2. Sort the results
    if sort_by == 'date':
        logging.info("Sorting final results by date (newest first).")
        final_articles.sort(key=lambda x: x.get('published_time', ''), reverse=True)
    # Default is similarity, which is the natural order from Milvus, so no action needed.

    # 3. Apply the final limit
    if limit != -1:
        final_articles = final_articles[:limit]

    time_taken_ms = (time.time() - start_time) * 1000
    return {
        "search_metadata": {
            "query": query,
            "sort_by": sort_by,
            "final_threshold_used": final_threshold,
            "total_results": len(final_articles),
            "time_taken_ms": time_taken_ms
        },
        "results": final_articles
    }

if __name__ == "__main__":
    # This is a module, not intended to be run directly.
    pass
