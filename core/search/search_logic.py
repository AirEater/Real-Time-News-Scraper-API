from pymilvus import connections, Collection, utility
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging
import io
import numpy as np
import json
import time
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

import sys
import os

# Add project root to Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import MilvusConfig, DatabaseConfig

# Import the on-demand classification utility
# Note: This will need its own import path updated later
from core.utils.llm_utils import classify_with_model

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- Embedding Function for Standard Search ---
def get_embedding_standard(text: str, model: SentenceTransformer) -> list[float]:
    """Generates an embedding for a single query using a BGE model, prepending the required instruction."""
    query_with_instruction = f"Represent this sentence for searching relevant passages: {text}"
    try:
        embedding = model.encode(query_with_instruction, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        logging.error(f"Failed to get embedding using SentenceTransformer: {e}")
        return None

# --- Search Logic 1: Standard Search (No On-Demand Classification) ---
def search_standard(
    milvus_collection: Collection, 
    mongo_collection,
    model: SentenceTransformer,
    query: str, 
    limit: int = -1,
    sort_by: str = "similarity",
    content_format: str = "summary",
    with_link: bool = False,
    category: str = "all",
    company: str = "all",
    start_date: str = None,
    duration_days: int = 365,
    custom_threshold: float = -1.0,
    company_matcher = None,
    company_variation_map = None
):
    start_time = time.time()

    valid_formats = ["summary", "full", "both"]
    if content_format not in valid_formats:
        content_format = "summary"
    if not isinstance(with_link, bool):
        with_link = False

    query_embedding = get_embedding_standard(query, model)
    if not query_embedding:
        return {"error": "Could not generate query embedding."}

    filters = []
    if company and company.strip().lower() != 'all' and company_matcher and company_variation_map:
        match = company_matcher.search(company.upper())
        if match:
            company_data = company_variation_map.get(match.group(0))
            if company_data:
                article_ids_cursor = mongo_collection.find({"tickers": company_data['ticker']}, {"_id": 1})
                article_ids = [str(doc['_id']) for doc in article_ids_cursor]
                if not article_ids:
                    return {"search_metadata": {"query": query, "status": f"No articles found for company '{company_data['company']}'."}}
                id_filter_str = ", ".join([f'"{_id}"' for _id in article_ids])
                filters.append(f'mongo_id in [{id_filter_str}]')
            else:
                 return {"search_metadata": {"query": query, "status": f"Company '{company}' could not be resolved by matcher."}}
        else:
            return {"search_metadata": {"query": query, "status": f"Company '{company}' not found."}}

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
            filters.append(f"(published_time >= '{start_dt.strftime('%Y-%m-%d')}' and published_time < '{exclusive_end_dt.strftime('%Y-%m-%d')}')")
    except (ValueError, TypeError):
        logging.warning(f"Invalid date format or value. No date filter applied.")

    if category and category.lower() != 'all':
        filters.append(f"category == '{category}'")

    expr = " and ".join(filters)
    logging.info(f"Executing Milvus search with expression: {expr or 'No filters'}")
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
    
    pre_flight_results = milvus_collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=1, expr=expr, output_fields=["mongo_id"])
    if not pre_flight_results[0] or pre_flight_results[0][0].distance < 0.33:
        return {"search_metadata": {"query": query, "status": "No sufficiently relevant results found."}}

    candidate_limit = 1000 if limit == -1 else min(150 + limit * 20, 2000)
    logging.info(f"Fetching top {candidate_limit} candidate chunks from Milvus...")
    
    results = milvus_collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=candidate_limit, expr=expr, output_fields=["mongo_id", "title", "link", "published_time", "chunk_text"])
    hits = results[0]
    if not hits:
        return {"search_metadata": {"query": query, "status": "No results found."}}

    top_score = hits[0].distance if hits else 0
    final_threshold = custom_threshold if custom_threshold != -1.0 else (0.50 if top_score >= 0.50 else (0.40 if top_score >= 0.40 else 0.38))
    
    final_hits = [hit for hit in hits if hit.distance >= final_threshold]
    if not final_hits:
        return {"search_metadata": {"query": query, "status": f"No chunks met the final threshold of {final_threshold:.4f}."}}

    processed_articles = []
    seen_article_ids = set()
    for hit in final_hits:
        if limit != -1 and len(processed_articles) >= limit:
            break 
        article_id = hit.entity.get('mongo_id')
        if article_id not in seen_article_ids:
            article_doc = mongo_collection.find_one({"_id": ObjectId(article_id)})
            if not article_doc: continue
            article_data = {
                "title": hit.entity.get('title'), "published_time": hit.entity.get('published_time'),
                "category": article_doc.get('category'), "companies": article_doc.get('companies', []),
                "tickers": article_doc.get('tickers', []), "similarity_score": hit.distance,
                "content_format_returned": content_format,
                "summary": article_doc.get("summary", "N/A") if content_format in ["summary", "both"] else None,
                "content": article_doc.get("content", "N/A") if content_format in ["full", "both"] else None
            }
            if with_link: article_data["link"] = hit.entity.get('link')
            processed_articles.append(article_data)
            seen_article_ids.add(article_id)
    
    if sort_by == 'date':
        processed_articles.sort(key=lambda x: x.get('published_time', ''), reverse=True)

    time_taken_ms = (time.time() - start_time) * 1000
    return {"search_metadata": {"query": query, "final_threshold_used": final_threshold, "total_results": len(processed_articles), "time_taken_ms": time_taken_ms}, "results": processed_articles}


# --- Search Logic 2: On-Demand Search (With Live Classification) ---
def search_on_demand(
    milvus_collection: Collection, 
    mongo_collection,
    model,
    query: str, 
    limit: int = -1,
    sort_by: str = "similarity",
    content_format: str = "full",
    with_link: bool = False,
    category: str = "all",
    company: str = "all",
    start_date: str = None,
    duration_days: int = 365,
    custom_threshold: float = -1.0,
    company_matcher = None,
    company_variation_map = None
):
    start_time = time.time()

    is_on_demand_triggered = content_format in ["summary", "both"] or category.lower().strip() != 'all'
    if is_on_demand_triggered:
        if limit > 50: limit = 50
        elif limit == -1: limit = 10

    valid_formats = ["summary", "full", "both"]
    if content_format not in valid_formats: content_format = "summary"
    if not isinstance(with_link, bool): with_link = False

    try:
        query_embedding = get_embedding_standard(query, model)
    except Exception as e:
        return {"error": "Could not generate query embedding."}
    
    if not query_embedding:
        return {"error": "Could not generate query embedding."}
    filters = []
    user_requested_category = category.lower().strip()

    if company and company.strip().lower() != 'all' and company_matcher and company_variation_map:
        match = company_matcher.search(company.upper())
        if match:
            company_data = company_variation_map.get(match.group(0))
            if company_data:
                article_ids_cursor = mongo_collection.find({"tickers": company_data['ticker']}, {"_id": 1})
                article_ids = [str(doc['_id']) for doc in article_ids_cursor]
                if not article_ids:
                    return {"search_metadata": {"query": query, "status": f"No articles found for company '{company_data['company']}'."}}
                id_filter_str = ", ".join([f'"{_id}"' for _id in article_ids])
                filters.append(f'mongo_id in [{id_filter_str}]')
            else:
                 return {"search_metadata": {"query": query, "status": f"Company '{company}' could not be resolved by matcher."}}
        else:
            return {"search_metadata": {"query": query, "status": f"Company '{company}' not found."}}

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
            filters.append(f"(published_time >= '{start_dt.strftime('%Y-%m-%d')}' and published_time < '{exclusive_end_dt.strftime('%Y-%m-%d')}')")
    except (ValueError, TypeError):
        logging.warning(f"Invalid date format or value. No date filter applied.")

    expr = " and ".join(filters)
    logging.info(f"Executing Milvus search with expression: {expr or 'No filters'}")
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}

    pre_flight_results = milvus_collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=1, expr=expr, output_fields=["mongo_id"])
    if not pre_flight_results[0] or pre_flight_results[0][0].distance < 0.33:
        return {"search_metadata": {"query": query, "status": "No sufficiently relevant results found."}}

    candidate_limit = 1000 if limit == -1 else min(150 + limit * 20, 2000)
    logging.info(f"Fetching top {candidate_limit} candidate chunks from Milvus...")
    
    results = milvus_collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=candidate_limit, expr=expr, output_fields=["mongo_id", "title", "link", "published_time", "chunk_text"])
    hits = results[0]
    if not hits:
        return {"search_metadata": {"query": query, "status": "No results found."}}

    top_score = hits[0].distance if hits else 0
    final_threshold = custom_threshold if custom_threshold != -1.0 else (0.50 if top_score >= 0.50 else (0.40 if top_score >= 0.40 else 0.38))

    final_hits = [hit for hit in hits if hit.distance >= final_threshold]
    if not final_hits:
        return {"search_metadata": {"query": query, "status": f"No chunks met the final threshold of {final_threshold:.4f}."}}

    all_processed_articles = []
    seen_article_ids = set()
    for hit in final_hits:
        article_id = hit.entity.get('mongo_id')
        if article_id not in seen_article_ids:
            article_doc = mongo_collection.find_one({"_id": ObjectId(article_id)})
            if not article_doc: continue

            needs_processing = not article_doc.get('summary') or article_doc.get('category') in ["unclassified", None, ""]
            if (content_format in ["summary", "both"] or user_requested_category != 'all') and needs_processing:
                full_content = f"title: {article_doc.get('title', '')}\n{article_doc.get('content', '')}"
                new_summary, new_category = classify_with_model(full_content)
                if new_summary != "daily limit exceeded":
                    mongo_collection.update_one({"_id": ObjectId(article_id)}, {"$set": {"summary": new_summary, "category": new_category}})
                    article_doc['summary'] = new_summary
                    article_doc['category'] = new_category
                else:
                    logging.error(f"Could not process article {article_id} due to API limits.")

            article_data = {
                "mongo_id": article_id, "title": hit.entity.get('title'), "published_time": hit.entity.get('published_time'),
                "category": article_doc.get('category'), "companies": article_doc.get('companies', []),
                "tickers": article_doc.get('tickers', []), "similarity_score": hit.distance,
                "content_format_returned": content_format,
                "summary": article_doc.get("summary", "N/A") if content_format in ["summary", "both"] else None,
                "content": article_doc.get("content", "N/A") if content_format in ["full", "both"] else None
            }
            if with_link: article_data["link"] = hit.entity.get('link')
            all_processed_articles.append(article_data)
            seen_article_ids.add(article_id)

            if limit != -1 and len(all_processed_articles) >= limit:
                logging.info(f"Sufficient number of articles ({limit}) collected. Stopping further processing.")
                break

    if user_requested_category != 'all':
        final_articles = [p for p in all_processed_articles if p.get('category') == user_requested_category]
    else:
        final_articles = all_processed_articles

    if sort_by == 'date':
        final_articles.sort(key=lambda x: x.get('published_time', ''), reverse=True)

    if limit != -1:
        final_articles = final_articles[:limit]

    time_taken_ms = (time.time() - start_time) * 1000
    return {"search_metadata": {"query": query, "sort_by": sort_by, "final_threshold_used": final_threshold, "total_results": len(final_articles), "time_taken_ms": time_taken_ms}, "results": final_articles}
