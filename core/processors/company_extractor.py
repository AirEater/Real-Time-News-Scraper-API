import json
import re
import logging
import os
from pymongo import MongoClient
from bson.objectid import ObjectId

import sys
import os

# Add project root to Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import  DatabaseConfig, DataConfig

# --- Configuration ---
db_config = DatabaseConfig()
data_config = DataConfig()
FILE_PATH = data_config.listed_companies_json_path
# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_company_variations(company_name):
    """
    Generate common variations of a company name for matching.
    Example: "SIME DARBY PROPERTY BERHAD" -> {"SIME DARBY PROPERTY BERHAD", "SIME DARBY PROPERTY"}
    """
    variations = set()
    name = company_name.upper().strip()
    
    # Add original name
    variations.add(name)
    
    # Remove common suffixes and clean up
    suffixes = ['(PUBLIC)', 'BERHAD', 'BHD', 'SDN BHD', 'GROUP BERHAD', 'HOLDINGS', 'LIMITED', 'LTD', 'PLC']
    for suffix in suffixes:
        # Use regex to remove suffix as a whole word, and handle potential parentheses
        name = re.sub(r'\s*\b' + re.escape(suffix) + r'\b\s*', '', name, flags=re.IGNORECASE).strip()

    # After removing suffixes, add the cleaned name if it's different
    if name != company_name.upper().strip():
        variations.add(name)
        
    # Further simplify by removing things like "(KL:XXXX)" which might be in the name
    name = re.sub(r'\s*\(KL:\w+\)', '', name).strip()
    variations.add(name)

    # Return a list of non-empty, unique variations
    return list(filter(None, variations))

def build_matcher():
    """
    Builds an efficient regex matcher and a lookup map from the companies JSON file.
    """
    logging.info(f"Loading company data from {FILE_PATH}...")
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        companies_data = json.load(f)
    
    variation_map = {}
    all_variations = set()

    for company in companies_data:
        if 'company' not in company or not company['company']:
            continue
        
        variations = create_company_variations(company['company'])
        for variation in variations:
            # Map each variation back to the original company data
            if variation not in variation_map:
                variation_map[variation] = company
            all_variations.add(variation)

    # Sort variations by length, descending, to match longer names first
    # (e.g., "GENTING MALAYSIA" before "GENTING")
    sorted_variations = sorted(list(all_variations), key=len, reverse=True)
    
    # Create a single, large regex pattern
    master_regex = r'\b(' + '|'.join(re.escape(v) for v in sorted_variations) + r')\b'
    
    logging.info(f"Built matcher with {len(sorted_variations)} variations for {len(companies_data)} companies.")
    return re.compile(master_regex, re.IGNORECASE), variation_map

def process_articles(mongo_collection, matcher, variation_map):
    """
    Find unprocessed articles, match all companies, and update them in MongoDB.
    """
    query = {"company_extraction_status": {"$exists": False}}
    articles_to_process = list(mongo_collection.find(query))
    
    if not articles_to_process:
        logging.info("No new articles to process.")
        return

    logging.info(f"Found {len(articles_to_process)} articles to process.")
    
    updated_count = 0
    not_found_count = 0

    for article in articles_to_process:
        article_id = article['_id']
        text_to_search = (article.get('title', '') + ' ' + article.get('content', '')).upper()
        
        matches = matcher.findall(text_to_search)
        
        if matches:
            # Use a dictionary to store unique companies based on their ticker
            unique_companies = {}
            for matched_variation in matches:
                company_data = variation_map.get(matched_variation)
                if company_data:
                    # Use ticker as the key to ensure uniqueness
                    unique_companies[company_data['ticker']] = company_data
            
            if unique_companies:
                # Extract lists of names and tickers from the unique companies
                company_names = [comp['company'] for comp in unique_companies.values()]
                tickers = [comp['ticker'] for comp in unique_companies.values()]

                update_result = mongo_collection.update_one(
                    {"_id": article_id},
                    {
                        "$set": {
                            "companies": company_names,
                            "tickers": tickers,
                            "company_extraction_status": "complete"
                        }
                    }
                )
                if update_result.modified_count > 0:
                    logging.info(f"Article '{article['title'][:50]}...' matched with {len(unique_companies)} companies: {tickers}")
                    updated_count += 1
            else:
                # This case handles if findall() returns matches but none are in variation_map (unlikely but safe)
                mongo_collection.update_one(
                    {"_id": article_id},
                    {"$set": {"company_extraction_status": "not_found"}}
                )
                not_found_count += 1
        else:
            # Mark as not found to avoid re-scanning
            mongo_collection.update_one(
                {"_id": article_id},
                {"$set": {"company_extraction_status": "not_found"}}
            )
            not_found_count += 1
            
    logging.info("--- Processing Complete ---")
    logging.info(f"Successfully updated {updated_count} articles with company data.")
    logging.info(f"{not_found_count} articles were scanned, but no companies were found.")


def job():
    """Wrapper function for the main processing logic."""
    client = None
    try:
        # --- Build Matcher ---
        master_matcher, var_map = build_matcher()
        
        # --- Connect to DB ---
        logging.info(f"Connecting to MongoDB at {db_config.mongo_uri}...")
        client = MongoClient(db_config.mongo_uri)
        db = client[db_config.mongo_db]
        collection = db[db_config.mongo_collection]
        
        # --- Run Processing ---
        logging.info("--- Running company extraction job ---")
        process_articles(collection, master_matcher, var_map)
        logging.info("--- Company extraction job finished ---")
        
    except FileNotFoundError:
        logging.error(f"CRITICAL: The file '{FILE_PATH}' was not found. Please check the path.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the job: {e}", exc_info=True)
    finally:
        if client:
            client.close()
            logging.info("MongoDB connection closed.")

if __name__ == "__main__":
    job()