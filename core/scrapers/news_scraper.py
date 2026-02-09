from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from datetime import datetime, timezone, timedelta
from typing import List
import time, random, logging, sys
from functools import partial
from multiprocessing import Pool, cpu_count

import sys
import os

# Add project root to Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.utils.scraper_utils import fetch_news, process_news_article, News
from config.settings import ScraperConfig, DatabaseConfig
# ================== LOGGING SETUP ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# =================================================

# ================== CONFIGURATION ==================
scraper_config = ScraperConfig()
db_config = DatabaseConfig()
# =================================================

# Global variables for multiple functions
client = None
db = None
collection = None

# Function to save news to MongoDB, avoiding duplicates based on the 'link' field
def save_to_mongo(news_list: List[News]):
    global collection
    for news in news_list:
        doc = news.to_dict()
        while True:
            try:
                collection.insert_one(doc)
                logging.info(f"Inserted: {news.title}")
                break  # Success, move to the next article
            except DuplicateKeyError:
                logging.warning(f"Skipped duplicate: {news.title}")
                break  # Duplicate, no need to retry, move to the next article
            except Exception as e:
                logging.error(f"Database connection error: {e}. Retrying in 15 seconds...")
                time.sleep(15)

# Main scraper function using multiprocessing
def scrape_and_store(n_page: int = 1, process_cnt: int = 1, offset: int = 0, section: str = "news", api_type: str = 'category', deep_scrape: bool = False):
    global collection

    if process_cnt >= cpu_count() - 2:
        logging.warning("Using too many processors")
        return False

    # Get the latest article's timestamp for incremental scraping for the specific section
    max_published_date_in_db = None
    if not deep_scrape:
        latest_doc = collection.find_one(
            {"section": section},
            sort=[("published_time", -1)]
        )
        if latest_doc:
            max_published_date_in_db = datetime.fromisoformat(latest_doc['published_time'])
            # If the datetime from the DB is naive, make it offset-aware.
            if max_published_date_in_db.tzinfo is None:
                logging.warning(f"Found naive datetime in DB for section '{section}' ({max_published_date_in_db}), assuming UTC+8.")
                max_published_date_in_db = max_published_date_in_db.replace(tzinfo=timezone(timedelta(hours=8)))
            logging.info(f"Newest article in DB for section '{section}' for incremental check: {max_published_date_in_db}")

    existing_links = set(doc["link"].replace("www.theedgemarkets.com", "theedgemalaysia.com") for doc in collection.find({}, {"link": 1}))
    logging.info(f"Found {len(existing_links)} existing links in the database.")

    count = 0
    total_fetched = 0
    total_inserted = 0

    while True:
        if n_page >= 0 and count >= n_page:
            break
        elif n_page < 0 and n_page != -1:
            break

        news_batch = fetch_news(offset, section=section, api_type=api_type)
        if not news_batch:
            logging.info("No more news found for this section, stopping.")
            break

        # For incremental runs, check if all articles on this page are older than our newest one for this section.
        if not deep_scrape and max_published_date_in_db:
            all_articles_are_old = True
            for news_item in news_batch:
                timestamp = news_item.get("created")
                if timestamp:
                    article_date = datetime.fromtimestamp(timestamp / 1000, tz=timezone(timedelta(hours=8)))
                    if article_date > max_published_date_in_db:
                        all_articles_are_old = False
                        break
            if all_articles_are_old:
                logging.warning(f"Stopping scrape for section '{section}': All articles on this page are older than the newest in DB for this section.")
                break

        offset += len(news_batch)
        count += 1
        logging.info(f"\nPage {count} loaded with {len(news_batch)} articles.")

        to_process = [news for news in news_batch if f"https://theedgemalaysia.com/{news['alias']}" not in existing_links]
        logging.info(f"Found {len(to_process)} new articles to scrape.")
        
        if not to_process:
            logging.info("This page contains no new articles. Skipping page.")
            continue

        total_fetched += len(to_process)
        
        with Pool(processes=process_cnt) as pool:
            try:
                # Use functools.partial to pass the section to the worker process
                process_with_section = partial(process_news_article, section=section)
                results = pool.map(process_with_section, to_process)
            except KeyboardInterrupt:
                logging.warning("\nKeyboardInterrupt detected in worker pool. Terminating workers...")
                pool.terminate()
                pool.join()
                raise

        valid_news = [news for news in results if news is not None]
        total_inserted += len(valid_news)

        save_to_mongo(valid_news)
        time.sleep(random.uniform(0.5, 1.5))

    logging.info(f"\nPages processed: {count}")
    logging.info(f"New articles found: {total_fetched}")
    logging.info(f"Successfully inserted: {total_inserted}")

def scheduled_job(n_page: int, process_cnt: int, offset: int, deep_scrape: bool = False):
    if deep_scrape:
        logging.info("\nStarting one-time DEEP SCRAPE task...")
    else:
        logging.info("\nStarting scheduled INCREMENTAL SCRAPE task...")

    try:
        start = time.time()

        sections_categories = [ "corporate", "economy", "malaysia", "world", "options", "city-country", "wealth", "court", "technology", "Frankly%20Speaking", "Forum", "The%20Edge%20Says", "news"]
        sections_options = ["digitaledge", "esg", "politics"]
        sections_flash = ["Economic%20Focus", "Hot%20Stock", "IPO"]

        for section in sections_categories:
            logging.info(f"\n--- Scraping Category: {section} ---")
            scrape_and_store(n_page=n_page, process_cnt=process_cnt, offset=offset, section=section, api_type='category', deep_scrape=deep_scrape)

        for section in sections_options:
            logging.info(f"\n--- Scraping Option: {section} ---")
            scrape_and_store(n_page=n_page, process_cnt=process_cnt, offset=offset, section=section, api_type='option', deep_scrape=deep_scrape)

        for section in sections_flash:
            logging.info(f"\n--- Scraping Flash Category: {section} ---")
            scrape_and_store(n_page=n_page, process_cnt=process_cnt, offset=offset, section=section, api_type='flash', deep_scrape=deep_scrape)

        end = time.time()
        logging.info(f"\nTotal time taken: {end - start:.2f}s")

    except KeyboardInterrupt:
        logging.info("\nManual interruption of scheduled job.")
        exit(0)

# Main entry point
if __name__ == "__main__":
    # Attempt to connect to MongoDB
    logging.info("Attempting to connect to MongoDB...")
    try:
        client = MongoClient(db_config.mongo_uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        logging.info("MongoDB connection successful.")
    except Exception as e:
        logging.critical(f"Could not connect to MongoDB. Please start the service. Error: {e}", exc_info=True)
        exit(1)

    db = client[db_config.mongo_db]
    collection = db[db_config.mongo_collection]

    # Create a unique index on the 'link' field to prevent duplicates
    collection.create_index([("link", 1)], unique=True)

    # ==================================================================
    # This script is now designed to be called from an external scheduler (like run_pipeline.py).
    # It performs one run and then exits.
    # The 'deep_scrape' flag is controlled by the command-line argument.
    import argparse
    parser = argparse.ArgumentParser(description="Scrape news from The Edge.")
    parser.add_argument("--deep", action="store_true", help="Perform a deep scrape of all pages. Default is incremental.")
    args = parser.parse_args()

    # When run from the pipeline, we want an incremental scrape. 
    # A manual run of this script can still do a deep scrape if DEEP_SCRAPE is True in the config.
    run_deep_scrape = args.deep

    scheduled_job(n_page=scraper_config.n_page_limit, process_cnt=scraper_config.process_count, offset=scraper_config.start_offset, deep_scrape=run_deep_scrape)

    logging.info("\nScraping task finished.")
