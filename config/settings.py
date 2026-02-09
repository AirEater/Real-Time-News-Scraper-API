from dataclasses import dataclass
from typing import Optional
import os

# The absolute path to the 'FYPII_Project' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class DatabaseConfig:
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "news_db"
    mongo_collection: str = "news_the_edge_market"
    
@dataclass
class MilvusConfig:
    host: str = "localhost"
    port: str = "19530"
    collection_name: str = "news_semantic_chunks"

@dataclass
class DataConfig:
    listed_companies_json_path: str = os.path.join(project_root, "data", "listed_companies.json")

@dataclass
class ScraperConfig:
    n_page_limit: int = 1000 # Page limit for scraping. Set to -1 to scrape all pages.
    process_count: int = 4 # Number of CPU cores for multiprocessing.
    schedule_minutes: int = 15
    start_offset: int = 0 # Initial offset for scraping the edge malaysia news, should usually be 0(for latest).

@dataclass
class LLMConfig:
    model: str = "google/gemma-3-27b-it:free"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    env_file_path: str = os.path.join(project_root, "config", "API_Key.env")
