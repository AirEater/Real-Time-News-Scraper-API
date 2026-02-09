import json
import re
import logging

import sys
import os

# Add project root to Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config.settings import DataConfig
# This file is a reusable module containing the core logic for matching company names.
# It is extracted from 1.1_extract_listed_company.py to be used by the API.

# --- Configuration ---
# The path is relative to this file's location, assuming listed_companies.json is in the parent directory.
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

data_config = DataConfig()
COMPANIES_JSON_FILE = data_config.listed_companies_json_path
COMPANIES_JSON_PATH = os.path.join(SCRIPT_DIR, COMPANIES_JSON_FILE)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_company_variations(company_name):
    """
    Generate common variations of a company name for matching.
    """
    variations = set()
    name = company_name.upper().strip()
    variations.add(name)
    
    suffixes = ['(PUBLIC)', 'BERHAD', 'BHD', 'SDN BHD', 'GROUP', 'HOLDINGS', 'LIMITED', 'LTD', 'PLC']
    for suffix in suffixes:
        name = re.sub(r'\s*\b' + re.escape(suffix) + r'\b\s*$', '', name, flags=re.IGNORECASE).strip()

    if name != company_name.upper().strip():
        variations.add(name)
        
    name = re.sub(r'\s*\(KL:\w+\)', '', name).strip()
    variations.add(name)

    return list(filter(None, variations))

def build_matcher():
    """
    Builds an efficient regex matcher and a lookup map from the companies JSON file.
    """
    logging.info(f"Loading company data from {COMPANIES_JSON_PATH}...")
    try:
        with open(COMPANIES_JSON_PATH, 'r', encoding='utf-8') as f:
            companies_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"CRITICAL: The company list '{COMPANIES_JSON_PATH}' was not found.")
        raise
    
    variation_map = {}
    all_variations = set()

    for company in companies_data:
        if 'company' not in company or not company['company']:
            continue
        
        variations = create_company_variations(company['company'])
        for variation in variations:
            if variation not in variation_map:
                variation_map[variation] = company
            all_variations.add(variation)

    sorted_variations = sorted(list(all_variations), key=len, reverse=True)
    
    master_regex_pattern = r'\b(' + '|'.join(re.escape(v) for v in sorted_variations) + r')\b'
    master_regex = re.compile(master_regex_pattern, re.IGNORECASE)
    
    logging.info(f"Built company matcher with {len(sorted_variations)} variations.")
    return master_regex, variation_map
