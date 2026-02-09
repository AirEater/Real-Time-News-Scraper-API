import re
import time
import requests
from typing import Optional
from dotenv import load_dotenv
import logging

import sys
import os

# Add project root to Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from config.settings import LLMConfig

llm_config = LLMConfig()
MODEL = llm_config.model
# --- API Key Management ---
# This assumes the script is run from the project root, so the path is relative to that.
load_dotenv(llm_config.env_file_path)

api_keys = [
    os.getenv("OPENROUTER_API_KEY1"),
    os.getenv("OPENROUTER_API_KEY2"),
    os.getenv("OPENROUTER_API_KEY3"),
    os.getenv("OPENROUTER_API_KEY4"),
    os.getenv("OPENROUTER_API_KEY5"),
]

class APIKeyManager:
    def __init__(self, keys):
        self.keys = [k for k in keys if k] # Filter out None keys
        self.current_index = 0
        self.exhausted_keys = set()

    def get_key(self):
        if not self.keys:
            return None
        return self.keys[self.current_index]

    def switch_key(self):
        if not self.keys:
            return
        self.exhausted_keys.add(self.keys[self.current_index])
        self.current_index = (self.current_index + 1) % len(self.keys)

    def all_keys_exhausted(self):
        return len(self.exhausted_keys) >= len(self.keys)

api_key_manager = APIKeyManager(api_keys)

# --- LLM Classification and Summarization Function ---
def classify_with_model(content: str, max_retries: int = 3, model=MODEL) -> Optional[tuple]:
    prompt = f"""
You are an assistant that processes news articles for a financial analysis system. 

Task: 
1. Summarize the news article in at least two concise sentences, capturing the core message and key details. This summary will be used as part of a Retrieval-Augmented Generation (RAG) knowledge base for a Large Language Model (LLM). Therefore, it is important that the summary accurately reflects the main content and essential points of the news. 
2. Classify the article into exactly one category: 
- financial news 
- government policy news 

Output Format (strictly, no explanations or extra text): 
Summary: [At least two-sentence summary] 
Category: [financial news or government policy news] 

News: [{content}]"""
    attempt = 0
    while attempt < max_retries:
        try:
            api_key = api_key_manager.get_key()
            if not api_key or api_key_manager.all_keys_exhausted():
                logging.error("All API keys are exhausted or no keys are available.")
                return "daily limit exceeded", "unclassified"

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://theedgemalaysia.com", # Optional: Can be customized
                    "X-Title": "News Classifier" # Optional: Can be customized
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=45 # Increased timeout for potentially long summaries
            )

            result = response.json()
            if "error" in result:
                error_message = result["error"].get("message", "")
                if "free-models-per-day" in error_message:
                    logging.warning(f"Daily limit reached for key: {api_key[-4:]}")
                    api_key_manager.switch_key()
                    if api_key_manager.all_keys_exhausted():
                        return "daily limit exceeded", "unclassified"
                    continue # Retry with the next key
                elif "free-models-per-min" in error_message:
                    logging.warning("Rate limit per minute exceeded. Waiting 60 seconds...")
                    time.sleep(60)
                    continue # Retry the same key after waiting
                else:
                    logging.error(f"OpenRouter API Error: {error_message}")
                    return "No summary", "unclassified"

            res = result["choices"][0]["message"]["content"].strip()
            summary_match = re.search(r"Summary:\s*(.+?)(?:\n|$)", res, re.DOTALL)
            category_match = re.search(r"Category:\s*(.+?)(?:\n|$)", res, re.DOTALL)

            summary = summary_match.group(1).strip() if summary_match else "No summary available"
            category = category_match.group(1).lower().strip() if category_match else "unclassified"
            
            # Basic validation for the category
            if category not in ["financial news", "government policy news"]:
                category = "unclassified"

            return summary, category

        except requests.exceptions.Timeout:
            logging.warning(f"Request timed out. Retrying... ({attempt + 1}/{max_retries})")
            attempt += 1
            time.sleep(5)
        except Exception as e:
            logging.error(f"An unexpected error occurred during classification: {e}. Retrying... ({attempt + 1}/{max_retries})")
            attempt += 1
            time.sleep(5)

    return "Failed to process", "unclassified"