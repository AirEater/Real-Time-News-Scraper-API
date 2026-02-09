
import os
import openai
from pymilvus import connections, Collection, utility
from dotenv import load_dotenv
import logging

# --- Basic Setup ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables for API keys
load_dotenv("openaiAPI.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Milvus Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "news_semantic_chunks"
TOP_K = 30 # Number of results to return

# --- OpenAI Embedding Function ---
def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    """Generates an embedding for a given text using OpenAI API."""
    try:
        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Failed to get embedding from OpenAI: {e}")
        return None

# --- Main Search Logic ---
def search_news_chunks(collection: Collection, query: str):
    """
    Searches for news chunks in Milvus based on a query string.
    """
    # 1. Generate embedding for the user's query
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("Could not generate query embedding. Please check your API key and network.")
        return

    # 2. Define search parameters for Milvus
    # These parameters are specific to the index type (e.g., IVF_FLAT)
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}

    # 3. Perform the search
    logging.info(f"Searching for top {TOP_K} results in '{COLLECTION_NAME}'...")
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=TOP_K,
        output_fields=["title", "link", "published_time", "chunk_text"]
    )

    # 4. Process and display results
    print("\n--- Search Results ---")
    hits = results[0]
    if not hits:
        print("No results found.")
        return

    for i, hit in enumerate(hits):
        print(f"\n--- Result {i+1} (Similarity: {hit.distance:.4f}) ---")
        print(f"Title: {hit.entity.get('title')}")
        print(f"Link: {hit.entity.get('link')}")
        print(f"Published: {hit.entity.get('published_time')}")
        print(f"\nRelevant Chunk:\n{hit.entity.get('chunk_text')}\n---")

if __name__ == "__main__":
    try:
        # Connect to Milvus
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        logging.info("✅ Milvus connection successful.")

        # Get collection and load it
        if not utility.has_collection(COLLECTION_NAME):
            logging.error(f"Collection '{COLLECTION_NAME}' not found in Milvus.")
            exit()
            
        collection = Collection(name=COLLECTION_NAME)
        collection.load()
        logging.info(f"Collection '{COLLECTION_NAME}' loaded with {collection.num_entities} entities.")

        # Interactive search loop
        print("\nWelcome to the Semantic News Chunk Search Demo!")
        print("Type your query and press Enter. Type 'exit' to quit.")
        
        while True:
            user_query = input("\nSearch Query > ")
            if user_query.lower() == 'exit':
                break
            if not user_query.strip():
                continue
            
            search_news_chunks(collection, user_query)

    except Exception as e:
        logging.critical(f"An error occurred: {e}", exc_info=True)
    finally:
        connections.disconnect("default")
        logging.info("Disconnected from Milvus.")
