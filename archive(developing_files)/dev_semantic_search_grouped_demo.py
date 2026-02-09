import os
import openai
from pymilvus import connections, Collection, utility
from dotenv import load_dotenv
import logging
import io
import sys

# --- Basic Setup ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Load environment variables for API keys
load_dotenv("openaiAPI.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Milvus Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "news_semantic_chunks"
TOP_K = 30  # Number of initial chunks to retrieve for better grouping

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
def search_and_group_news(collection: Collection, query: str):
    """
    Searches for news, groups the results by article, and displays the most relevant snippet from each.
    """
    # 1. Generate embedding for the user's query
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("Could not generate query embedding. Please check your API key and network.")
        return

    # 2. Define search parameters
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}

    # 3. Perform the search to get the top K chunks
    logging.info(f"Searching for top {TOP_K} chunks in '{COLLECTION_NAME}'...")
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=TOP_K,
        output_fields=["mongo_id", "title", "link", "published_time", "chunk_text"]
    )

    # 4. Process and group the results
    hits = results[0]
    if not hits:
        print("No results found.")
        return

    # Group chunks by article ID to avoid duplicates
    processed_articles = {}
    for hit in hits:
        article_id = hit.entity.get('mongo_id')
        # If we haven't seen this article yet, add it to our results.
        # Since the hits are sorted by relevance, the first one we see for each article is the most relevant.
        if article_id not in processed_articles:
            processed_articles[article_id] = {
                "title": hit.entity.get('title'),
                "link": hit.entity.get('link'),
                "published_time": hit.entity.get('published_time'),
                "relevant_chunk": hit.entity.get('chunk_text'),
                "similarity": hit.distance
            }

    # 5. Display the grouped and de-duplicated results
    print(
"---\n--- Top Search Results (Grouped by Article) ---")
    if not processed_articles:
        print("No articles found matching the query.")
        return
        
    for i, article in enumerate(processed_articles.values()):
        print(f"\n--- Result {i+1} (Best Similarity: {article['similarity']:.4f}) ---")
        print(f"Title: {article['title']}")
        print(f"Link: {article['link']}")
        print(f"Published: {article['published_time']}")
        print(f"\nMost Relevant Snippet:\n{article['relevant_chunk']}\n---")


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
        print("\nWelcome to the Grouped Semantic Search Demo!")
        print("Type your query and press Enter. Type 'exit' to quit.")
        
        while True:
            user_query = input("\nSearch Query > ")
            if user_query.lower() == 'exit':
                break
            if not user_query.strip():
                continue
            
            search_and_group_news(collection, user_query)

    except Exception as e:
        logging.critical(f"An error occurred: {e}", exc_info=True)
    finally:
        connections.disconnect("default")
        logging.info("Disconnected from Milvus.")
