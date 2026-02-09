import os
import openai
from pymilvus import connections, Collection, utility
from dotenv import load_dotenv
import logging
import io
import sys
import numpy as np

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
MAX_K = 1300 # Max possible K after expansions

# --- OpenAI Embedding Function ---
def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    """Generates an embedding for a given text using OpenAI API."""
    try:
        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Failed to get embedding from OpenAI: {e}")
        return None

# --- Helper Function for Search Expansion ---
def _get_adaptive_sample(collection: Collection, query_embedding: list, search_params: dict):
    """
    Iteratively expands the search window (K) to find an optimal sample of results.
    """
    total_vectors = collection.num_entities
    if total_vectors < 100:
        initial_k = min(30, total_vectors)
    elif total_vectors < 1000:
        initial_k = int(total_vectors * 0.3)
    else:
        initial_k = 300
    if total_vectors > 0 and initial_k == 0:
        initial_k = 1

    k = initial_k
    hits = []
    MAX_EXPANSIONS = 20
    EXPANSION_INCREMENT = 50

    for i in range(MAX_EXPANSIONS + 1):
        logging.info(f"Performing search with K={k} (Expansion iter: {i})")
        results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=k, output_fields=["mongo_id", "title", "link", "published_time", "chunk_text"])
        hits = results[0]

        if len(hits) < k:
            logging.info("Returned fewer results than requested, using these as final.")
            break

        if i == MAX_EXPANSIONS:
            logging.info("Max expansions reached, using final results.")
            break

        scores = [hit.distance for hit in hits]
        is_tail_strong = scores[-1] >= 0.48
        is_slope_flat = (scores[0] - scores[-1]) <= 0.07

        if not (is_tail_strong or is_slope_flat):
            logging.info(f"Stopping expansion at K={k}. Conditions not met.")
            break
        
        k += EXPANSION_INCREMENT
        logging.info(f"Expansion conditions met. Increasing K to {k}.")
    
    return hits

# --- Main Search Logic ---
def search_with_tiered_threshold(
    collection: Collection, 
    query: str, 
    relevance_level: str = "Related"
):
    """
    Performs a semantic search with a tiered, dynamic threshold system.
    """
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("Could not generate query embedding. Please check your API key and network.")
        return

    # --- Stage 1: Pre-flight Check ---
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
    logging.info("Performing pre-flight check with top 1 result...")
    pre_flight_results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=1, output_fields=["mongo_id"])
    if not pre_flight_results[0] or pre_flight_results[0][0].distance < 0.33:
        print("No sufficiently relevant results found (top result score < 0.33). Aborting search.")
        return
    logging.info("Pre-flight check passed. Proceeding with search.")

    # --- Stage 2: Iterative Sample Expansion ---
    hits = _get_adaptive_sample(collection, query_embedding, search_params)

    # --- Stage 3: Calculate Metrics & Boundaries ---
    if not hits:
        print("No results found after search expansion.")
        return

    scores = [hit.distance for hit in hits]
    top_score = scores[0]
    mean_score = np.mean(scores)
    std_dev_score = np.std(scores)
    dyn_threshold = mean_score + 0.5 * std_dev_score
    q70_threshold = np.quantile(scores, 0.70)

    logging.info(f"Metrics from final sample (K={len(scores)}): Top={top_score:.4f}, Mean={mean_score:.4f}, StdDev={std_dev_score:.4f}, Q70={q70_threshold:.4f}, Dyn={dyn_threshold:.4f}")

    if top_score < 0.5:
        logging.info("Tough Query detected. Using relative thresholds.")
        highly_related_boundary = max(dyn_threshold, q70_threshold)
        related_boundary = dyn_threshold
    else:
        logging.info("Normal Query detected. Using stable thresholds.")
        highly_related_boundary = max(dyn_threshold, q70_threshold, 0.5)
        related_boundary = 0.50
    possibly_related_boundary = 0.33

    logging.info(f"Boundaries: Highly>={highly_related_boundary:.4f}, Related>={related_boundary:.4f}, Possibly>={possibly_related_boundary:.4f}")

    # --- Stage 4: Filter and Group ---
    if relevance_level == "Highly Related":
        final_threshold = highly_related_boundary
    elif relevance_level == "Related":
        final_threshold = related_boundary
    else: # Possibly Related
        final_threshold = possibly_related_boundary

    logging.info(f"User requested level '{relevance_level}'. Using final threshold: {final_threshold:.4f}")
    final_hits = [hit for hit in hits if hit.distance >= final_threshold]
    if not final_hits:
        print(f"No chunks met the final threshold of {final_threshold:.4f}.")
        return

    processed_articles = {}
    for hit in final_hits:
        article_id = hit.entity.get('mongo_id')
        if article_id not in processed_articles:
            processed_articles[article_id] = {"title": hit.entity.get('title'), "link": hit.entity.get('link'), "published_time": hit.entity.get('published_time'), "relevant_chunk": hit.entity.get('chunk_text'), "similarity": hit.distance}

    # --- Stage 5: Display ---
    print(f"\n--- Top Search Results (Level: {relevance_level}, Threshold > {final_threshold:.4f}) ---")
    for i, article in enumerate(processed_articles.values()):
        print(f"\n--- Result {i+1} (Similarity: {article['similarity']:.4f}) ---")
        print(f"Title: {article['title']}")
        print(f"Link: {article['link']}")
        print(f"Published: {article['published_time']}")
        print(f"\nMost Relevant Snippet:\n{article['relevant_chunk']}\n---")

if __name__ == "__main__":
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        if not utility.has_collection(COLLECTION_NAME):
            logging.error(f"Collection '{COLLECTION_NAME}' not found.")
            exit()
        collection = Collection(name=COLLECTION_NAME)
        collection.load()
        logging.info(f"Collection '{COLLECTION_NAME}' loaded with {collection.num_entities} entities.")

        print("\nWelcome to the Tiered Semantic Search Demo!")
        while True:
            user_query = input("\nSearch Query (or 'exit') > ")
            if user_query.lower() == 'exit': break
            if not user_query.strip(): continue

            print("Select a relevance level:")
            print("  1: Highly Related (Strictest filter)")
            print("  2: Related (Balanced filter)")
            print("  3: Possibly Related (Broadest filter)")
            level_choice = input("Choice [1, 2, 3] > ")

            if level_choice == '1':
                level_str = "Highly Related"
            elif level_choice == '2':
                level_str = "Related"
            else:
                level_str = "Possibly Related"

            search_with_tiered_threshold(collection, user_query, level_str)

    except Exception as e:
        logging.critical(f"An error occurred: {e}", exc_info=True)
    finally:
        if "default" in connections.list_connections():
            connections.disconnect("default")
        logging.info("Disconnected from Milvus.")
