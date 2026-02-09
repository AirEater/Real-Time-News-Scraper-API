import os
import re
from pymongo import MongoClient
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
import torch

import sys
import os

# Add project root to Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from config.settings import DatabaseConfig, MilvusConfig

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
db_config = DatabaseConfig()
milvus_config = MilvusConfig()

# --- Embedding Model Configuration ---
# Using a powerful model with a higher dimension for better semantic capture.
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
EMBEDDING_DIM = 768 
# The max_length for the VARCHAR field in Milvus. Must match the schema.
CHUNK_MAX_LENGTH = 4000

# --- Milvus Schema Definition ---
def init_milvus_chunk_collection():
    """Initializes or gets a Milvus collection and ensures it is loaded."""
    connections.connect(host=milvus_config.host, port=milvus_config.port)
    
    if utility.has_collection(milvus_config.collection_name):
        logging.info(f"Found existing Milvus collection '{milvus_config.collection_name}'.")
        collection = Collection(milvus_config.collection_name)
    else:
        logging.info(f"Creating new Milvus collection '{milvus_config.collection_name}'.")
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="mongo_id", dtype=DataType.VARCHAR, max_length=36),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="published_time", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=CHUNK_MAX_LENGTH),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ]
        schema = CollectionSchema(fields, description="Collection for news article chunks")
        collection = Collection(name=milvus_config.collection_name, schema=schema)
        
        logging.info("Creating IVF_FLAT index on 'embedding' field...")
        collection.create_index(
            "embedding",
            {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
            index_name="embedding_idx"
        )
        logging.info("Index on embedding created.")

        logging.info("Creating scalar index on 'category' field...")
        collection.create_index("category", index_name="category_idx")
        logging.info("Index on category created.")

    logging.info(f"Loading collection '{milvus_config.collection_name}' into memory...")
    collection.load()
    logging.info("Collection loaded successfully.")
    return collection

# --- Text Processing ---
def recursive_text_splitter(text: str, separators: list[str], max_length: int) -> list[str]:
    """Helper function to recursively split a text chunk using a list of separators."""
    final_chunks = []
    # If the text is already small enough, return it as a single chunk.
    if len(text) <= max_length:
        return [text]

    # If we've run out of separators, do a hard character-level split.
    if not separators:
        final_chunks.extend([text[i:i+max_length] for i in range(0, len(text), max_length)])
        return final_chunks

    # Take the first separator
    current_separator = separators[0]
    remaining_separators = separators[1:]

    # Split the text by the current separator.
    splits = text.split(current_separator)
    
    current_chunk = ""
    for part in splits:
        # If the part itself is too long, it needs to be split further.
        if len(part) > max_length:
            # Before processing the long part, add what we have in the buffer.
            if current_chunk:
                final_chunks.append(current_chunk)
                current_chunk = ""
            # Recursively call the splitter on the oversized part.
            final_chunks.extend(recursive_text_splitter(part, remaining_separators, max_length))
        # If adding the next part would make the current chunk too long, finalize the current chunk.
        elif len(current_chunk) + len(part) + len(current_separator) > max_length:
            if current_chunk:
                final_chunks.append(current_chunk)
            current_chunk = part
        # Otherwise, merge the part into the current chunk.
        else:
            if current_chunk:
                current_chunk += current_separator + part
            else:
                current_chunk = part
    
    # Add the last remaining chunk.
    if current_chunk:
        final_chunks.append(current_chunk)
        
    return [c.strip() for c in final_chunks if c.strip()]

def split_text_intelligently(text: str, max_length: int) -> list[str]:
    """Splits text by single newlines, then applies recursive chunking to any oversized paragraphs."""
    if not text:
        return []

    final_chunks = []
    # Use single newline as the paragraph separator, per user's information.
    paragraphs = text.split('\n')

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        # If the paragraph is within the size limit, add it as is.
        if len(p) <= max_length:
            final_chunks.append(p)
        else:
            # If the paragraph is too long, use the recursive splitter on it.
            logging.warning(f"Paragraph of length {len(p)} is too long. Applying recursive chunking.")
            # Start recursive splitting with spaces, as newlines have been used.
            recursive_separators = [" ", ""]
            paragraph_chunks = recursive_text_splitter(p, recursive_separators, max_length)
            
            print("--- Recursive Chunking Results ---")
            for i, chunk in enumerate(paragraph_chunks):
                print(f"Chunk {i+1} (length: {len(chunk)}):\n---\n{chunk}\n---")

            final_chunks.extend(paragraph_chunks)
            
    return final_chunks

# --- Sentence Transformer Embedding ---
def get_sentence_transformer_embeddings(texts: list[str], model) -> list[list[float]]:
    """Generates embeddings for a list of texts using a Sentence Transformer model."""
    try:
        logging.info(f"Generating embeddings for {len(texts)} texts with model {EMBEDDING_MODEL}...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        return []

# --- Main Processing Logic ---
def embed_and_store_chunks(mongo_collection, milvus_collection, model, limit: int = -1):
    """
    Fetches all documents from MongoDB, chunks their content, creates embeddings, and inserts them.
    """
    query = {}  # Empty query to fetch all documents
    
    all_mongo_docs = list(mongo_collection.find(query).limit(limit)) if limit > 0 else list(mongo_collection.find(query))
    logging.info(f"Found {len(all_mongo_docs)} classified documents in MongoDB to process.")

    # Create a map of string-ID to full document for easy lookup later
    mongo_docs_map = {str(doc['_id']): doc for doc in all_mongo_docs}
    all_mongo_ids_str = list(mongo_docs_map.keys())

    new_ids_to_process = set()
    batch_size = 500 # Process in batches to avoid huge 'in' clauses

    for i in range(0, len(all_mongo_ids_str), batch_size):
        batch_ids = all_mongo_ids_str[i:i+batch_size]
        expr = f'mongo_id in {batch_ids}'
        
        try:
            results = milvus_collection.query(expr, output_fields=["mongo_id"])
            existing_ids_in_batch = {item['mongo_id'] for item in results}
            new_ids_in_batch = set(batch_ids) - existing_ids_in_batch
            new_ids_to_process.update(new_ids_in_batch)
        except Exception as e:
            logging.error(f"Failed to query Milvus for batch {i//batch_size + 1}: {e}. Assuming all in batch are new.")
            # As a fallback, assume all documents in this failed batch need processing.
            new_ids_to_process.update(batch_ids)

    docs_to_process = [mongo_docs_map[id_str] for id_str in new_ids_to_process]
    logging.info(f"Processing {len(docs_to_process)} new documents.")

    total_chunks_embedded = 0
    consecutive_failures = 0
    total_docs = len(docs_to_process)
    for i, doc in enumerate(docs_to_process):
        if consecutive_failures >= 3:
            logging.critical("Stopping process due to 3 consecutive insertion failures.")
            break

        logging.info(f"--- Processing document {i+1}/{total_docs}: {doc['_id']} ---")
        milvus_data_payload = []
        texts_to_embed = []

        mongo_id = str(doc['_id'])
        title = doc.get('title', '')
        content_html = doc.get('content', '')
        link = doc.get('link', '')
        published_time = str(doc.get('published_time', ''))
        # Ensure category is a string, defaulting to 'unclassified' if it's None to meet Milvus schema requirements.
        category = doc.get('category') or 'unclassified'

        # Pre-process HTML to Plain Text
        text_with_newlines = re.sub(r'<br\s*/?>', '\n', content_html, flags=re.IGNORECASE)
        plain_text_content = re.sub(r'<[^>]+>', '', text_with_newlines)

        content_chunks = split_text_intelligently(plain_text_content, 2048)
        for chunk in content_chunks:
            texts_to_embed.append(chunk)
            milvus_data_payload.append({"mongo_id": mongo_id, "title": title, "link": link, "published_time": published_time, "category": category, "chunk_text": chunk})
        
        if not texts_to_embed:
            logging.warning(f"Document {mongo_id} has no content to chunk. Skipping.")
            continue

        embeddings = get_sentence_transformer_embeddings(texts_to_embed, model)
        if len(embeddings) != len(texts_to_embed):
            logging.error(f"Embedding count mismatch for doc {mongo_id}. Skipping.")
            continue

        for j, data in enumerate(milvus_data_payload):
            data["embedding"] = embeddings[j]

        try:
            data_for_milvus = [[item[key] for item in milvus_data_payload] for key in ["mongo_id", "title", "link", "published_time", "category", "chunk_text", "embedding"]]
            milvus_collection.insert(data_for_milvus)
            total_chunks_embedded += len(milvus_data_payload)
            logging.info(f"Successfully inserted {len(milvus_data_payload)} chunks for doc {mongo_id}.")
            consecutive_failures = 0 # Reset on success
        except Exception as e:
            logging.error(f"Failed to insert chunks for doc {mongo_id} into Milvus: {e}")
            consecutive_failures += 1

    milvus_collection.flush()
    logging.info(f"Flushed data to Milvus. Total new chunks embedded: {total_chunks_embedded}")

def job(mongo_collection, milvus_chunk_collection, model):
    """Runs the embedding and storage process."""
    logging.info("--- Running embedding job ---")
    try:
        # Set a limit for testing, e.g., 100. Use -1 for no limit.
        embed_and_store_chunks(mongo_collection, milvus_chunk_collection, model, limit=-1)
    except Exception as e:
        logging.error("An error occurred during the job.", exc_info=True)
    logging.info("--- Embedding job finished ---")


if __name__ == "__main__":
    # --- One-time setup ---
    client = None
    try:
        # Determine the device to run the model on
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {device}")
        if device == 'cpu':
            logging.warning("CUDA not available, running on CPU. This may be significantly slower.")

        logging.info(f"Loading Sentence Transformer model: {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        logging.info("✅ Model loaded successfully.")

        client = MongoClient(db_config.mongo_uri)
        mongo_collection = client[db_config.mongo_db][db_config.mongo_collection]
        logging.info("✅ MongoDB connection successful.")

        milvus_chunk_collection = init_milvus_chunk_collection()
        
        # Run the job once and exit
        job(mongo_collection, milvus_chunk_collection, model)

    except Exception as e:
        logging.critical("An error occurred during the main process setup.", exc_info=True)
    finally:
        if client:
            client.close()
            logging.info("MongoDB connection closed.")
        if "default" in connections.list_connections():
            connections.disconnect("default")
            logging.info("Disconnected from Milvus.")