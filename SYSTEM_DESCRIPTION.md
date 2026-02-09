# System Description: Semantic News Analysis and Subscription Platform

## 1. Overview

This project is a comprehensive, multi-stage news processing and delivery system designed to provide real-time, semantically relevant financial news. It automates the entire pipeline from data acquisition to intelligent retrieval and proactive delivery.

The system continuously scrapes news articles from `theedgemalaysia.com`, enriches them with financial context, generates vector embeddings for semantic understanding, and exposes this data through a API. The core features include semantic search, on-demand data processing, and a live subscription service that pushes relevant news to users via webhooks.

---

### Prequisites
```bash
# System Requirements
- Python 3.8+
- MongoDB
- Milvus 

# Install dependencies
pip install -r requirements.txt
```

### Configuration
```python
# config/API_Key.env, open router api key
OPENROUTER_API_KEY1=your_api_key_here
OPENROUTER_API_KEY2=your_api_key_here
# ... up to 5 keys for rotation
```

### Running the System
```bash
# Start the API server
python FYPII_Project/api/main.py

# Start automated schedulers
python FYPII_Project/schedulers/run_all_schedules.py

# Test webhook receiver (development)
python FYPII_Project/webhook_demo/test_receiver.py
```

---

## Core Components

FYPII_Project/
├── api/                   # FastAPI application layer
├── core/                  # Business logic and processing
│   ├── scrapers/          # News collection modules
│   ├── processors/        # Content enrichment pipeline
│   ├── search/            # Semantic search engine
│   └── utils/             # Shared utilities
├── config/                # Configuration management
├── schedulers/            # Automated task orchestration
├── data/                  # Company reference data
└── webhook_demo/          # Development tools

---

## 2. Core Technologies

- **Backend Framework**: FastAPI
- **Web Scraping**: `requests`, `BeautifulSoup4`, `selenium`
- **Databases**:
  - **MongoDB**: Primary data store for raw and processed news articles.
  - **Milvus**: Vector database for storing text embeddings to power high-speed semantic search.
- **AI & Machine Learning**:
  - **`sentence-transformers`**: Used for generating high-quality vector embeddings from article text (`BAAI/bge-base-en-v1.5`).
  - **OpenRouter API**: Leveraged for on-demand, LLM-based article summarization and classification.
- **Orchestration**: `schedule` library for running automated, periodic tasks.
- **Asynchronous Operations**: `asyncio` and `aiohttp` for handling concurrent operations, particularly for the subscription delivery system.

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI, Python 3.8+ |
| **Databases** | MongoDB (documents), Milvus (vectors) |
| **ML/NLP** | Sentence Transformers, OpenRouter LLM API |
| **Web Scraping** | Selenium, BeautifulSoup, requests |
| **Scheduling** | Python schedule library |
| **Deployment** | Uvicorn ASGI server |

---

## 3. System Architecture & Data Flow

The system is architected as a decoupled pipeline, ensuring modularity and scalability. The data flows through three main stages: Ingestion, Processing, and Serving.

### Core Components (/core/)

#### 1. API Layer (/api/)

- **main.py**: Primary FastAPI application providing semantic search and subscription management
- Offers dual search modes: standard (pre-processed) and on-demand (live classification)
- Real-time subscription system with webhook delivery
- RESTful endpoints for news retrieval, filtering, and subscription management

#### 2. Scrapers (/core/scrapers)

- `news_scraper.py`: News article extraction with incremental updates
- `company_downloader.py`: Bursa Malaysia company list downloader

#### 3. Processors  (/core/processors)

- `company_extractor.py`: Entity recognition for Malaysian public companies from `data/listed_companies.json`
- `embedding_processor.py`: Semantic vector generation using BGE embeddings

#### 4. Search  (/core/search/)

- `search_logic.py`: Semantic search with thresholding and filtering

#### 5. Utils  (/core/utils/)

- `company_matcher.py`: Company name matching with variations
- `llm_utils.py`: LLM-powered classification and summarization
- `scraper_utils.py`: Web scraping utilities with retry logic

#### 6. Configuration  (/config/)

- `settings.py`: Centralized configuration management for databases, models, and API keys

#### 7. Schedulers  (/schedulers/)

- `run_all_schedules.py`: Parallel execution of scraping and processing pipelines
- `run_processing_schedule.py`: Automated content processing workflow
- `run_scraper_schedule.py`: Scheduled news collection with configurable intervals

#### 8. Data Storate  (/data/)

- `listed_companies.json`: Malaysian public company directory with ticker mappings, downloaded from `(/core/processors/company_extractor.py)`

#### 9. Web Hook Testing (/webhook_demo/)

- `test_receiver.py`: Local webhook testing server for subscription validation

---

## API Endpoints

### Search Endpoints
```
POST /search/standard     - Standard semantic search (pre-processed data)
POST /search/on-demand    - Advanced search with live classification
GET  /all_news           - Browse all articles with filtering
```

### Subscription Management
```
POST /subscribe          - Create new subscription
POST /unsubscribe        - Remove subscription
GET  /subscriptions      - List active subscriptions
PUT  /subscription/{id}/pause   - Pause subscription
PUT  /subscription/{id}/resume  - Resume subscription
```

### Webhooks
```
POST /news_scraped_webhook  - Notify of new articles for immediate processing
```

---

## Technical Specifications

### Database Schema

**MongoDB Collections:**
- `news_the_edge_market`: Article documents with metadata
- `subscriptions`: User subscription configurations

**Milvus Collections:**
- `news_semantic_chunks`: Vector embeddings with metadata fields

---

### Stage 1: Data Ingestion (Scraping)

1.  **Scheduled Execution**: The `run_all_schedules.py` script launches two parallel schedulers. The `run_scraper_schedule.py` is responsible for data ingestion.
2.  **Company List Update**: The pipeline begins by running `company_downloader.py`, which uses Selenium to download the latest list of publicly traded companies from Bursa Malaysia and saves it as a local JSON file.
3.  **News Scraping**: The `news_scraper.py` is then triggered. It performs an incremental scrape of `theedgemalaysia.com`, fetching only new articles since its last run.
4.  **Initial Storage**: Raw article content is stored in a dedicated MongoDB collection (`news_the_edge_market`).

### Stage 2: Data Processing & Enrichment

1.  **Scheduled Execution**: The `run_processing_schedule.py` orchestrates the enrichment pipeline after the scraping stage.
2.  **Company Extraction**: The `company_extractor.py` processor reads new articles from MongoDB. It uses the previously downloaded company list to build an efficient regex matcher, identifies mentions of listed companies within the articles, and updates the MongoDB documents with the corresponding company names and stock tickers.
3.  **Embedding Generation**: The `embedding_processor.py` takes over. It splits the content of processed articles into smaller, semantically meaningful chunks.
4.  **Vector Storage**: It then uses the Sentence Transformer model to generate vector embeddings for each chunk. These embeddings, along with the chunk text and metadata, are stored in a Milvus collection (`news_semantic_chunks`), creating the foundation for semantic search.

### Stage 3: API & Data Serving

The `api/main.py` file defines the FastAPI application that serves as the primary user interface to the system.

1.  **Semantic Search**:
    -   **/search/standard**: Performs a fast semantic search on data that has already been fully processed and classified.
    -   **/search/on-demand**: An advanced endpoint that can perform real-time summarization and classification using the OpenRouter LLM for any articles that haven't been processed yet, providing the most up-to-date analysis at the cost of slightly higher latency.

2.  **Subscription & Live Delivery**:
    -   **/subscribe**: Users can subscribe to a specific search query by providing the query and a `callback_url`. The system stores this subscription.
    -   **Background Delivery Task**: A persistent background task (`check_and_deliver_news`) runs within the API, periodically checking for new articles that match the criteria of any active subscription.
    -   **Webhook Notification**: When a new article is scraped, the scraper can call the `/news_scraped_webhook` endpoint. This triggers an immediate check against all subscriptions.
    -   **Push Delivery**: If a match is found, the API system makes a POST request to the subscriber's `callback_url`, pushing the relevant news data in real-time.

3.  **Direct Data Access**:
    -   **/all_news**: Provides a conventional, non-semantic way to browse all news articles stored in the database, with standard filters for company, category, and date.

---

## 4. Key Features

- **Automated End-to-End Pipeline**: Fully automated system from scraping to delivery, managed by schedulers.
- **Advanced Semantic Search**: Go beyond keyword matching to find articles based on conceptual relevance and meaning.
- **Real-Time Push Notifications**: The subscription system allows users to receive news alerts via webhooks as soon as they are found and processed, rather than having to poll for updates.
- **On-the-Fly Data Enrichment**: The system can classify and summarize articles in real-time, ensuring that even the newest, unprocessed data is intelligently searchable.
- **Company-First Filtering**: A core feature allowing users to filter news by specific publicly traded companies before performing a semantic search.
- **Modular and Scalable Design**: Each component (scraper, processor, API) is independent, allowing for individual scaling and maintenance.

---

## Development History

The `archive(developing_files)` directory documents the project's evolution from simple command-line prototypes to a full-featured, asynchronous API. The progression is as follows:

1.  **Core Search Algorithm Prototyping (CLI Demos)**:
    *   `dev_semantic_search_chunks_demo.py`: The initial prototype. A simple CLI tool to perform a semantic search and return a list of the most relevant text *chunks* from Milvus.
    *   `dev_semantic_search_grouped_demo.py`: The first major improvement. This version introduced logic to group the resulting chunks by their parent article, ensuring that the final output showed only one result per news article, represented by its most relevant snippet.
    *   `dev_semantic_search_dynamic_threshold.py`: A more sophisticated prototype that moved away from a fixed number of results (`top_k`). It introduced a dynamic, multi-tiered thresholding system to determine relevance based on the statistical distribution of search scores, improving the quality and relevance of results.

2.  **API-ization and Feature Enrichment**:
    *   `semantic_search_api.py`: The first version of the search logic exposed as a web API. This marked the transition from a CLI tool to a service-oriented architecture.
    *   `company_matcher.py`: As the need for filtering grew, the company matching logic was refactored out of the main notebooks and into this reusable module to support company-based filtering in the API.
    *   `on_demand_api.py`: A significant functional upgrade. This version introduced the "on-demand" processing capability, allowing the API to call an external LLM to classify and summarize articles in real-time if they hadn't been processed by the batch pipeline yet.

3.  **Full-Fledged API and Subscription System**:
    *   `main_api.py`: A more mature FastAPI application that integrated the on-demand search logic, company filtering, and other parameters into a robust set of endpoints.
    *   `subscription-api.py`: The final and most complex iteration in the archive. This version built upon the `main_api` by adding the entire asynchronous subscription and webhook delivery system. It includes logic for managing subscribers, checking for new content, and pushing live updates, forming the blueprint for the final production API in `api/main.py`.