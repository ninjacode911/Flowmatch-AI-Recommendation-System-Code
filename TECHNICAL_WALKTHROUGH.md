# FlowMatch AI Recommendation System -- Technical Walkthrough

A detailed explanation of every component built across all three phases.
This document is written for learning -- it explains not just WHAT was built, but WHY and HOW.

---

## Table of Contents

- [Phase 1: MVP Foundation](#phase-1-mvp-foundation)
  - [1. Project Setup & Architecture](#1-project-setup--architecture)
  - [2. Shared Layer](#2-shared-layer)
  - [3. Synthetic Data Generation](#3-synthetic-data-generation)
  - [4. Embedding Service](#4-embedding-service)
  - [5. Candidate Service (Vector Store)](#5-candidate-service-vector-store)
  - [6. Phase 1 Recommendation Pipeline](#6-phase-1-recommendation-pipeline)
  - [7. API Gateway](#7-api-gateway)
  - [8. Evaluation Harness](#8-evaluation-harness)
  - [9. CI/CD Pipeline](#9-cicd-pipeline)
- [Phase 2: Core ML Models](#phase-2-core-ml-models)
  - [10. Two-Tower Retrieval Model](#10-two-tower-retrieval-model)
  - [11. Training Dataset](#11-training-dataset)
  - [12. Two-Tower Training Script](#12-two-tower-training-script)
  - [13. Neural Collaborative Filtering (NCF)](#13-neural-collaborative-filtering-ncf)
  - [14. Feature Engineering](#14-feature-engineering)
  - [15. LightGBM Learning-to-Rank](#15-lightgbm-learning-to-rank)
  - [16. Re-Ranking Service](#16-re-ranking-service)
  - [17. Full Pipeline V2 (Integration)](#17-full-pipeline-v2-integration)
- [Phase 3: Production Infrastructure](#phase-3-production-infrastructure)
  - [18. Event Collector Service](#18-event-collector-service)
  - [19. User Feature Service (Feature Store)](#19-user-feature-service-feature-store)
  - [20. Ranking Service (Model Serving)](#20-ranking-service-model-serving)
  - [21. LLM Augmentation Service](#21-llm-augmentation-service)
  - [22. Experiment Tracking (MLflow)](#22-experiment-tracking-mlflow)
  - [23. Prometheus Monitoring](#23-prometheus-monitoring)
  - [24. Docker & Docker Compose](#24-docker--docker-compose)
  - [25. Kubernetes Manifests](#25-kubernetes-manifests)
  - [26. Testing Strategy](#26-testing-strategy)
- [How It All Fits Together](#how-it-all-fits-together)

---

# Phase 1: MVP Foundation

## 1. Project Setup & Architecture

### Why a Monorepo?

We use a single repository containing all services. This is how companies like Google, Meta, and Uber organize large projects. Benefits:
- All code in one place -- easy to search, refactor, and keep consistent
- Shared code (schemas, config, utils) is imported directly, no package publishing
- One CI/CD pipeline tests everything together
- Atomic commits: a change that touches 3 services is ONE commit, not three

### The Service Architecture

The system is split into 10 microservices. Each service is a separate Python package with its own responsibility:

```
services/
  api_gateway/          --> The front door. Receives HTTP requests from clients.
  embedding_svc/        --> Converts text (titles, descriptions) into numerical vectors.
  candidate_svc/        --> Stores vectors in Qdrant and finds similar items (ANN search).
  training_pipeline/    --> Trains all ML models (Two-Tower, NCF, LightGBM).
  reranking_svc/        --> Reshuffles final recommendations for diversity.
  ranking_svc/          --> Serves the LTR model for online scoring.
  event_collector/      --> Ingests user events into Kafka.
  user_feature_svc/     --> Computes and serves real-time user features.
  llm_augment_svc/      --> Generates natural language explanations.
```

**Why separate services?** In production, each service:
- Can be scaled independently (embedding service needs more CPU, API gateway needs more instances)
- Can be deployed independently (update the ranking model without touching the API)
- Can fail independently (if the LLM service is down, recommendations still work, just without explanations)

### Configuration Files

**`pyproject.toml`** -- The central Python project config:
- Tells `ruff` (our linter) to use 120-character line width and Python 3.12
- Tells `pytest` to use async mode and look for tests in the `tests/` directory
- Tells `mypy` (type checker) which packages to check

**`requirements.txt`** -- Every Python library the project depends on:
- `fastapi` + `uvicorn` -- web framework and server
- `pydantic` + `pydantic-settings` -- data validation and config management
- `numpy` + `pandas` + `scikit-learn` -- data science basics
- `torch` -- PyTorch for neural network training
- `sentence-transformers` -- pre-trained text embedding models
- `qdrant-client` -- vector database client
- `lightgbm` -- gradient boosted trees for ranking
- `sqlalchemy` + `asyncpg` -- PostgreSQL ORM
- `redis` -- Redis client for caching
- `structlog` -- structured logging
- `prometheus-client` -- metrics collection

**`requirements-dev.txt`** -- Additional tools for development:
- `pytest`, `pytest-asyncio`, `pytest-cov` -- testing framework
- `ruff` -- linter and code formatter (replaces flake8, black, isort)
- `mypy` -- static type checker
- `pre-commit` -- runs checks before each git commit

**`docker-compose.yml`** -- Defines the infrastructure services:
```yaml
PostgreSQL 16   --> port 5432  (user data, item metadata, experiment configs)
Redis 7         --> port 6379  (feature cache, session data)
Qdrant v1.9.7   --> port 6333  (vector database for ANN search)
api-gateway     --> port 8000  (our FastAPI application)
```
Each service has a health check. The API gateway waits for all databases to be healthy before starting.

**`Makefile`** -- Command shortcuts:
```bash
make install      # pip install requirements
make lint         # ruff check .
make test         # pytest
make docker-up    # docker compose up -d
make run          # uvicorn api_gateway
```

**`.github/workflows/ci.yml`** -- GitHub Actions CI pipeline that runs on every push:
1. Set up Python 3.12
2. Install dependencies (with pip cache for speed)
3. Run `ruff check` (are there linting errors?)
4. Run `ruff format --check` (is the code properly formatted?)
5. Run `mypy` on the shared/ package (are there type errors?)
6. Run `pytest` with coverage reporting

**`.gitignore`** -- Tells git to never track:
- `__pycache__/`, `*.pyc` -- Python compiled files
- `.venv/` -- virtual environment (each developer has their own)
- `data/synthetic/*.npy`, `*.json` -- generated data files (large, reproducible)
- `models/artifacts/` -- trained model checkpoints (large, ~200MB total)
- `.env` -- environment variables with secrets
- `.claude/` -- Claude Code config

---

## 2. Shared Layer

Code that multiple services import. Lives in `shared/`.

### Settings (`shared/config/settings.py`)

Uses Pydantic's `BaseSettings` to define all configuration with defaults:

```python
class Settings(BaseSettings):
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "recsys"
    redis_host: str = "localhost"
    qdrant_host: str = "localhost"
    model_serving_host: str = "localhost"
    llm_api_key: str = ""
    ...
```

**Why Pydantic?** It validates types automatically. If someone sets `postgres_port = "not_a_number"`, it throws an error at startup instead of crashing later. It also reads from environment variables, so you can override settings in Docker/Kubernetes without changing code.

A singleton `settings = Settings()` is created once and imported everywhere.

### Schemas (`shared/schemas/`)

Define the exact shape of data using Pydantic models. These are used for:
1. **API validation** -- FastAPI automatically validates incoming requests against these
2. **Documentation** -- FastAPI auto-generates OpenAPI/Swagger docs from these
3. **Type safety** -- your IDE provides autocomplete and catches errors

**`recommendation.py`**:
```python
class SessionEvent(BaseModel):
    item_id: str
    event_type: str        # "view", "click", "add_to_cart", "purchase"
    timestamp: str | None

class RecommendationRequest(BaseModel):
    user_id: str
    session_events: list[SessionEvent] = []    # what the user did this session
    query: str | None = None                    # optional search query
    top_k: int = 10                             # how many recommendations

class RecommendedItem(BaseModel):
    item_id: str
    score: float           # relevance score (higher = better)
    title: str
    category: str
    explanation: str       # why we recommended this

class RecommendationResponse(BaseModel):
    user_id: str
    items: list[RecommendedItem]
    model_version: str
    explanation: str       # overall explanation of the strategy used
```

**`item.py`**: Item with item_id, title, description, category, tags, price, image_url, rating, popularity_score, created_at.

**`user.py`**: User with user_id, age, gender, location, preferred_categories, interaction_count.

### Logging (`shared/utils/logging.py`)

Sets up `structlog` which outputs structured JSON logs:
```json
{"event": "api_gateway.ready", "items_indexed": 50000, "timestamp": "2026-03-16T10:30:00Z"}
```

Instead of unstructured `print("loaded 50000 items")`, structured logs can be parsed, filtered, and queried in production monitoring systems (like Grafana Loki or AWS CloudWatch).

---

## 3. Synthetic Data Generation

### Why Synthetic Data?

We don't have real users or real purchases. But ML models need data to train on. So we generate fake data that has the same statistical properties as real e-commerce data.

### `scripts/generate_synthetic_data.py`

**Users (50,000)**:
- Each user belongs to one of **20 taste clusters**. A cluster defines which categories the user prefers.
- Age: normal distribution centered around 30 (std=10)
- Gender: randomly assigned (M/F/NB)
- Location: random city

**Items (50,000)**:
- Spread across **8 categories**: electronics, clothing, food, beauty, sports, home, toys, books
- **Popularity follows a power law**: A few items are mega-popular (like iPhone), most items have very few interactions. This is called the "long tail" and it's how real e-commerce works. Mathematically: `popularity ~ rank^(-alpha)` where alpha controls how steep the curve is.
- **Prices follow a log-normal distribution**: Most items are cheap ($10-50), a few are expensive ($200+). This matches real-world pricing.
- Each item has a generated title (brand + adjective + product type) and description.

**Interactions (2,000,000)**:
- Simulates a **conversion funnel**:
  ```
  100% of interactions are views
   -> 50% become views (the rest bounce)
   -> 30% become clicks
   -> 12% become add_to_cart
   -> 8% become purchases
  ```
  This funnel matches typical e-commerce conversion rates.

- **Category affinity**: Users have an **80%** chance of interacting with items in their cluster's preferred categories, and 20% with random items. This creates a stronger collaborative filtering signal for models to learn from.

- **Vectorized generation**: All random numbers are pre-generated in bulk using NumPy (`rng.integers`, `rng.random`, `rng.choice`), with only final dict assembly in the loop. This makes generating 2M interactions take ~1 minute instead of hanging indefinitely.

- **Seed = 42**: The random number generator uses a fixed seed, so running the script twice produces identical data. This is critical for reproducible experiments.

**Output format**: JSONL (one JSON object per line). Example:
```json
{"user_id": "user_000042", "age": 28, "gender": "F", "cluster_id": 7, ...}
{"item_id": "item_000001", "title": "NovaTech Premium Headphones", "category": "electronics", "price": 89.99, ...}
{"user_id": "user_000042", "item_id": "item_000001", "event_type": "click", "timestamp": "2026-01-15T14:23:00Z"}
```

---

## 4. Embedding Service

### The Core Problem

Computers can't understand text. They work with numbers. To find "similar items," we need to convert item text into numbers that preserve meaning.

### `services/embedding_svc/app/embedder.py`

**Model**: `all-MiniLM-L6-v2` from the SentenceTransformers library.
- Pre-trained on millions of text pairs from the internet
- 22M parameters, ~80MB download
- Input: any text string
- Output: a 384-dimensional vector (a list of 384 floating-point numbers)

**The key property**: Texts with similar meaning produce similar vectors.

```
"wireless noise-cancelling headphones" -> [0.12, -0.34, 0.78, ..., 0.45]  (384 numbers)
"Bluetooth ANC earbuds"                -> [0.11, -0.31, 0.76, ..., 0.43]  (384 numbers)
"organic cotton yoga pants"            -> [-0.56, 0.22, -0.18, ..., 0.67] (384 numbers)
```

The first two vectors are close together (cosine similarity = 0.374), while the third is far away (similarity = 0.158). The model learned that headphones and earbuds are semantically related, even though the words are different.

**How we use it**:

`embed_text(texts)` -- Encodes any list of strings into a (N, 384) numpy array. L2-normalized so cosine similarity = dot product (faster to compute).

`embed_catalogue(items_path, output_path)` -- Processes all 50,000 items in batches of 1000:
1. For each item, combines `title + ". " + description` into one string
2. Encodes all strings through the model
3. Saves the (50000, 384) matrix as `item_embeddings.npy`
4. Saves the item_id-to-index mapping as `item_embeddings.json`

This takes ~2 minutes on CPU. The .npy file is ~73MB.

**GPU support**: The RTX 5070 GPU (sm_120 Blackwell) requires PyTorch 2.10+ (CUDA 12.8) via WSL2. All models are trained on GPU with AMP mixed precision for ~2x speedup. The embedding service auto-detects GPU availability.

---

## 5. Candidate Service (Vector Store)

### The Problem of Scale

We have 50,000 item vectors. When a user makes a request, we need to find the most similar items. A brute-force approach would compare the query vector against all 50,000 items -- that's 50,000 dot products. It works, but it's slow and doesn't scale to millions of items.

### `services/candidate_svc/app/vector_store.py`

**Qdrant** is a vector database that solves this with **HNSW (Hierarchical Navigable Small World)** -- a graph-based approximate nearest neighbor algorithm.

**How HNSW works (simplified)**:
1. Items are organized into a multi-layer graph
2. Top layers have few items (like an express highway -- takes big jumps)
3. Bottom layers have all items (like local streets -- precise navigation)
4. To find nearest neighbors, start at the top layer, jump to the approximate region, then refine at lower layers
5. Complexity: O(log N) instead of O(N). For 50K items, this means ~17 comparisons instead of 50,000

**Our VectorStore class**:
- `__init__(in_memory=True)` -- For development, runs Qdrant entirely in memory (no Docker needed). In production, connects to a Qdrant Docker container.
- `create_collection(vector_size=384)` -- Creates a collection with cosine distance metric.
- `index_embeddings(embeddings, items)` -- Uploads vectors + metadata (title, category, price, rating, brand) in batches of 500.
- `search(query_vector, top_k=10)` -- Returns the top-K most similar items with scores.

Each search result includes:
```python
{
    "item_id": "item_042",
    "score": 0.892,        # cosine similarity (0 to 1)
    "title": "Premium Wireless Headphones",
    "category": "electronics",
    "price": 89.99,
    "rating": 4.5,
    "brand": "NovaTech"
}
```

---

## 6. Phase 1 Recommendation Pipeline

### `services/candidate_svc/app/pipeline.py`

The Phase 1 pipeline is simple -- content-based retrieval only.

**`RecommendationPipeline`** has three strategies:

**1. `recommend_by_query(query, top_k=10)`**
```
User types "wireless headphones"
  -> ItemEmbedder encodes it to a 384-d vector
  -> VectorStore.search() finds 10 most similar item vectors
  -> Return results
```

**2. `recommend_by_history(item_ids, embeddings, id_to_idx, top_k=10)`**
```
User viewed items [A, B, C]
  -> Look up their embedding vectors
  -> Average them: taste_vector = mean(emb_A, emb_B, emb_C)
  -> Normalize to unit length
  -> VectorStore.search() finds 10 most similar items
  -> Return results
```
The averaged vector represents the user's "taste" -- it's in the center of the items they liked.

**3. `recommend_popular(top_k=10)`**
```
New user, no history, no query
  -> Embed the text "popular trending bestseller product"
  -> Search for similar items (these tend to be popular items)
  -> Return results
```
This is the cold-start fallback. In Phase 3, we'll replace this with actual popularity rankings from Redis.

**`build_pipeline_from_data(data_dir)`** -- Convenience function that loads items.jsonl + item_embeddings.npy, creates an in-memory Qdrant index, and returns a ready pipeline.

---

## 7. API Gateway

### `services/api_gateway/app/main.py`

The entry point for the entire system.

**FastAPI application with lifespan -- auto-detecting pipeline version**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check for trained model artifacts
    model_dir = Path(__file__).resolve().parents[3] / "models" / "artifacts"
    has_models = (
        (model_dir / "two_tower_best.pt").exists()
        and (model_dir / "two_tower_item_embeddings.npy").exists()
        and (model_dir / "ltr_lightgbm.txt").exists()
    )

    if has_models:
        # Phase 2: Full ML pipeline (Two-Tower + LTR + Re-ranking)
        pipeline = build_pipeline_v2()
        app.state.pipeline = pipeline
        app.state.pipeline_version = "v2"
    else:
        # Phase 1 fallback: Content-based retrieval
        pipeline, embeddings, id_to_idx = build_pipeline_from_data()
        app.state.pipeline = pipeline
        app.state.embeddings = embeddings
        app.state.id_to_idx = id_to_idx
        app.state.pipeline_version = "v1"
    yield
```

**Auto-detection**: On startup, the gateway checks if trained model artifacts exist (`two_tower_best.pt`, `two_tower_item_embeddings.npy`, `ltr_lightgbm.txt`). If all three are present, it loads the full Phase 2 ML pipeline. Otherwise, it falls back to the Phase 1 content-based pipeline. This means the same codebase works before and after model training.

The pipeline is loaded once and stored on `app.state`. Every request handler accesses it from there -- no reloading per request.

**CORS middleware**: Allows requests from any origin (`allow_origins=["*"]`). In production, you'd lock this down to your frontend domain only.

### `routes/recommendations.py`

```python
@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: Request, body: RecommendationRequest):
```

**Pipeline V2 logic** (when model artifacts are available):
1. If `body.user_id` is a known user -> `pipeline.recommend(user_id)` -- full ML pipeline (Two-Tower retrieval -> LTR ranking -> MMR re-ranking)
2. If anonymous user with `body.session_events` -> `pipeline.recommend_by_history(item_ids)` -- history-based retrieval + re-ranking
3. Returns model_version="v2.0.0"

**Pipeline V1 fallback** (no model artifacts):
1. If `body.query` -> `recommend_by_query()` (semantic search)
2. Else if `body.session_events` -> `recommend_by_history()` (average embeddings -> ANN search)
3. Else -> `recommend_popular()` (cold-start fallback)
4. Returns model_version="mvp-0.1.0"

Returns a `RecommendationResponse` with the strategy name, model version, and list of `RecommendedItem` objects.

### `routes/health.py`

Two endpoints:
- `GET /health` -- always returns `{"status": "healthy"}` (tells load balancers the server is alive)
- `GET /health/ready` -- checks if the pipeline is loaded (tells Kubernetes the server is ready for traffic)

---

## 8. Evaluation Harness

### `scripts/evaluate.py`

Measures recommendation quality with standard Information Retrieval (IR) metrics.

**Protocol**:
1. Sample 500 users who have both views and purchases
2. For each user:
   - Input: their view + click history (up to 20 items)
   - Ground truth: items they actually purchased
   - Prediction: top-K recommendations from the pipeline
3. Compute metrics across all users

**Metrics explained**:

**Recall@K**: "Of everything the user bought, what fraction did we find in our top-K?"
```
User purchased: {A, B, C, D, E}  (5 items)
We recommended top-10: [X, A, Y, Z, C, ...]
Recall@10 = 2/5 = 0.40  (we found A and C out of 5 purchases)
```

**NDCG@K (Normalized Discounted Cumulative Gain)**: "Did we put the good items at the TOP of the list?"
```
Recommended: [irrelevant, RELEVANT, irrelevant, RELEVANT, ...]
Position 2 gets more credit than position 4 because users see top results first.
NDCG uses a logarithmic discount: credit = 1/log2(position + 1)
```

**Hit Rate@K**: "For what % of users did we find at least ONE relevant item in top-K?"
```
If ANY of the top-K items is in the user's purchase set -> hit = 1
Otherwise -> hit = 0
Average across all users = hit rate
```

**Coverage**: "What fraction of the 50K catalogue did we recommend across all users?"
```
If we only ever recommend the same 100 popular items -> coverage = 100/50000 = 0.2%
Higher coverage means we're recommending diverse items from the long tail.
```

**Phase 1 results**: All metrics ~0.0. This is expected because text similarity doesn't predict purchase behavior. An item about "premium headphones" is textually similar to other headphone items, but that doesn't mean someone who bought headphones will buy MORE headphones.

---

## 9. CI/CD Pipeline

### `.github/workflows/ci.yml`

Runs automatically on:
- Every push to `main` or `develop` branches
- Every pull request to `main`

**Steps**:
1. **Checkout code** -- `actions/checkout@v4`
2. **Setup Python 3.12** -- `actions/setup-python@v5` with pip cache
3. **Install dependencies** -- `pip install -r requirements-dev.txt`
4. **Lint** -- `ruff check .` (catches unused imports, undefined variables, style violations)
5. **Format check** -- `ruff format --check .` (ensures consistent code formatting)
6. **Type check** -- `mypy shared/` (catches type errors in the shared layer)
7. **Test** -- `pytest --cov` (runs all tests, reports code coverage)

If any step fails, the whole pipeline fails and you can't merge to main (if branch protection is enabled).

---

# Phase 2: Core ML Models

## 10. Two-Tower Retrieval Model

### The Problem with Content-Based Retrieval

Phase 1 used text similarity: "this item's description is similar to items you viewed." But this misses a critical signal: **collaborative filtering** -- "users like you bought this item."

Example: Someone who bought a camera might also want a camera bag. The text descriptions are completely different ("DSLR camera with 24MP sensor" vs "padded shoulder bag for camera equipment"), but there's a strong purchase correlation.

### The Two-Tower Architecture

The idea: learn **separate** neural networks for users and items that map them into the **same** 256-dimensional vector space. If a user would like an item, their vectors should be close together.

### `services/training_pipeline/app/models/two_tower.py`

**UserTower** -- Encodes user features into a 256-d vector:

```
Input:
  user_id (integer)    -> Embedding lookup table -> 64-d vector
  cluster_id (integer) -> Embedding lookup table -> 16-d vector
  age (float, normalized around mean=30, std=15)  \
  gender_M (0 or 1)                                |-> 4 features
  gender_F (0 or 1)                                |
  gender_NB (0 or 1)                              /

All concatenated: 64 + 16 + 4 = 84-dimensional input

Processing:
  Linear(84 -> 128)    # fully connected layer, learns weights
  ReLU()               # non-linearity: max(0, x), lets the network learn complex patterns
  BatchNorm(128)       # normalizes activations, stabilizes training
  Dropout(0.1)         # randomly zeros 10% of neurons during training (prevents overfitting)
  Linear(128 -> 256)   # final projection to embedding space
  L2-normalize         # makes the vector unit length (so dot product = cosine similarity)

Output: 256-d unit vector representing this user's preferences
```

**What are Embedding lookup tables?** Instead of representing user_id=42 as the number 42, we learn a unique 64-d vector for each user. There are 50,000 users, so this is a 50000x64 matrix. During training, the model adjusts these vectors so similar users get similar embeddings.

**ItemTower** -- Encodes item features into a 256-d vector:

```
Input:
  item_id (integer)       -> Embedding lookup table -> 64-d vector
  category (integer)      -> Embedding lookup table -> 16-d vector
  content_embedding       -> Linear(384 -> 64)      -> 64-d vector (compressed)
  price (float, normalized with mean/std)     \
  rating (float, centered at 3.0)              |-> 3 features
  popularity_score (float, 0-1)               /

All concatenated: 64 + 16 + 64 + 3 = 147-dimensional input

Processing:
  Linear(147 -> 256) -> ReLU -> BatchNorm -> Dropout(0.1) -> Linear(256 -> 256) -> L2-normalize

Output: 256-d unit vector representing this item
```

**Why compress content_embedding from 384-d to 64-d?** If we used all 384 dimensions directly, the content embedding would dominate the input (384 out of 531 total dimensions). The projection to 64-d lets the model decide how much to rely on text similarity vs. collaborative signals.

**TwoTowerModel** -- Combines both towers:

```python
def forward(self, user_ids, cluster_ids, user_features, item_ids, category_ids, content_embs, item_features):
    user_embs = self.user_tower(user_ids, cluster_ids, user_features)    # (batch, 256)
    item_embs = self.item_tower(item_ids, category_ids, content_embs, item_features)  # (batch, 256)
    return user_embs, item_embs
```

### In-Batch Sampled Softmax Loss

This is the training objective. It's elegant and efficient.

**Setup**: A batch of 512 (user, item) positive pairs. We compute:
- 512 user embeddings: `U` shape (512, 256)
- 512 item embeddings: `I` shape (512, 256)
- Similarity matrix: `S = U @ I.T / temperature` shape (512, 512)

**What S[i][j] means**: How similar user_i is to item_j. The diagonal `S[i][i]` is the positive pair (user_i actually interacted with item_i). Everything else is a negative pair.

**Loss**: Cross-entropy where the "correct class" for row i is column i (the diagonal).
```python
labels = [0, 1, 2, 3, ..., 511]  # each user should match their own item
loss = cross_entropy(S, labels)   # pushes diagonal scores up, off-diagonal down
```

**Why it works**: From one batch of 512, we get:
- 512 positive pairs (the real interactions)
- 512 x 511 = 261,632 negative pairs (all other combinations)
- No explicit negative sampling needed!

**Temperature (0.05)**: Dividing by a small number makes the softmax "sharper" -- the model must produce very high scores for positives to overwhelm the negatives. Without temperature, the scores might all be close and the model won't discriminate well.

**Symmetric loss**: We compute loss in both directions (user-to-item AND item-to-user) and average them. This ensures both towers learn equally.

---

## 11. Training Dataset

### `services/training_pipeline/app/dataset.py`

A PyTorch `Dataset` that loads all data files and serves (user, item) pairs for training.

**`InteractionDataset.__init__`**:
1. Loads `users.jsonl` -> builds `user_id_to_idx` mapping and stores user profiles
2. Loads `items.jsonl` -> builds `item_id_to_idx` mapping, tracks categories
3. Loads `item_embeddings.npy` -> (50000, 384) matrix of pre-computed content embeddings
4. Loads `interactions.jsonl` -> filters to click + add_to_cart + purchase events
5. Builds list of (user_id, item_id) positive pairs
6. **Deduplicates**: same (user, item) pair counted only once -> 999,581 unique pairs

**Normalization**: Raw numbers have different scales (age: 18-60, price: $5-$500, rating: 1-5). Neural networks train better when all inputs are roughly the same scale (~0 mean, ~1 std).
- Age: `(age - 30) / 15` -- centers at 30, most values between -1 and +2
- Price: `(price - mean) / std` -- standard z-score normalization
- Rating: `(rating - 3.0) / 1.0` -- centered at 3 stars
- Gender: one-hot encoded `[1,0,0]` for M, `[0,1,0]` for F, `[0,0,1]` for NB

**`__getitem__(idx)`** returns a dictionary of tensors ready for the model:
```python
{
    "user_id": tensor(42),                         # long
    "cluster_id": tensor(7),                       # long
    "user_features": tensor([0.13, 0, 1, 0]),      # float: [age_norm, gender_M, gender_F, gender_NB]
    "item_id": tensor(1337),                        # long
    "category_id": tensor(3),                       # long
    "content_emb": tensor([0.12, -0.34, ...]),      # float, 384-d
    "item_features": tensor([-0.5, 0.8, 0.3]),      # float: [price_norm, rating_norm, popularity]
}
```

---

## 12. Two-Tower Training Script

### `services/training_pipeline/app/train_two_tower.py`

Orchestrates the full training process.

**Data splitting**: 90% train, 10% validation. The validation set checks if the model generalizes to unseen (user, item) pairs. If train loss drops but val loss doesn't, the model is overfitting (memorizing instead of learning).

**DataLoader settings**:
- `batch_size=1024` -- large batches give more in-batch negatives (1024x1023 = ~1M negatives per batch)
- `shuffle=True` -- critical! Each batch must have diverse users/items for good negatives
- `drop_last=True` -- drops incomplete final batch (in-batch negatives need consistent batch size)
- `num_workers=4` -- multi-process loading on WSL2/Linux (0 on Windows)
- `pin_memory=True` -- pinned memory for faster CPU→GPU transfer
- `persistent_workers=True` -- keep worker processes alive between batches

**Optimizer: AdamW**
- Adam with decoupled weight decay
- Weight decay = 1e-4 (gently pushes weights toward zero, prevents overfitting)
- Learning rate = 1e-3 (standard starting point for Adam)

**Scheduler: Cosine Annealing**
- LR starts at 1e-3 and smoothly decays to 1e-6 following a cosine curve
- No need to manually pick step milestones
- Helps the model converge to a better minimum at the end of training

**Gradient clipping (max_norm=1.0)**: If gradients become very large (can happen with low temperature), clip them to magnitude 1.0. Prevents "exploding gradients" that destabilize training.

**Training loop**:
```
For each epoch (up to 100, with early stopping patience=12):
  For each batch:
    1. Move tensors to GPU (RTX 5070 via WSL2)
    2. Forward pass with AMP (float16): both towers produce embeddings
    3. Compute in-batch sampled softmax loss
    4. GradScaler scales loss, backward pass: compute gradients
    5. GradScaler unscales gradients, clip to max_norm=1.0
    6. GradScaler steps optimizer: update weights in float32
  Step the LR scheduler
  Run validation (no gradients, no weight updates, AMP inference)
  If val_loss improved -> save checkpoint, reset patience counter
  If no improvement for 12 epochs -> stop training early
```

**AMP (Automatic Mixed Precision)**: Forward and backward passes use float16 for ~2x GPU speedup, while weight updates remain in float32 for numerical stability. GradScaler handles loss scaling to prevent float16 underflow.

**Checkpoint saving**: We save the model state dict, optimizer state dict, and metrics. Only the best model (lowest val_loss) is kept. Early stopping with patience=12 prevents overfitting -- if validation loss doesn't improve for 12 consecutive epochs, training halts.

**Item embedding export**: After training, we run all 50K items through the trained Item Tower to get 256-d embeddings. These replace the 384-d content embeddings for ANN search. Saved as `two_tower_item_embeddings.npy`.

**Our results**:
- 6.5M parameters, 977 batches/epoch (batch_size=1024)
- Loss started at 6.41 (near random; `log(512) = 6.24`)
- Best val_loss = 6.2646 at epoch 5 out of 17 (early stopping triggered at patience=12)
- Training time: ~471 seconds on GPU (RTX 5070, AMP enabled)
- The model learned to distinguish the correct item from ~1024 in-batch negatives

---

## 13. Neural Collaborative Filtering (NCF)

### Why NCF in Addition to Two-Tower?

Two-Tower produces **embeddings** for ANN search -- it's a retrieval model. It's great at narrowing 50K items to 200 candidates.

NCF produces a **scalar score** (0 to 1) for any specific (user, item) pair -- it's a scoring model. It's more precise for re-ranking those 200 candidates.

### `services/training_pipeline/app/models/ncf.py`

**GMF (Generalized Matrix Factorization)**:
```
user_id -> Embedding(50000, 64) -> u_emb (64-d)
item_id -> Embedding(50000, 64) -> i_emb (64-d)

output = u_emb * i_emb  (element-wise product, 64-d)
```
This captures linear interaction patterns. It's like classic matrix factorization (SVD) but with learned embeddings instead of fixed decomposition.

**MLP (Multi-Layer Perceptron)**:
```
user_id -> Embedding(50000, 64) -> u_emb (64-d)
item_id -> Embedding(50000, 64) -> i_emb (64-d)

[u_emb | i_emb]  (concatenated, 128-d)
  -> Linear(128, 256) -> ReLU -> BatchNorm -> Dropout(0.2)
  -> Linear(256, 128) -> ReLU -> BatchNorm -> Dropout(0.2)
  -> Linear(128, 64)  -> ReLU -> BatchNorm -> Dropout(0.2)
output: 64-d
```
The MLP can learn **non-linear** patterns that GMF misses. Example: "users who buy expensive electronics also buy premium cables" -- this cross-price-category pattern requires non-linear reasoning.

**NeuMF (fused model)**:
```
[GMF output (64-d) | MLP output (64-d)]  (concatenated, 128-d)
  -> Linear(128, 1)
output: raw logit (no sigmoid -- required for AMP compatibility)
```
**Why no sigmoid?** AMP (Automatic Mixed Precision) doesn't support `binary_cross_entropy` with float16 inputs. By outputting raw logits and using `binary_cross_entropy_with_logits`, the sigmoid is fused into the loss function with numerically stable log-sum-exp, which is AMP-safe.

**Important**: GMF and MLP use **separate** embedding tables. This lets them learn different representations -- GMF embeddings capture linear factors, MLP embeddings capture non-linear relationships.

### `services/training_pipeline/app/train_ncf.py`

**Key difference from Two-Tower: Explicit negative sampling**

Two-Tower uses in-batch negatives (efficient, no sampling needed).
NCF needs explicit negatives because it produces a scalar score, not an embedding.

For each positive (user, item) pair where the user actually interacted:
```
Positive: (user_42, item_1337) -> label = 1.0
Negative: (user_42, item_random1) -> label = 0.0
Negative: (user_42, item_random2) -> label = 0.0
Negative: (user_42, item_random3) -> label = 0.0
Negative: (user_42, item_random4) -> label = 0.0
Negative: (user_42, item_random5) -> label = 0.0
Negative: (user_42, item_random6) -> label = 0.0
```
Ratio: 6 negatives per positive. The negative items are randomly sampled from items the user hasn't interacted with. Higher ratio (up from 4) gives the model more negative signal per epoch.

**Negatives are re-sampled each epoch**: Every epoch, we generate new random negatives. This prevents the model from memorizing specific negative examples.

**Loss**: Binary cross-entropy with logits (AMP-safe)
```
loss = -[label * log(sigmoid(logit)) + (1-label) * log(1-sigmoid(logit))]
```
Using `binary_cross_entropy_with_logits` instead of `binary_cross_entropy` -- the sigmoid is fused into the loss computation using a numerically stable log-sum-exp formulation, which is safe for AMP float16 training.

**Training**: 100 epochs max with early stopping (patience=10), AMP enabled, GPU-accelerated on RTX 5070 via WSL2.

**Our results**:
- 12.9M parameters (larger MLP: [256, 128, 64])
- ~7M training samples per epoch (999K positives x 7)
- Best at epoch 2 (val_loss = 0.3907)
- Early stopping triggered at epoch 12 (overfitting: train loss dropped while val loss rose)
- Training time: ~180 seconds on GPU (RTX 5070, AMP enabled)
- The epoch 2 checkpoint generalizes best

---

## 14. Feature Engineering

### Why Feature Engineering?

LightGBM (the ranking model) is a decision tree algorithm. Unlike neural networks that learn features automatically, decision trees need **pre-computed numerical features** that capture different signals.

Think of it this way:
- Neural networks are like giving someone raw ingredients and letting them cook
- Decision trees are like giving someone a scored rubric and letting them make a decision

### `services/training_pipeline/app/feature_engineering.py`

**`FeatureEngineer`** computes 32 features for each (user, item) pair:

**User Features (14 features)** -- "Who is this person?"

| Feature | How it's computed | What it captures |
|---------|-------------------|------------------|
| `user_age_norm` | `(age - 30) / 15` | Age group (younger users like different things) |
| `user_gender_M/F/NB` | One-hot encoding | Gender-based preferences |
| `user_cluster_id` | Raw cluster number | User segment membership |
| `user_interaction_count` | `log(1 + count)` | How active the user is (log-scale to handle power law) |
| `user_pref_electronics` | % of interactions in electronics | Category preference strength |
| `user_pref_clothing` | % of interactions in clothing | (same for all 8 categories) |
| ... | ... | ... |
| `user_avg_price_norm` | `(avg_price - mean) / std` | Is this a budget or premium shopper? |

**Item Features (12 features)** -- "What is this product?"

| Feature | How it's computed | What it captures |
|---------|-------------------|------------------|
| `item_price_norm` | `(price - mean) / std` | Price point relative to catalogue |
| `item_rating_norm` | `(rating - mean) / std` | Quality signal |
| `item_popularity` | Raw popularity score (0-1) | How well-known this item is |
| `item_interaction_count` | `log(1 + count)` | Total engagement (demand signal) |
| `item_cat_electronics` | 1 if electronics, else 0 | Category identity |
| `item_cat_clothing` | ... | (one-hot for all 8 categories) |

**Cross Features (5 features)** -- "How does this user relate to this item?"

| Feature | How it's computed | What it captures |
|---------|-------------------|------------------|
| `cross_cat_match` | 1 if item's category = user's top category | Direct category match |
| `cross_cat_preference` | User's preference % for this item's category | How much user likes this category |
| `cross_price_ratio` | item_price / user_avg_price | Is this item within the user's budget? |
| `cross_price_diff_norm` | \|item_price - user_avg_price\| / std | How far from typical spend |
| `cross_emb_popularity_bucket` | L2 norm of item's Two-Tower embedding | How "distinctive" this item is in embedding space |

**Why cross features matter**: Individual user and item features don't capture compatibility. A $500 item might be great for a premium shopper but wrong for a budget shopper. The `cross_price_ratio` captures this directly.

---

## 15. LightGBM Learning-to-Rank

### What is Learning-to-Rank (LTR)?

LTR is the final ranking stage. Given 200 candidate items from retrieval, LTR decides which 10 to show first.

### Why LightGBM?

LightGBM is a **gradient-boosted decision tree** framework. It's the industry standard for LTR because:
1. **Fast**: Trains in seconds/minutes, not hours like neural rankers
2. **Handles mixed features**: Works naturally with both categorical and numerical features
3. **Interpretable**: You can see which features matter most (feature importance)
4. **Robust**: Handles missing values, outliers, and different scales without careful preprocessing
5. **Directly optimizes NDCG**: The LambdaRank objective pushes relevant items to the top

### LambdaRank Objective

Normal classification: "Is this item relevant? Yes or no."
LambdaRank: "Given this LIST of items, which ORDER maximizes NDCG?"

The key insight: swapping two items near the top of the list matters more than swapping two items at positions 50 and 51. LambdaRank computes gradients that are weighted by the NDCG change that would result from swapping each pair of items.

### `services/training_pipeline/app/train_ltr.py`

**Data format for LTR**: Data is organized in "queries" (groups):
```
Query 1 (user_42):
  item_A: relevance=3 (purchased), features=[...]
  item_B: relevance=1 (clicked), features=[...]
  item_C: relevance=0 (no interaction), features=[...]
  item_D: relevance=2 (added to cart), features=[...]
  ...50 items total

Query 2 (user_99):
  item_X: relevance=3, features=[...]
  ...50 items total
```

**Relevance grades**: purchase=3, add_to_cart=2, click=1, view/no interaction=0

For each user, we include:
- All items they interacted with (positive, with relevance grades)
- Random items they didn't interact with (negative, grade=0)
- Up to 100 total candidates per user (expanded from 50 for richer training signal)

**Training/validation split**: 5000 users for training, 1000 for validation. Larger training set (up from 2000/500) improves generalization.

**LightGBM parameters**:
```python
{
    "objective": "lambdarank",        # LambdaRank loss function
    "metric": "ndcg",                 # evaluate with NDCG
    "eval_at": [5, 10, 20],           # track NDCG at these cutoffs
    "num_leaves": 63,                 # max leaves per tree (complexity)
    "learning_rate": 0.03,            # step size (lower for more trees, better convergence)
    "max_depth": 8,                   # max depth per tree
    "feature_fraction": 0.8,          # use 80% of features per tree (regularization)
    "bagging_fraction": 0.8,          # use 80% of data per tree (regularization)
    "lambdarank_truncation_level": 20 # optimize NDCG at top-20
}
```

**Early stopping**: If validation NDCG doesn't improve for 100 rounds, stop training (increased from 50 for more patience with the lower learning rate).

**Our results**:
- ~500K training pairs from 5000 users x 100 candidates
- Best NDCG@10 (validation) = 0.4169 (best iteration = 1)
- Most important features: `user_interaction_count`, `item_interaction_count`, `cross_price_diff_norm`, `item_price_norm`, `user_age_norm`
- Category preference features (`user_pref_food`, `user_pref_beauty`) are highly useful
- Note: Best iteration being 1 (single tree) indicates limited discrimination ability on synthetic data -- with real-world data containing genuine user preference patterns, LTR would build deeper ensembles

---

## 16. Re-Ranking Service

### The Problem

LTR maximizes relevance, but pure relevance makes boring recommendations:
- All 10 results are electronics (user's favorite category)
- No new/trending items appear
- No business objectives are met (promotions, inventory clearance)

### `services/reranking_svc/app/reranker.py`

**MMR (Maximal Marginal Relevance)** algorithm:

```
Given: 20 scored candidates, need to pick 10

Step 1: Pick the highest-scored item. Add to selected list.

Step 2: For each remaining item, compute:
  MMR(item) = 0.7 * relevance(item) - 0.3 * max_similarity(item, selected_items)

  The first term rewards relevance.
  The second term PENALIZES items too similar to what's already selected.

Step 3: Pick the item with highest MMR. Add to selected list.

Repeat steps 2-3 until we have 10 items.
```

`lambda=0.7` means we weight relevance at 70% and diversity at 30%. This prevents the list from becoming a monoculture while still showing relevant items.

**Similarity** is computed using cosine similarity between item embeddings. Two electronics items will have high similarity, so if one is already selected, the other gets penalized.

**Freshness boost**: Items with high popularity/trending scores get a small bonus (`freshness_weight=0.05`). This gently favors newer content.

**Business rules** (`BusinessRules` dataclass):
- `max_same_category=3` -- No more than 3 items from the same category
- `promoted_items` -- Set of item_ids that should get a score boost (for marketing campaigns)
- `promoted_boost=0.1` -- How much to boost promoted items
- Price range enforcement -- ensure mix of budget and premium items

**Category-diverse fallback**: When item embeddings aren't available (anonymous users), uses a simpler algorithm that greedily selects items while enforcing the max_same_category constraint.

**Explanation generation**: Each recommended item gets a human-readable explanation:
```
"[electronics] relevance=0.92, trending"
"[food] relevance=0.85, promoted"
```

---

## 17. Full Pipeline V2 (Integration)

### `services/candidate_svc/app/pipeline_v2.py`

Chains all Phase 2 components into one callable pipeline.

**`RecommendationPipelineV2`** holds references to:
- VectorStore (Qdrant)
- TwoTowerModel (for encoding users)
- LightGBM Booster (for scoring)
- FeatureEngineer (for computing features)
- Reranker (for diversity)
- User data and item embeddings

**`recommend(user_id, top_k=10, num_candidates=200)`**:

```
Stage 1: RETRIEVAL (~15ms)
  ├─ User Tower encodes user features -> 256-d vector
  └─ ANN search in Qdrant -> top 200 candidates

Stage 2: RANKING (~55ms)
  ├─ FeatureEngineer computes 32 features for each of 200 (user, item) pairs
  └─ LightGBM scores each candidate

Stage 3: RE-RANKING (~2ms)
  ├─ MMR diversity selection (lambda=0.7)
  ├─ Freshness boost
  └─ Business rules enforcement
  └─ Return top 10

Total latency: ~72ms
```

**`recommend_by_history(item_ids, top_k)`**: For anonymous users with session history but no user_id. Falls back to averaging item embeddings (like Phase 1) with diversity re-ranking.

**`build_pipeline_v2()`**: Loads all artifacts from disk:
1. Items and users from JSONL files
2. Two-Tower model checkpoint (loads weights, sets to eval mode)
3. Two-Tower item embeddings (50K x 256 matrix)
4. LightGBM model from saved text file
5. Initializes FeatureEngineer with interaction statistics
6. Creates Reranker with default settings
7. Indexes all item embeddings in Qdrant
8. Returns a ready-to-use pipeline

---

# How It All Fits Together

```
USER ACTION (view, click, purchase, search)
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
┌──────────────────────┐            ┌──────────────────────┐
│  EVENT COLLECTOR     │            │  API GATEWAY         │
│  POST /events        │            │  POST /recommend     │
│  Kafka / local buffer│            │  Routes to pipeline  │
└──────────┬───────────┘            └──────────┬───────────┘
           │                                   │
           ▼                                   ▼
┌──────────────────────┐            ┌──────────────────────┐
│  USER FEATURE SVC    │◄───────────│  PIPELINE V2         │
│  Redis feature store │  features  │                      │
│  Sessions, recents   │            │  Stage 1: RETRIEVAL  │  Two-Tower -> Qdrant ANN
└──────────────────────┘            │  200 candidates      │
                                    │          │           │
                                    │          ▼           │
                                    │  Stage 2: RANKING    │  Features -> LightGBM LTR
                                    │  RANKING SVC         │  POST /rank
                                    │  200 -> scored       │
                                    │          │           │
                                    │          ▼           │
                                    │  Stage 3: RE-RANK    │  MMR diversity + business rules
                                    │  scored -> top 10    │
                                    └──────────┬───────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │  LLM AUGMENT SVC     │
                                    │  POST /explain       │  "Why was this recommended?"
                                    │  POST /parse-query   │  "What did the user mean?"
                                    └──────────┬───────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │  RESPONSE to client  │  10 items + scores +
                                    │                      │  explanations
                                    └──────────────────────┘

MONITORING (always running):
  Prometheus (scrapes /metrics from all services every 15s)
  Grafana (dashboards for latency, throughput, errors)
```

**Offline (runs periodically)**:
```
generate_synthetic_data.py  ->  users.jsonl, items.jsonl, interactions.jsonl
embed_catalogue()           ->  item_embeddings.npy (384-d content vectors)
train_two_tower.py          ->  two_tower_best.pt + two_tower_item_embeddings.npy (256-d)
train_ncf.py                ->  ncf_best.pt
train_ltr.py                ->  ltr_lightgbm.txt
experiment_tracker.py       ->  logs params, metrics, artifacts to MLflow / JSON
```

**Online (runs per request)**:
```
User Tower encoding         ~5ms
ANN search (200 candidates) ~10ms
Feature store lookup         ~3ms
Feature engineering          ~50ms
LightGBM scoring            ~5ms
Re-ranking (MMR)            ~2ms
LLM explanation             ~50ms (API) / ~1ms (template)
─────────────────────────────────
Total                       ~75-125ms
```

**Infrastructure**:
```
Docker Compose (local dev):  10 containers (3 infra + 5 app + 2 monitoring)
Kubernetes (production):     12 manifests, 11 pods, auto-scaling ready
Prometheus:                  Scrapes RED metrics + ML metrics from all services
Grafana:                     Visualization and alerting
```

This architecture handles 50K items today and scales to millions with minimal changes (just add more Qdrant shards and increase num_candidates).

---

# Phase 3: Production Infrastructure

Phase 2 built the ML brain. Phase 3 wraps it in production-grade infrastructure: real-time event ingestion, low-latency feature serving, model serving APIs, LLM-powered explanations, monitoring, containerization, and orchestration.

---

## 18. Event Collector Service

### Why Event Collection?

Every user action (view, click, purchase, search) is a signal that the recommendation system can learn from. In production, these events drive:

1. **Real-time features**: "What did this user click in the last 5 minutes?" feeds into session-based recommendations
2. **Model retraining**: When enough new events accumulate, models are retrained with fresh data
3. **Analytics**: Conversion funnels, A/B test metrics, engagement tracking
4. **Monitoring**: If click-through rate drops suddenly, something is broken

### `services/event_collector/app/schemas.py`

Defines the event contract -- every client (web, mobile, API) sends events in this format:

```python
class EventType(str, Enum):
    VIEW = "view"              # User saw the item on screen
    CLICK = "click"            # User tapped/clicked to see details
    ADD_TO_CART = "add_to_cart"  # Strong purchase intent
    REMOVE_FROM_CART = "remove_from_cart"
    PURCHASE = "purchase"      # Strongest positive signal
    SEARCH = "search"          # Text query (captures intent)
    RATE = "rate"              # Explicit feedback
    SHARE = "share"            # Social signal

class UserEvent(BaseModel):
    event_id: str              # Server-generated UUID if not provided
    user_id: str               # Required: who did this
    item_id: str               # What item (empty for search events)
    event_type: EventType      # What happened
    timestamp: datetime        # When (defaults to now)
    session_id: str            # Browser/app session for grouping
    query: str                 # Search query text (for search events)
    position: int              # Where in the list the user clicked
    source: str                # "homepage", "search", "recommendations"
    device: str                # "web", "ios", "android"
    price: float               # Price at time of event
```

**Why these fields matter**:
- `position` tells us about position bias (users click top results more)
- `source` tells us which recommendation strategy drove the action
- `device` lets us serve different recommendations per platform

### `services/event_collector/app/producer.py`

The **EventProducer** is responsible for reliably delivering events to Kafka for downstream processing.

**Topic routing** -- different events go to different Kafka topics for independent processing:
```
purchase events  -> "purchases" topic   (high-priority, triggers immediate feature updates)
search events    -> "searches" topic    (feeds query understanding pipeline)
everything else  -> "user-events" topic (general event stream)
```

**Why separate topics?** Purchase events need immediate processing (update recommendation models, trigger post-purchase flows). Mixing them with high-volume view events would slow down processing.

**Dual-mode operation**:
- **Kafka mode (production)**: Events are sent to a real Kafka cluster with `acks="all"` (waits for all replicas to confirm), 3 retries, and 10ms batching (`linger_ms=10`) for throughput
- **Local mode (development)**: Events go to an in-memory `deque` buffer (max 10K events) so development doesn't require a Kafka cluster

**Resilience pattern**: If Kafka send fails, the event is buffered locally instead of being dropped. This gives at-least-once delivery -- the event will be sent when the connection recovers.

### `services/event_collector/app/main.py`

FastAPI application with three endpoints:
- `POST /events` -- ingest a single event (generates `event_id` if not provided)
- `POST /events/batch` -- ingest multiple events at once (more efficient for high-volume clients)
- `GET /events/stats` -- producer health: total events processed, buffer size, Kafka connection status

---

## 19. User Feature Service (Feature Store)

### What is a Feature Store?

In ML systems, "features" are the processed inputs to models. A feature store is a specialized database optimized for:
- **Low-latency reads** (features must be served in <5ms per recommendation request)
- **Point-in-time correctness** (don't leak future data into training)
- **Versioning** (roll back if a bad feature corrupts model predictions)

Think of it as a cache that sits between raw data and ML models.

### `services/user_feature_svc/app/feature_store.py`

**Redis key design** -- Redis is an in-memory key-value store, so key structure determines access patterns:

```
user:{user_id}:features:v1    -> JSON blob of pre-computed user features
item:{item_id}:features:v1    -> JSON blob of pre-computed item features
session:{session_id}:items    -> List of item IDs viewed in this session
user:{user_id}:recent         -> Sorted set of recent interactions (score = timestamp)
```

**Why these data structures?**

| Redis Type | Use Case | Why |
|------------|----------|-----|
| `STRING` (JSON) | User/item features | Single read gets all features at once |
| `LIST` | Session items | Ordered, append-only (tracks viewing sequence) |
| `SORTED SET` | Recent interactions | Score = timestamp enables "last N interactions" and time-range queries |

**TTL (Time-To-Live)** prevents stale features:
- User features: 24 hours (preferences change slowly)
- Item features: 6 hours (popularity/trending can shift faster)
- Sessions: 1 hour (sessions expire naturally)

**Batch operations** for efficiency:
- `get_item_features_batch()` uses Redis `MGET` -- one round-trip for N items instead of N round-trips
- `bulk_load_user_features()` uses Redis pipelines -- batches 1000s of writes into a single network call

**Local mode**: Mirrors Redis behavior using Python dicts, lists, and sorted lists. This means tests and development work without a Redis server, and the API is identical.

### `services/user_feature_svc/app/main.py`

Six REST endpoints that abstract the feature store:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/features/user/{user_id}` | GET | Get pre-computed user features |
| `/features/item/{item_id}` | GET | Get pre-computed item features |
| `/features/items` | POST | Batch get features for multiple items |
| `/features/session/{session_id}` | GET | Get items viewed in session |
| `/features/session/{session_id}` | POST | Track item viewed in session |
| `/features/recent/{user_id}` | GET | Get user's recent interactions |

**Why a service instead of direct Redis access?** Encapsulation. Other services don't need to know about Redis key format, TTLs, or versioning. If we migrate from Redis to DynamoDB tomorrow, only this service changes.

---

## 20. Ranking Service (Model Serving)

### Why a Separate Ranking Service?

In Phase 2, the LightGBM model was embedded inside the pipeline code. In production, model serving is separated because:
1. **Independent scaling**: Ranking is CPU-intensive; it may need more replicas than the API gateway
2. **Model updates**: Deploy a new model version without restarting the entire system
3. **A/B testing**: Route different users to different model versions

### `services/ranking_svc/app/main.py`

**Startup (lifespan)**:
1. Loads LightGBM model from `models/artifacts/ltr_lightgbm.txt`
2. Initializes FeatureEngineer with user/item/interaction data
3. Both stay in memory for the lifetime of the process

**`POST /rank`** -- the core scoring endpoint:
```
Input:  { user_id, candidate_item_ids: [...], top_k: 20 }

Steps:
  1. Build (user, item) pairs for all candidates
  2. FeatureEngineer computes 32 features per pair
  3. LightGBM predicts relevance scores
  4. Sort by score descending, return top_k

Output: { ranked_items: [{ item_id, score, rank }], model_version }
```

**`GET /rank/info`** -- returns model metadata:
- Number of features (32)
- Feature names (for debugging)
- Number of trees
- Best iteration (from early stopping)

This lets the API gateway verify it's talking to the correct model version.

---

## 21. LLM Augmentation Service

### Why LLM Augmentation?

Recommendations without explanations feel like a black box. Users trust recommendations more when they understand WHY something was recommended. An LLM generates natural language explanations.

Additionally, LLMs enable **natural language search**: "I need something for a rainy day hike" gets parsed into structured intent rather than just keyword matching.

### `services/llm_augment_svc/app/main.py`

**LLMClient** -- dual-mode like other services:

1. **API mode**: Calls Claude (Anthropic API) for high-quality explanations and query parsing
2. **Template mode**: Rule-based fallback that works without an API key

**Template-based explanations** (local mode):
```python
if i == 0:
    reason = "Top pick based on your browsing history"
elif score > 0.8:
    reason = f"Highly relevant to your interest in {category}"
elif price < 20:
    reason = f"Great value find in {category}"
else:
    reason = f"Popular in {category} -- others with similar taste loved this"
```

This covers the most common explanation patterns. The LLM does better for nuanced cases, but templates work for development and testing.

**Query parsing** (natural language understanding):

```
Input:  "recommend me affordable wireless headphones"

Output: {
    "intent": "recommendation",        # detected from "recommend"
    "categories": ["electronics"],      # inferred
    "price_range": "budget",           # detected from "affordable"
    "attributes": ["wireless"],        # detected keyword
    "refined_query": "recommend me affordable wireless headphones"
}
```

The template parser uses keyword detection:
- **Intent**: "recommend/suggest/best" -> recommendation, "compare/vs" -> comparison, ends with "?" -> question, else -> search
- **Price range**: "cheap/affordable/budget" -> budget, "premium/luxury" -> premium
- **Attributes**: Scans for known keywords like "wireless", "organic", "waterproof", "compact"
- **Categories**: Matches against known categories (electronics, clothing, food, etc.)

**Endpoints**:
- `POST /explain` -- generates explanations for a list of recommended items
- `POST /parse-query` -- parses natural language into structured search intent
- `GET /health` -- reports mode (template vs API) and status

---

## 22. Experiment Tracking (MLflow)

### Why Experiment Tracking?

When training models, you change hyperparameters, features, data splits, and training procedures. Without tracking, you lose track of which combination produced the best model.

### `services/training_pipeline/app/experiment_tracker.py`

**ExperimentTracker** wraps MLflow with a local JSON fallback:

```python
tracker = ExperimentTracker(experiment_name="two_tower_v2")

tracker.log_params({
    "learning_rate": 1e-3,
    "batch_size": 512,
    "temperature": 0.05,
})

tracker.log_metric("val_loss", 5.78, step=4)
tracker.log_metric("train_loss", 5.21, step=4)

tracker.log_artifact("models/artifacts/two_tower_best.pt")
tracker.set_tag("model_type", "two_tower")
```

**Dual-mode**:
- **MLflow mode**: Full MLflow tracking server with UI, artifact storage, model registry
- **Local mode**: Writes experiment logs to `experiments/{name}/run_{timestamp}.json` -- a simple JSON file with all params, metrics, and artifact paths

**Why local fallback?** MLflow requires a server (even locally). For quick experiments on a laptop, the JSON fallback gives you tracking without setup overhead. When you move to production, switch to MLflow for its UI and collaboration features.

---

## 23. Prometheus Monitoring

### Why Monitoring?

In production, you need to know:
- Is the system up? (availability)
- How fast is it responding? (latency)
- Are requests failing? (error rate)
- How many recommendations are being served? (throughput)

Without monitoring, you find out about problems when users complain.

### `shared/utils/metrics.py`

Implements the **RED method** (Rate, Errors, Duration) -- the standard for microservice monitoring:

**Standard HTTP metrics** (auto-collected by middleware):

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `http_requests_total` | Counter | method, path, status_code | Total request count |
| `http_request_duration_seconds` | Histogram | method, path | Latency distribution |
| `http_requests_in_progress` | Gauge | method, path | Currently active requests |

**ML-specific metrics** (manually instrumented in service code):

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `recommendation_latency_seconds` | Histogram | strategy | End-to-end recommendation time |
| `candidates_retrieved` | Histogram | -- | How many candidates from retrieval stage |
| `model_inference_duration_seconds` | Histogram | model | Time in model scoring (two_tower, ltr) |
| `events_ingested_total` | Counter | event_type | Events by type (view, click, purchase) |
| `feature_store_latency_seconds` | Histogram | operation | Redis read latency |

**PrometheusMiddleware** automatically wraps every FastAPI endpoint:
```python
# Before request
REQUEST_IN_PROGRESS.inc()
start = time.perf_counter()

# After request
REQUEST_COUNT.labels(method, path, status).inc()
REQUEST_DURATION.labels(method, path).observe(duration)
REQUEST_IN_PROGRESS.dec()
```

The `/metrics` endpoint returns all metrics in Prometheus text format for scraping.

**Histogram buckets** are tuned per metric:
- HTTP latency: [5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s]
- Model inference: [1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 500ms] (tighter, since inference should be fast)

### `infra/prometheus/prometheus.yml`

Prometheus scrape configuration -- tells Prometheus where to find metrics:

```yaml
scrape_configs:
  - job_name: "api-gateway"
    static_configs:
      - targets: ["api-gateway:8000"]
        labels:
          service: "api-gateway"
          tier: "gateway"
```

Each service is a separate scrape job with labels for filtering in Grafana. Scrape interval is 15 seconds -- frequent enough for alerting, not so frequent that it overloads services.

**Grafana** (port 3000) connects to Prometheus as a data source and provides dashboards for visualizing these metrics.

---

## 24. Docker & Docker Compose

### Why Containers?

Containers ensure "it works on my machine" is no longer a problem. Each service runs in an isolated environment with its exact dependencies.

### Dockerfiles

Each service follows the same pattern:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ ./shared/
COPY services/{service_name}/ ./services/{service_name}/
EXPOSE {port}
CMD ["uvicorn", "services.{service_name}.app.main:app", "--host", "0.0.0.0", "--port", "{port}"]
```

**Why `python:3.12-slim`?** The slim variant is ~150MB vs ~1GB for the full image. It includes everything needed to run Python but drops documentation, man pages, and build tools.

**Why copy `shared/` into every container?** All services import from `shared.config`, `shared.schemas`, and `shared.utils`. Each container needs its own copy since containers are isolated filesystems.

### `docker-compose.yml`

Orchestrates the entire stack locally:

```
Infrastructure (3 services):
  postgres:16-alpine     port 5432    -- persistent user/item data
  redis:7-alpine         port 6379    -- feature store cache
  qdrant:v1.9.7          port 6333/6334 -- vector search (REST + gRPC)

Application (5 services):
  api-gateway            port 8000    -- public entry point
  event-collector        port 8001    -- event ingestion
  user-feature-svc       port 8002    -- feature serving
  ranking-svc            port 8003    -- ML model serving
  llm-augment-svc        port 8004    -- LLM explanations

Monitoring (2 services):
  prometheus             port 9090    -- metrics collection
  grafana                port 3000    -- dashboards
```

**Health checks** ensure services start in the right order:
- Postgres: `pg_isready -U recsys`
- Redis: `redis-cli ping`
- Qdrant: `curl -f http://localhost:6333/healthz`
- Application services wait for their dependencies via `depends_on: condition: service_healthy`

---

## 25. Kubernetes Manifests

### Why Kubernetes?

Docker Compose works for local development, but production needs:
- **Auto-scaling**: Add more replicas when traffic spikes
- **Self-healing**: Restart crashed containers automatically
- **Rolling updates**: Deploy new versions with zero downtime
- **Resource limits**: Prevent one service from consuming all CPU/memory

### `infra/k8s/` Structure

12 manifest files organized by component:

**Cluster-level resources**:
- `namespace.yaml` -- `recsys` namespace isolates our workloads
- `configmap.yaml` -- shared environment variables (Redis host, Postgres host, etc.)
- `secrets.yaml` -- sensitive values (DB passwords, API keys) with base64 placeholders

**Infrastructure (StatefulSets for persistent storage)**:
- `postgres.yaml` -- StatefulSet with 5Gi PVC, readiness probe via `pg_isready`
- `qdrant.yaml` -- StatefulSet with 10Gi PVC, REST + gRPC service ports
- `redis.yaml` -- Deployment (ephemeral cache, no PVC needed)

**Why StatefulSets for databases?** StatefulSets provide stable network identities and persistent volumes. If a pod restarts, it reattaches to the same storage. Deployments create throwaway pods -- fine for stateless services, wrong for databases.

**Application services (Deployments)**:

| Service | Replicas | CPU Request | Memory Request | Why |
|---------|----------|-------------|----------------|-----|
| api-gateway | 2 | 250m | 256Mi | Public-facing, needs redundancy |
| event-collector | 2 | 200m | 256Mi | High-throughput ingestion |
| user-feature-svc | 2 | 200m | 256Mi | Read-heavy feature serving |
| ranking-svc | 2 | 500m | 512Mi | ML inference is CPU-intensive |
| llm-augment-svc | 1 | 200m | 256Mi | Low traffic, mostly API calls |

**Readiness probes** tell Kubernetes when a pod is ready to receive traffic:
```yaml
readinessProbe:
  httpGet:
    path: /health      # or /rank/health, /events/stats
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10
```

If the probe fails, Kubernetes stops sending traffic to that pod. This prevents routing requests to pods that are still loading models or connecting to databases.

**Monitoring**:
- `monitoring.yaml` -- Prometheus Deployment with ConfigMap for scrape config, Grafana Deployment
- Prometheus ConfigMap embeds the same scrape config as the Docker Compose version

**Service types**:
- `api-gateway`: `LoadBalancer` -- exposed to the internet
- Everything else: `ClusterIP` -- internal-only, accessible within the cluster

---

## 26. Testing Strategy

### Test Pyramid

Our tests follow the standard test pyramid:

```
         /  E2E  \         2 tests  -- full user journey across services
        /----------\
       / Integration \     23 tests -- HTTP endpoints per service
      /----------------\
     /    Unit Tests     \  39 tests -- individual components in isolation
    /______________________\

    Total: 64 tests, all passing
```

### Unit Tests

Test individual components without any network or framework:

- **`test_event_producer.py`** (9 tests): Topic routing logic, buffer behavior, stats reporting, batch sending
- **`test_feature_store.py`** (12 tests): User/item feature CRUD, session tracking, deduplication, sorted set behavior, bulk loading
- **`test_llm_client.py`** (10 tests): Template explanations, query intent detection, category/attribute/price parsing
- **`test_config.py`** (3 tests): Settings defaults, URL construction
- **`test_schemas.py`** (5 tests): Request/response model validation

### Integration Tests

Test full HTTP request/response cycle through FastAPI:

```python
# Example: sending an event and verifying the response
async def test_collect_single_event(client):
    payload = {"user_id": "user_001", "item_id": "item_100", "event_type": "click"}
    resp = await client.post("/events", json=payload)
    assert resp.status_code == 200
    assert resp.json()["events_received"] == 1
```

**Key technique**: Since `ASGITransport` doesn't trigger FastAPI lifespan events, each test file has an `autouse` fixture that manually initializes `app.state`:

```python
@pytest.fixture(autouse=True)
def _setup_app_state():
    app.state.producer = EventProducer(local_mode=True)
```

This ensures each test gets a clean state without needing a running server.

### E2E Tests

Test the complete user journey across multiple services:

1. User views items -> events sent to Event Collector
2. Session tracked in Feature Store
3. User features stored and retrieved
4. Recommendations explained by LLM service
5. Search query parsed into structured intent
6. Purchase event captured

Also tests **cold-start** (brand-new user with no history) to verify graceful degradation.
