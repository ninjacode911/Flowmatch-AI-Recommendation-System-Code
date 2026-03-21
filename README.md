<div align="center">

# FlowMatch Backend

**10-Microservice AI Recommendation System — GPU-Trained Two-Tower, LightGBM LTR, and MMR**

*The complete ML backend powering the FlowMatch recommendation engine.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Source%20Available-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](docker-compose.yml)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat&logo=nvidia&logoColor=white)](training/)

[**Frontend →**](https://github.com/ninjacode911/Project-FlowMatch)&nbsp;&nbsp;|&nbsp;&nbsp;[**Live Demo →**](https://projectflowmatch.vercel.app/)

</div>

---

## Overview

This is the complete ML backend for [FlowMatch](https://github.com/ninjacode911/Project-FlowMatch) — a production-grade AI recommendation engine. The backend is a monorepo with 10 independently deployable microservices, 4 custom-trained ML models, and a 3-stage inference pipeline that processes 50,000 items in approximately 75ms.

All models were trained locally on an RTX 5070 via WSL2 with PyTorch 2.10, CUDA 12.8, and AMP mixed precision (FP16 forward pass, FP32 optimizer state).

**What makes this different from typical recommendation backends:**
- **3-stage pipeline** — retrieval, ranking, and diversity re-ranking as separate, independently scalable microservices.
- **GPU training on consumer hardware** — Two-Tower trained with AMP mixed precision on an RTX 5070 via WSL2 + CUDA 12.8.
- **Dual API** — V1 content-based filter for cold-start users; V2 full ML pipeline for warm users with interaction history.
- **10 microservices** — each service independently deployable and horizontally scalable via Docker Compose or Kubernetes.
- **Full observability** — Prometheus metrics scraped from every service, visualized in Grafana.

---

## Architecture

```
Client Request
      |
      v
+----------------+      +--------------------------------------------------+
|  Next.js       | ---> |           API Gateway (FastAPI)                  |
|  Frontend      |      |  Detects V1 (content) or V2 (ML) route          |
+----------------+      +------------------------+-------------------------+
                                                  |
                         +------------------------v-------------------------+
                         |                10 Microservices                  |
                         |                                                   |
                         |  User Service     Item Service     Auth Service  |
                         |  Two-Tower        LightGBM LTR     MMR Service   |
                         |  Feedback         Analytics         Events        |
                         |  Notification                                     |
                         +------------------------+-------------------------+
                                                  |
                         +------------------------v-------------------------+
                         |                Infrastructure                     |
                         |  PostgreSQL    Redis    Qdrant    Kafka           |
                         |  Prometheus    Grafana                            |
                         +--------------------------------------------------+
```

---

## ML Models

| Model | Architecture | Training | Latency | Purpose |
|-------|-------------|---------|---------|---------|
| **Two-Tower** | PyTorch, 6.5M params, 256-dim | RTX 5070, AMP FP16, ~2h | ~15ms | ANN candidate retrieval over 50K items |
| **LightGBM LTR** | 500 trees, 32 features, NDCG loss | GPU-accelerated, ~20min | ~55ms | Score and rank 200 candidates |
| **MMR Reranker** | Maximal Marginal Relevance | No training required | ~2ms | Diversity re-ranking of top-K |
| **Content Filter** | Embedding similarity + rules | N/A | ~5ms | V1 cold-start fallback |

---

## Features

| Feature | Detail |
|---------|--------|
| **3-stage pipeline** | Two-Tower (~15ms) + LightGBM LTR (~55ms) + MMR (~2ms) = ~75ms total |
| **Dual API** | V1 content-based for cold-start users + V2 full ML pipeline for warm users |
| **10 microservices** | Each service independently deployable, scalable, and observable |
| **GPU training** | RTX 5070 + WSL2 + CUDA 12.8 + PyTorch 2.10 + AMP mixed precision |
| **Event streaming** | Kafka for real-time user interaction event ingestion |
| **Vector search** | Qdrant for ANN candidate retrieval in the Two-Tower stage |
| **Caching** | Redis for session state and result caching |
| **Observability** | Prometheus metrics from all services + Grafana dashboards |
| **8 categories** | Electronics, clothing, food, beauty, sports, home, toys, books |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API Gateway** | FastAPI, Python 3.11+ | Request routing, V1/V2 detection, rate limiting |
| **ML Training** | PyTorch 2.10, CUDA 12.8, AMP | GPU training with FP16 forward + FP32 optimizer |
| **ML Inference** | PyTorch, LightGBM, scikit-learn | 3-stage inference pipeline serving |
| **Vector DB** | Qdrant | ANN search in the Two-Tower retrieval stage |
| **Database** | PostgreSQL | User profiles, item catalog, interaction history |
| **Cache** | Redis | Session state, result caching, deduplication |
| **Events** | Kafka | Real-time user interaction event streaming |
| **Monitoring** | Prometheus + Grafana | Service metrics and dashboards |
| **Deployment** | Docker Compose, Kubernetes-ready | Local and cloud-native deployment |

---

## Project Structure

```
services/
├── gateway/             # API Gateway — routes V1/V2, rate limiting
├── user/                # User profiles, embeddings, interaction history
├── item/                # Item catalog, features, embeddings
├── auth/                # JWT authentication and authorization
├── two-tower/           # Two-Tower model loading and ANN inference
├── lgbm/                # LightGBM LTR model loading and scoring
├── mmr/                 # MMR diversity re-ranking
├── feedback/            # Implicit and explicit feedback collection
├── analytics/           # Usage analytics and A/B experiment tracking
└── notification/        # User notification dispatch

training/
├── data_gen.py          # Synthetic 50K user / 50K item dataset generation
├── two_tower.py         # PyTorch Two-Tower model definition + training loop
└── lgbm_ltr.py          # LightGBM LTR feature engineering + training

infrastructure/
├── docker-compose.yml   # Full stack local deployment (all 10 services)
└── k8s/                 # Kubernetes manifests for production deployment
```

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- For model training: WSL2 with CUDA 12.8 + PyTorch 2.10 (RTX GPU recommended)

### 1. Clone and start

```bash
git clone https://github.com/ninjacode911/Flowmatch-AI-Recommendation-System-Code.git
cd Flowmatch-AI-Recommendation-System-Code
docker-compose up -d     # Starts all 10 microservices
```

### 2. Train the models (GPU required)

```bash
# Run inside WSL2 with CUDA 12.8 + PyTorch 2.10
cd training
python data_gen.py       # Generate synthetic 50K user / 50K item dataset
python two_tower.py      # Train Two-Tower model (~2 hours on RTX 5070)
python lgbm_ltr.py       # Train LightGBM LTR model (~20 minutes)
```

### 3. Connect the frontend

See [Project-FlowMatch](https://github.com/ninjacode911/Project-FlowMatch) for the Next.js frontend that consumes this API.

---

## License

**Source Available — All Rights Reserved.** See [LICENSE](LICENSE) for full terms.

The source code is publicly visible for viewing and educational purposes. Any use in personal, commercial, or academic projects requires explicit written permission from the author.

To request permission: navnitamrutharaj1234@gmail.com

**Author:** Navnit Amrutharaj
