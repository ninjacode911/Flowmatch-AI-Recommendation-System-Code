<div align="center">

# FlowMatch Backend

**10-microservice AI recommendation backend with GPU-trained Two-Tower, LightGBM LTR, and MMR models**

[![License](https://img.shields.io/badge/License-Source_Available-f59e0b?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python)](services/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?style=for-the-badge&logo=nvidia)](training/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ed?style=for-the-badge&logo=docker)](docker-compose.yml)

Trained on RTX 5070 via WSL2 + PyTorch 2.10 + AMP mixed precision.

</div>

---

## Overview

This is the complete ML backend for [FlowMatch](https://github.com/ninjacode911/Project-FlowMatch) — a production-grade AI recommendation engine. The backend is a monorepo with 10 microservices, 4 custom-trained ML models, and a 3-stage inference pipeline that processes 50K items in ~75ms.

Models were trained locally on an RTX 5070 via WSL2 with PyTorch 2.10, CUDA 12.8, and AMP mixed precision (FP16 forward pass, FP32 optimizer state).

---

## Architecture

```
Client Request
      |
      v
+----------------+     +--------------------------------------------+
|  Next.js       |---->|         API Gateway (FastAPI)              |
|  Frontend      |     |  Auto-detects V1 (content) or V2 (ML)     |
+----------------+     +--------------------+-----------------------+
                                            |
                     +----------------------v--------------------+
                     |              10 Microservices             |
                     |                                           |
                     |  User Service    Item Service    Auth     |
                     |  Two-Tower       LightGBM LTR    MMR      |
                     |  Feedback        Analytics       Events   |
                     |  Notification                             |
                     +----------------------+--------------------+
                                            |
                     +----------------------v--------------------+
                     |             Infrastructure                 |
                     |  PostgreSQL   Redis   Qdrant   Kafka       |
                     |  Prometheus   Grafana                      |
                     +-------------------------------------------+
```

---

## ML Models

| Model | Architecture | Training | Purpose |
|-------|-------------|---------|---------|
| **Two-Tower** | PyTorch, 6.5M params, 256-dim | RTX 5070, AMP FP16 | ANN candidate retrieval |
| **LightGBM LTR** | 500 trees, 32 features | GPU-accelerated, NDCG loss | Score and rank candidates |
| **MMR Reranker** | Maximal Marginal Relevance | No training required | Diversity re-ranking |
| **Content Filter** | Rule-based + embedding similarity | N/A | V1 content-based cold-start fallback |

---

## Features

| Feature | Details |
|---------|---------|
| **3-stage pipeline** | Two-Tower (~15ms) + LightGBM LTR (~55ms) + MMR (~2ms) = ~75ms total |
| **Dual API** | V1 content-based (cold-start) + V2 ML-based (warm users) |
| **10 microservices** | Each service independently deployable with Docker |
| **Event streaming** | Kafka for real-time user interaction events |
| **Observability** | Prometheus metrics + Grafana dashboards |
| **Vector search** | Qdrant for ANN search in Two-Tower retrieval stage |
| **Caching** | Redis for session management and result caching |
| **8 categories** | Electronics, clothing, food, beauty, sports, home, toys, books |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Gateway | FastAPI, Python 3.11+ |
| ML Training | PyTorch 2.10, CUDA 12.8, AMP (FP16 forward / FP32 optimizer) |
| ML Runtime | PyTorch inference, LightGBM, scikit-learn |
| Vector DB | Qdrant (Two-Tower ANN search) |
| Database | PostgreSQL (user and item data) |
| Cache | Redis (sessions, result cache) |
| Events | Kafka (interaction event streaming) |
| Monitoring | Prometheus + Grafana |
| Deployment | Docker Compose, Kubernetes-ready |

---

## Project Structure

```
services/
├── gateway/             # API Gateway - routes V1/V2 requests
├── user/                # User profiles, embeddings, history
├── item/                # Item catalog, features, embeddings
├── auth/                # JWT authentication
├── two-tower/           # Two-Tower inference service
├── lgbm/                # LightGBM LTR inference service
├── mmr/                 # MMR diversity re-ranking
├── feedback/            # Implicit/explicit feedback collection
├── analytics/           # Usage analytics + A/B testing
└── notification/        # User notifications

training/
├── two_tower.py         # PyTorch Two-Tower model + training loop
├── lgbm_ltr.py          # LightGBM LTR training
└── data_gen.py          # Synthetic dataset generation (50K users, 50K items)

infrastructure/
├── docker-compose.yml   # Full stack local deployment
└── k8s/                 # Kubernetes manifests
```

---

## Quick Start

```bash
git clone https://github.com/ninjacode911/Flowmatch-AI-Recommendation-System-Code.git
cd Flowmatch-AI-Recommendation-System-Code
docker-compose up -d     # Starts all 10 microservices
```

For model training (requires GPU + WSL2):

```bash
# Recommended: WSL2 with CUDA 12.8 + PyTorch 2.10
cd training
python data_gen.py       # Generate synthetic 50K user/item dataset
python two_tower.py      # Train Two-Tower (RTX 5070, ~2 hours)
python lgbm_ltr.py       # Train LightGBM LTR (~20 minutes)
```

See [Project-FlowMatch](https://github.com/ninjacode911/Project-FlowMatch) for the Next.js frontend.

---

## License

Source Available — All Rights Reserved. See [LICENSE](LICENSE) for details.
