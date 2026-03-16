FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY services/ranking_svc/ services/ranking_svc/
COPY services/training_pipeline/ services/training_pipeline/
COPY models/artifacts/ models/artifacts/
COPY data/synthetic/ data/synthetic/

EXPOSE 8003

RUN useradd -m -u 1000 appuser
USER appuser

CMD ["uvicorn", "services.ranking_svc.app.main:app", "--host", "0.0.0.0", "--port", "8003"]
