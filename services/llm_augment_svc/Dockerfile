FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY services/llm_augment_svc/ services/llm_augment_svc/

EXPOSE 8004

RUN useradd -m -u 1000 appuser
USER appuser

CMD ["uvicorn", "services.llm_augment_svc.app.main:app", "--host", "0.0.0.0", "--port", "8004"]
