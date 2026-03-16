FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY services/user_feature_svc/ services/user_feature_svc/

EXPOSE 8002

RUN useradd -m -u 1000 appuser
USER appuser

CMD ["uvicorn", "services.user_feature_svc.app.main:app", "--host", "0.0.0.0", "--port", "8002"]
