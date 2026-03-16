FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY services/event_collector/ services/event_collector/

EXPOSE 8001

RUN useradd -m -u 1000 appuser
USER appuser

CMD ["uvicorn", "services.event_collector.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
