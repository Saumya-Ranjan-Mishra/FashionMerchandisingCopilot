FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  MODEL_METADATA_PATH=/app/fashion_model.metadata.json

WORKDIR /app

COPY deploy/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/inference_api.py /app/inference_api.py
COPY src/static/index.html /app/static/index.html
COPY checkpoints/fashion_model.metadata.json /app/fashion_model.metadata.json

EXPOSE 8080

CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8080"]
