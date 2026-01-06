FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY backend/requirements.txt backend/requirements.txt
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY backend backend
COPY frontend frontend

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
