FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements-app.txt ./requirements-app.txt
RUN pip install --no-cache-dir -r requirements-app.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn deploy.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
