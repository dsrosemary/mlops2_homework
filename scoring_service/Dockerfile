FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY kafka_app.py .
COPY src      ./src
COPY models   ./models

CMD ["python", "kafka_app.py"]