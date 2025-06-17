FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY saver.py .
COPY ui.py    .

CMD ["bash", "-c", \
     "python saver.py & streamlit run ui.py --server.port 8501 --server.address 0.0.0.0"]