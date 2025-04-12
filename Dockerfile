FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]