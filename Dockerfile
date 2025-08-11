FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY src/ ./src
COPY static/ ./static
COPY templates/ ./templates
COPY lstm_sentiment_classifier.pt .

# Expose port
EXPOSE 8000

# Run the app
WORKDIR /app/src
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]