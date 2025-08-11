FROM python:3.12-slim

WORKDIR /app

# Install dependencies for Ollama & Supervisord
RUN apt-get update && apt-get install -y \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# ** Copy files into docker container **

# Install Python dependencies for app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code, review classifier weights
COPY src/ ./src
COPY static/ ./static
COPY templates/ ./templates
COPY lstm_sentiment_classifier.pt .

# Supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Startup script
COPY init_container.sh ./init_container.sh
RUN chmod +x ./init_container.sh

# Expose port - Uvicorn & Ollama
EXPOSE 8000
EXPOSE 11434

# Run startup script
CMD ["./init_container.sh"]