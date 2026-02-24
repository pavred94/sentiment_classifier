<h1>Movie/TV Review Classification Model</h1>
A sentiment analysis web application that classifies Amazon movie and TV reviews using a bi-directional LSTM neural network. The app provides real-time predictions with AI-powered responses using Llama 3.1:8b.

<h2>Table of Contents</h2>

<!-- TOC -->
  * [Overview](#overview)
    * [Project Structure](#project-structure)
    * [File Descriptions](#file-descriptions)
    * [Data](#data)
  * [Installation & Setup](#installation--setup)
    * [Prerequisites](#prerequisites)
    * [Without Docker](#without-docker)
    * [With Docker](#with-docker)
      * [Build & Run](#build--run)
<!-- TOC -->


## Overview

This project builds and deploys a bi-directional LSTM (Long Short-Term Memory) sentiment classifier trained on Amazon movie and TV reviews. The model predicts sentiment across three categories: negative, neutral, and positive.

### Features

* **Bi-directional LSTM Neural Network**: Deep learning model for accurate sentiment classification
* **Real-time Predictions**: Instant sentiment analysis through a web interface
* **AI-Enhanced Responses**: Utilizes Llama 3.1:8b LLM to generate creative, contextual responses based on predictions
* **Interactive Web UI**: Simple, user-friendly interface for inputting reviews
* **Docker Support**: Containerized deployment for easy setup and portability
* **Training Visualizations**: Accuracy and loss plots to monitor model performance

### Project Structure

```
├── Movies_and_TV.jsonl (LSTM classifier training/validation data)
├── README.md
├── classifier/
    ├── accuracy_plot.png
    ├── cross_entropy_plot.png
    ├── example_webpage_output.png
    ├── lstm_analyzer.log
    └── lstm_sentiment_classifier.pt
├── src/
     ├── constants.py
     ├── helper.py
     ├── lstm_analyzer.py
     └──  main.py
├── static/
    └── style.css
├── templates/
    ├── base.html
    └── index.html
└── docker/
    ├── docker-compose.yml
    ├── Dockerfile
    ├── init_container.sh
    ├── requirements.txt
    └── supervisord.conf
```

### File Descriptions

#### Classifier Directory
* `accuracy_plot.png`: Visualization showing training and validation accuracy over epochs
* `cross_entropy_plot.png`: Visualization showing cross-entropy loss during training and validation over epochs
* `example_webpage_output.png`: Screenshot of the web interface showing a sample prediction
* `lstm_analyzer.log`: Detailed logs from the training process
* `lstm_sentiment_classifier.pt`: PyTorch model checkpoint containing trained weights

#### Source Code Directory
* `constants.py`: Defines hyperparameters/configuration for bi-directional LSTM classifier
* `helper.py`: Contains utility functions for data preprocessing and text processing
* `lstm_analyzer.py`: Implements the bi-directional LSTM model architecture, training loop, and validation
* `main.py`: FastAPI application that serves the web interface and handles predictions

#### Templates & Static
* `base.html`: Base HTML template
* `index.html`: Main page featuring the review input form and results display
* `style.css`: CSS styles for the web interface

#### Docker Directory
* `docker-compose.yml`: Defines services, ports, and volumes for Docker deployment
* `Dockerfile`: Instructions for building the Docker image
* `init_container.sh`: Shell script to initialize the container environment
* `requirements.txt`: Python package dependencies
* `supervisord.conf`: Configuration for managing the Ollama and Uvicorn processes in the container

### Data

* **Source**: [Amazon Reviews 2023 Dataset](https://amazon-reviews-2023.github.io/)
* **Subset Used**: `Movies_and_TV` data from the "Grouped by Category" section
* **Format**: JSONL (JSON Lines) format with review text and ratings
* **Size**: Contains movie and TV product reviews from Amazon

The dataset includes:
- Review text
- Star ratings (1-5)
- Product metadata
- Reviewer information

**Sentiment Label Binning:**
- **Negative**: 1-2 stars
- **Neutral**: 3 stars
- **Positive**: 4-5 stars

## Installation & Setup

### Prerequisites

**Without Docker:**
* Python 3.12 or higher
* pip (Python package manager)
* Virtual environment (recommended)
* **Ollama 0.11.3** with Llama 3.1:8b model installed

**With Docker:**
* Docker Engine 29.1.3 or higher
* Docker Compose 5.0.1 or higher

### Without Docker

1. Clone the repository:
```bash
git clone https://github.com/pavred94/sentiment_classifier.git
cd sentiment_classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r docker/requirements.txt
```

4. Download the dataset:
```
# Download Movies_and_TV.jsonl from the Amazon Reviews 2023 dataset
# Place it in the project root directory
```

5. Install and setup Ollama:
- Linux: `curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.11.3 sh`
- macOS/Windows: Download v0.11.3 from [https://github.com/ollama/ollama/releases/tag/v0.11.3](https://github.com/ollama/ollama/releases/tag/v0.11.3)

**Pull the Llama 3.1:8b model:**
```bash
ollama pull llama3.1:8b
```

**Start Ollama server** (if not already running):
```bash
ollama serve
```

The Ollama API will be available at `http://localhost:11434`

6. Run the application:

From the project's root directory:

```bash
cd src
uvicorn main:app --reload --port 8000
```

The application will be available at `http://localhost:8000`

**Options:**
* `--reload`: Enable auto-reload on code changes (development mode)
* `--port 8000`: Specify the port (default: 8000)
* `--host 0.0.0.0`: Make the server accessible from other machines

### With Docker
#### Build & Run
Builds and runs image/container in the background. 
From project's root, execute the following shell commands:
```
cd docker
sudo docker compose up -d --build
```
