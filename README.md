<h1>Movie/TV Review Classification Model</h1>

<h2>Table of Contents</h2>

<!-- TOC -->
  * [Overview](#overview)
    * [Project Structure](#project-structure)
    * [Descriptions](#descriptions)
  * [Run App](#run-app)
    * [Without Docker](#without-docker)
    * [With Docker](#with-docker)
      * [Build & Run](#build--run)
<!-- TOC -->

## Overview
- Build & deploy an LSTM sentiment classifier trained on Amazon movie and TV reviews and predicts sentiment: negative, neutral, positive.
- Generates simple GUI/webpage to allow the user to input new reviews for the classifier to predict in realtime.
- Utilizes llama3.1 LLM as a creative method to convey the predicted rating and review of their review to the user.

### Project Structure
```
├── docker-compose.yml
├── Dockerfile
├── init_container.sh
├── Movies_and_TV.jsonl
├── README.md
├── requirements.txt
├── supervisord.conf
├── classifier/
    ├── accuracy_plot.png
    ├── cross_entropy_plot.png
    ├── lstm_analyzer.log
    └── lstm_sentiment_classifier.pt
├── src/
     ├── constants.py
     ├── helper.py
     ├── lstm_analyzer.py
     ├── main.py
├── static/
    └── style.css
└── templates/
    ├── base.html
    └── index.html

```
### Descriptions
- `classifier`: Directory containing LSTM classifier weights (.pt) and information on training/validation performance.
- `src`: Directory containing Python code for project.
  - `lstm_analyzer.py`: Train and validation LSTM sentiment classifier.

## Run App
### Without Docker
```
cd src
uvicorn main:app --reload --port 8000
```

### With Docker
#### Build & Run
Builds and runs image/container in the background.
```
docker compose up -d --build
```
