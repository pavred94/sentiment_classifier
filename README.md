# Movie/TV Review Classification Model

## Project Overview
```angular2html
├── docker-compose.yml
├── Dockerfile
├── init_container.sh
├── lstm_sentiment_classifier.pt
├── Movies_and_TV.jsonl (Data for classifier)
├── requirements.txt
├── supervisord.conf
├── src/
    ├── helper.py
    ├── lstm_analyzer.py
    ├── main.py
    ├── model_classes.py
├── static/
    └── style.css
└── templates/
    ├── base.html
    └── index.html
```

## Docker
### Build & Run
Builds and runs image/container in the background.
```angular2html
docker compose up -d --build
```
