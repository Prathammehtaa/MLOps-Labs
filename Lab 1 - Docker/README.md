# Penguin Species Classifier - Docker Compose Deployment

ML model deployment using Docker Compose with Flask API and Streamlit frontend.

## Technologies Used
- ML: Random Forest (scikit-learn)
- Backend: Flask REST API
- Frontend: Streamlit
- Deployment: Docker Compose with custom Dockerfiles
- Dataset: Palmer Penguins (344 samples, 3 species)

## Prerequisites
- Docker Desktop installed and running

## How to Run

Start all services:
```bash
docker-compose up --build
```

Access the application:
- Streamlit UI: http://localhost:8501
- Flask API: http://localhost:8000

Stop services:
```bash
docker-compose down
```

## Architecture

The application consists of 3 Docker services:
1. **model-training**: Trains Random Forest model, saves to shared volume
2. **flask-api**: REST API serving predictions on port 8000
3. **streamlit-app**: Frontend interface on port 8501

## Model Performance
- Accuracy: 100%
- Algorithm: Random Forest Classifier
- Features: culmen length/depth, flipper length, body mass, island, sex

## Project Customizations
- Dataset: Changed from Iris to Palmer Penguins
- Model: Random Forest instead of Neural Network
- Frontend: Streamlit with interactive sliders
- Architecture: Multi-service Docker Compose with custom Dockerfiles