# Used Car Price Prediction & Analysis System — Streamlit app container
FROM python:3.12-slim

WORKDIR /app

# System deps needed to build a couple of the pinned wheels (xgboost/scipy) on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project. The trained model (models/), dataset (data/raw/)
# and generated reports (reports/) are already included in the repo, so
# the app runs immediately without retraining inside the container.
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
