# Build from the REPOSITORY ROOT as context:
#   docker build -f frontend/Dockerfile -t rossmann-frontend .

FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt ./frontend/requirements.txt
RUN pip install --no-cache-dir -r frontend/requirements.txt

COPY frontend/ ./frontend/

EXPOSE 8501

# API_BASE_URL should be set to the deployed backend's URL at runtime
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
