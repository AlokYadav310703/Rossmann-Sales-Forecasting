# Build from the REPOSITORY ROOT as context (not from inside backend/),
# since this image needs the sibling src/ and models/ folders too:
#   docker build -f backend/Dockerfile -t rossmann-backend .
#
# On Render: set "Dockerfile Path" = backend/Dockerfile and leave
# "Docker Build Context Directory" as the repo root (default).

FROM python:3.11-slim

WORKDIR /app

# Install backend-only dependencies first (better Docker layer caching --
# this layer is only rebuilt when requirements.txt changes, not on every code change)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the actual application code and trained model artifacts
COPY src/ ./src/
COPY backend/ ./backend/
COPY models/ ./models/

# So `from src...` and `from backend...` imports resolve correctly at runtime
ENV PYTHONPATH=/app

EXPOSE 8000

# $PORT is set automatically by Render (and most PaaS platforms);
# defaults to 8000 for local `docker run`
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
