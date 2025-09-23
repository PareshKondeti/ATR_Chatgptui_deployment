FROM python:3.11-slim

# ffmpeg needed for webm->wav conversion for Whisper; git optional
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY "ATR Model/requirements.txt" /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend and static assets
COPY "ATR Model/backend" /app/backend
COPY "ATR Model/index.html" "ATR Model/script.js" "ATR Model/styles.css" /app/

# Free-tier friendly defaults
ENV DISABLE_TTS=1
ENV WHISPER_MODEL=base
ENV PORT=8000

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]


