FROM python:3.11-slim

# ffmpeg needed for webm->wav conversion for Whisper; git optional
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ["ATR Model/requirements.txt", "/app/requirements.txt"]
# Reduce image size: preinstall minimal torch CPU (let pip resolve if needed)
# If transformers/sentence-transformers pull torch, they will install the minimal CPU build automatically.
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend and static assets
COPY ["ATR Model/backend", "/app/backend"]
COPY ["ATR Model/index.html", "ATR Model/script.js", "ATR Model/styles.css", "/app/"]

# Free-tier friendly defaults
ENV DISABLE_TTS=1
ENV WHISPER_MODEL=base
ENV PORT=8000

EXPOSE 8000
# Respect platform-provided PORT and allow low-memory startup via SKIP_STARTUP_LOAD
ENV SKIP_STARTUP_LOAD=1
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]



