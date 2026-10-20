# Mirrors the Render free plan (see render.yaml); CI boots this under a 512 MB memory limit.
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/zt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app . .
# WORKDIR is created as root; the app writes its model, vault, index and audit log here
RUN chown app:app /home/app/zt
USER app
# Fetch the embedding model into ./models at build time, not on the first request
RUN python scripts/prefetch_models.py

ENV PORT=5000 MALLOC_ARENA_MAX=2 ZT_MAX_INPUT_CHARS=30000
EXPOSE 5000
CMD gunicorn --worker-class gthread --threads 8 --workers 1 --timeout 120 --bind 0.0.0.0:$PORT app:app
