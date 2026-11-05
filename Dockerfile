# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Install dependencies in a separate layer so they are cached between builds.
FROM python:3.12-slim AS builder

WORKDIR /app

# System deps needed to compile some packages (e.g. numpy C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only what the API needs at runtime
COPY src/        ./src/
COPY models/     ./models/
COPY data/eda_meta.json     ./data/eda_meta.json
COPY data/feature_meta.json ./data/feature_meta.json

# Run as non-root for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Cloud Run injects PORT at runtime (default 8080)
ENV PORT=8080
EXPOSE 8080

# Uvicorn: bind to 0.0.0.0 so Cloud Run can reach the container
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 1"]
