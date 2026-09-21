FROM python:3.10-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (CPU-only, no LLM) ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Build Vite frontend (install deps first for layer caching) ─────────────────
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci --silent

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

# ── Build Vite frontend (production bundle) ────────────────────────────────────
RUN cd frontend && npm run build
# Output: frontend/dist/ — served as static files by FastAPI

# ── Environment ───────────────────────────────────────────────────────────────
ENV PHASE=1
ENV PYTHONUNBUFFERED=1
# PORT is injected at runtime by the platform:
#   Render        → PORT=10000
#   Cloud Run     → PORT=8080
#   Local         → defaults to 8000 (set in app.py)

# ── Non-root user (security best practice) ────────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ── Expose default port (informational only — runtime PORT env overrides this) ─
EXPOSE 8080

# ── Start — reads PORT from environment ───────────────────────────────────────
CMD ["python", "app.py"]

