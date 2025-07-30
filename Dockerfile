# Stage 1: Build and Data Gen
FROM python:3.10-slim AS builder

WORKDIR /app

# Copy the entire project context
COPY . .

# Install the project and all its dependencies + CLI
RUN pip install --no-cache-dir .

# Run examples scripts
RUN python examples/train_model.py
RUN python examples/minimal_tracking.py
RUN python examples/sweep/sweep.py

# Stage 2: Image
FROM python:3.10-slim

WORKDIR /app

RUN addgroup --system app && adduser --system --group --home /app app
ENV HOME=/app
ENV PYTHONPATH=/app

# Copy installed Python packages and data from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder --chown=app:app /app/.mlruns /app/.mlruns
COPY --from=builder --chown=app:app /app/.registry /app/.registry

# Copy the application code and entrypoint
COPY --chown=app:app app/ ./app
COPY --chown=app:app .streamlit/ .streamlit/

USER app

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["python", "app/docker-entrypoint.py"]
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=error"]