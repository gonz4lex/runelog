## Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc

COPY app/requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


## Final Stage
FROM python:3.10-slim

WORKDIR /app

# Create a non-root user and its home directory
RUN addgroup --system app && adduser --system --group --home /home/app app

ENV HOME=/home/app
ENV PATH="/home/app/.local/bin:${PATH}"

USER app

COPY --from=builder /app/wheels /wheels/

RUN pip install --user --no-cache /wheels/*

# Copy RuneLog source code and Streamlit app to ensure the image has access 
COPY --chown=app:app src/runelog/ ./runelog
COPY --chown=app:app app/ ./app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
