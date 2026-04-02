ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Install server-specific Python dependencies
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Copy the entire package into the container
COPY . /app/cybersec_soc_env

# Ensure the package is importable
ENV PYTHONPATH="/app:$PYTHONPATH"

# Runtime configuration (overridable at docker run / compose level)
ENV ENABLE_WEB_INTERFACE=1
ENV TASK_LEVEL=medium
ENV SEED=42

# Liveness probe: curl the health endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "cybersec_soc_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
