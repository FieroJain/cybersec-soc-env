FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir openenv-core networkx numpy fastapi uvicorn pydantic

COPY . /app/cybersec_soc_env

ENV PYTHONPATH=/app
ENV TASK_LEVEL=medium
ENV SEED=42

EXPOSE 7860

CMD ["uvicorn", "cybersec_soc_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
