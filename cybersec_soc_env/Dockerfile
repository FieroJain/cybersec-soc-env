FROM python:3.11-slim

WORKDIR /app

# Install all dependencies including Gradio and matplotlib for the dashboard
RUN pip install --no-cache-dir \
    openenv-core \
    networkx \
    numpy \
    fastapi \
    uvicorn \
    pydantic \
    gradio \
    matplotlib

COPY . /app/cybersec_soc_env

ENV PYTHONPATH=/app
ENV TASK_LEVEL=medium
ENV SEED=42
# Tell matplotlib to use non-interactive Agg backend (no display needed)
ENV MPLBACKEND=Agg

EXPOSE 7860

CMD ["uvicorn", "cybersec_soc_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
