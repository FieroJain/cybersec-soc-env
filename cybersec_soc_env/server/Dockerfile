python -c "
content = '''FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH=\"/home/user/.local/bin:\$PATH\"

WORKDIR /app

RUN pip install --no-cache-dir openenv-core networkx numpy fastapi uvicorn pydantic

COPY --chown=user . /app/cybersec_soc_env

ENV PYTHONPATH=\"/app:\$PYTHONPATH\"
ENV ENABLE_WEB_INTERFACE=1
ENV TASK_LEVEL=medium
ENV SEED=42

EXPOSE 7860

CMD [\"uvicorn\", \"cybersec_soc_env.server.app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"7860\"]
'''
with open('c:/Users/HP/Documents/openenv/cybersec_soc_env/Dockerfile', 'w') as f:
    f.write(content)
print('Dockerfile created!')
"