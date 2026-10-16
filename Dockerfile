# API image for Hugging Face Spaces (Docker SDK). Python matches the training
# venv so pickled models load with the exact library versions that wrote them.
FROM python:3.14-slim

# libgomp: OpenMP runtime LightGBM/XGBoost link against, absent from slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir --user -r requirements-api.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["sh", "deploy/start.sh"]
