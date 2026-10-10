# Aegis web UI container. Works on Hugging Face Spaces (port 7860) and on hosts that set $PORT (Render etc.).
# HF Spaces: python -m scripts.deploy_space <user>/<space>    Locally: docker build -t aegis . && docker run -p 7860:7860 aegis
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface AEGIS_HOST=0.0.0.0
WORKDIR /home/user/app

# CPU-only torch (the default wheel pulls ~3 GB of CUDA), then the versions the models were trained with
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt scikit-learn==1.8.0 numpy==2.4.6 pennylane==0.45.1

# bake the two small Hugging Face models in, so the first message does not download them
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/nli-deberta-v3-small')"

COPY --chown=user . .
RUN mkdir -p logs results && python -m scripts.setup_data
EXPOSE 7860
CMD ["sh", "-c", "AEGIS_PORT=${PORT:-7860} python -m web.server"]
