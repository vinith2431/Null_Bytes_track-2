# Aegis web UI for Hugging Face Spaces (Docker SDK, free CPU). Deploy: python -m scripts.deploy_space <user>/<space>
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    AEGIS_HOST=0.0.0.0 AEGIS_PORT=7860 HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# CPU-only torch (the default wheel pulls ~3 GB of CUDA), then the same versions the models were trained with
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt sentence-transformers \
        scikit-learn==1.8.0 numpy==2.4.6 pennylane==0.45.1

# bake the two small Hugging Face models in, so the first message on stage does not download them
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/nli-deberta-v3-small')"

COPY --chown=user . .
RUN python -m scripts.setup_data
EXPOSE 7860
CMD ["python", "-m", "web.server"]
