FROM node:22-bookworm-slim AS web
WORKDIR /web
COPY tools/regression-ci/web/package.json ./
RUN npm install
COPY tools/regression-ci/web/ ./
RUN npm run build

FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    WEB_DIST=/app/web/dist/browser \
    RUNS_DIR=/data/runs \
    WORKSPACES_DIR=/workspaces \
    MNIST_CACHE_DIR=/data/mnist \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        graphviz \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirments.txt /tmp/requirments.txt
COPY tools/regression-ci/requirements.txt /tmp/regression-ci-requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/requirments.txt \
    && pip install -r /tmp/regression-ci-requirements.txt

COPY tools/regression-ci/worker /app/worker
COPY tools/regression-ci/jobs.json /app/jobs.json
COPY tools/regression-ci/baselines.json /app/baselines.json
COPY --from=web /web/dist /app/web/dist

RUN mkdir -p /data/runs /data/mnist /workspaces
EXPOSE 8080
CMD ["uvicorn", "worker.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
