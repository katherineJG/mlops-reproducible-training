FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" MLFLOW_DISABLE_AGENT_HINT=1 GIT_PYTHON_REFRESH=quiet
WORKDIR /app

# dependencies first so this layer is cached until the lockfile changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev

# No .git in the image, so the MLflow git_sha tag is "unknown" here. Outputs land in /app/{data,models,reports,mlflow.db}.
CMD ["sh", "-c", "python -m churn_pipeline.ingest && python -m churn_pipeline.train"]
