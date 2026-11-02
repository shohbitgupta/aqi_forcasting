# Prediction service: the final XGBoost model behind a small FastAPI app.
#   docker build -t aqi-forecast .
#   docker run -p 8080:8080 aqi-forecast
FROM python:3.13-slim

# uv, the same version that wrote uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /bin/uv

# AWS Lambda Web Adapter: on Lambda it turns each request event into a normal HTTP request to the
# app on port 8080. Outside Lambda it does nothing, so `docker run` works as before.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_READINESS_CHECK_PATH=/health

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

# 1. Dependencies, exactly as in uv.lock (--frozen: fail rather than change a version).
#    Copied before the code, so this slow layer is rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

# 2. The service and the model, in the same folders as in the project, so predict.py finds
#    the model at ../models/. utility.py is needed to load it: the pickle refers to utility.CategoryXGB.
COPY scripts/predict.py scripts/utility.py ./scripts/
COPY models/xgb_final.bin ./models/

EXPOSE 8080
CMD ["uvicorn", "predict:app", "--app-dir", "scripts", "--host", "0.0.0.0", "--port", "8080"]
