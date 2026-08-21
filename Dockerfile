FROM litellm/litellm:v1.99.0-dev.2

WORKDIR /app

# Install NeMo Guardrails — the only dependency not in the base image.
# Pin to a known-good version that matches local development.
RUN /app/.venv/bin/python -m ensurepip --upgrade && \
    /app/.venv/bin/python -m pip install --no-cache-dir "nemoguardrails>=0.23.0"

# Copy application files.
COPY nemo_guardrail.py .
COPY litellm_config.yaml .
COPY config/ ./config/

# LiteLLM proxy default port.
EXPOSE 4000

# Environment variables — override at runtime via --env-file or -e.
# OPENAICOMPATIBLE_API_KEY  — API key for the upstream model endpoint (required)
# UPSTREAM_BASE_URL         — base URL of the upstream OpenAI-compatible endpoint (required)
# HTTPS_PROXY / HTTP_PROXY  — corporate proxy if required
# GUARDRAIL_CONCURRENCY     — max parallel NeMo checks (default: 4)

CMD ["--config", "litellm_config.yaml"]
