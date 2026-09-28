ARG VLLM_VERSION=0.30.0
FROM vllm/vllm-openai:v${VLLM_VERSION}

# The upstream image already owns the OpenAI-compatible server entrypoint.
# Running as its built-in non-root user keeps cache mounts OpenShift-friendly.
USER 2000:0
EXPOSE 8000

ENTRYPOINT ["vllm", "serve"]
CMD ["Qwen/Qwen3-Embedding-0.6B", "--runner", "pooling", "--host", "0.0.0.0", "--port", "8000", "--dtype", "float16"]
