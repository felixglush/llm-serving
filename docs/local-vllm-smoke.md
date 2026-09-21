# Local vLLM embedding smoke path

This runbook proves that the application can talk to a real vLLM process without involving a
deployment backend. "Local" means that the server is started manually on a machine supported by
vLLM; it does not require that the client and GPU process share the same Python environment.

## Boundary being tested

```text
Qwen3 embedding model -> vLLM /v1/embeddings -> VllmEmbeddingService -> verification use case
```

`ReplicaTopology` remains provider-neutral. `VllmEngineConfig` translates its
`tensor_parallel_size` into the vLLM `--tensor-parallel-size` flag. A single `vllm serve` command
represents one replica; starting and routing multiple replicas belongs to a deployment backend.

## 1. Prepare a server environment

Keep the vLLM server environment separate from this project's client environment. For example, on
a supported Linux GPU machine:

```bash
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python vllm
```

The first server start downloads the model from Hugging Face. Confirm that the machine has enough
disk space and access to the model before continuing.

## 2. Start one replica on one GPU

These are conservative smoke-test values, not benchmark conclusions:

```bash
.venv-vllm/bin/vllm serve Qwen/Qwen3-Embedding-0.6B \
  --runner pooling \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --tensor-parallel-size 1
```

The flags correspond to:

- `max_model_len`: the longest input accepted by this server configuration.
- `max_num_seqs`: the most sequences vLLM may process in one scheduler iteration.
- `max_num_batched_tokens`: the token budget for one scheduler iteration.
- `tensor_parallel_size`: the workers that split one model replica; `1` uses one GPU worker.

vLLM can usually detect the pooling runner automatically. It is explicit here so the smoke path
documents that this process serves embeddings rather than generated text.

## 3. Verify readiness and the embedding contract

In another shell, use the project environment to check the server health and then exercise the
same application path used by other embedding-service adapters:

```bash
curl --fail http://127.0.0.1:8000/health

uv run embedding-lab verify \
  --service vllm \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-Embedding-0.6B \
  --expected-dimensions 1024 \
  --text "The capital of Canada is Ottawa." \
  --text "Gravity attracts objects with mass."
```

Success is a JSON result with `"passed": true`, two returned vectors, and 1,024 dimensions. The
verification use case also rejects non-finite, zero, or inconsistent vectors.

## 4. Tensor-parallel follow-up

On a machine with two compatible GPUs, keep one replica and change only:

```text
--tensor-parallel-size 2
```

This is not two replicas. It is one model replica whose tensors are distributed across two vLLM
workers. Performance comparison belongs to the later benchmark phases.

## Troubleshooting boundary

- A failed model download is an environment or access problem, not an embedding-client failure.
- An out-of-memory error should first be investigated by reducing sequence and batch limits.
- A connection error before `/health` succeeds is a server-startup problem.
- An invalid `/v1/embeddings` response is an adapter-contract problem and should be reproducible
  through `embedding-lab verify`.

The automated test suite validates configuration translation and the HTTP contract without
installing vLLM or requiring a GPU. This runbook is the explicit live integration check.
