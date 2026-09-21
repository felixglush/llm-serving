# Architecture boundaries

The lab follows a simple dependency direction:

```text
entrypoints -> application -> ports <- adapters
                  |
                  v
                domain
```

The domain describes concepts such as a model, a replica topology, a workload, and a benchmark
result. It is pure Python and has no imports from Modal, vLLM, `httpx`, or any cloud SDK.

Application use cases coordinate those concepts through small protocols in `ports/`. Adapters
implement the protocols for a particular boundary: a deterministic fake for offline tests, an
OpenAI-compatible vLLM HTTP client, or a Modal deployment definition. Entry points translate CLI
arguments into application calls.

The vLLM adapter also owns engine-specific scheduler settings. `ReplicaTopology` describes the
portable shape of a deployment, while `VllmEngineConfig` turns the topology for one replica into
vLLM command-line flags. See [the local vLLM smoke path](local-vllm-smoke.md) for the manual
end-to-end boundary check.

The important test is architectural: importing `embedding_lab.domain` must remain possible in an
environment that has none of the serving or cloud dependencies installed. Provider configuration
belongs in an adapter configuration object; domain objects use `tensor_parallel_size` to describe
one replica spread across workers rather than naming a vendor GPU string.
