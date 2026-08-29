# Phase 1 — LLM Inference Engineering Sprint

**Timeline:** September 1–14, 2026  
**Environment:** Google Colab + Hugging Face  
**Goal:** Build a simple single-GPU LLM inference engine, establish a benchmark baseline, understand prefill/decode, and add a KV-cache-aware generation path.

## Final Phase 1 deliverable

By September 14, the project should support:

- Loading a small open-weight causal LM from Hugging Face
- Naive autoregressive generation
- Greedy / temperature / top-k / top-p sampling
- Basic latency benchmarking
- TTFT, TPOT, end-to-end latency, tokens/sec, and GPU-memory measurements
- A clean `ModelRunner` / `Generator` structure
- KV-cache-aware generation
- A small FastAPI `/generate` endpoint
- Streaming token output
- A concurrency experiment
- A README documenting architecture, measurements, and observations

## Recommended model

Start with a small model that fits comfortably in Colab GPU memory. Good choices include:

- Qwen 2.5/3 class ~0.5B–1.5B models
- Llama class ~1B–3B models
- Gemma small models

Use a model that is easy to download from Hugging Face and avoid large models initially. The objective is inference-engine understanding, not model size.

---

# Week 1 — Baseline inference

## Day 1 — Colab + model loading

Learn:

- GPU availability
- CUDA vs CPU
- BF16/FP16
- Hugging Face tokenizer/model loading
- Model parameter count
- GPU memory usage

Deliverable:

```text
Colab notebook
  ├── environment check
  ├── model download
  ├── tokenizer
  ├── model
  └── first successful generation
```

Checkpoint questions:

1. Which GPU did Colab give me?
2. How much VRAM does it have?
3. How much memory does my model consume?
4. What dtype is the model using?
5. How many parameters does the model have?

---

## Day 2 — Understand the forward pass

Trace:

```text
text
 ↓
tokenizer
 ↓
input_ids [B, S]
 ↓
embedding [B, S, H]
 ↓
Transformer layers
 ↓
attention
 ↓
MLP
 ↓
LM head
 ↓
logits [B, S, vocab_size]
```

Focus on tensor shapes.

Do not optimize anything yet.

Checkpoint questions:

- What does `B` mean?
- What does `S` mean?
- What is hidden size `H`?
- Why are logits shaped `[B, S, vocab_size]`?
- Why do we only use the last position's logits for the next token?

---

## Day 3 — Implement naive generation

Implement the simplest autoregressive loop yourself.

Conceptually:

```python
input_ids = tokenizer(prompt)

for _ in range(max_tokens):
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    next_token = sample(logits)
    input_ids = append(input_ids, next_token)
```

Implement initially:

- Greedy decoding
- Temperature
- Top-k
- Top-p

Do not use a generation abstraction as the core of the exercise.

Checkpoint:

```text
prompt
 ↓
token IDs
 ↓
forward pass
 ↓
last-token logits
 ↓
sampling
 ↓
next token
 ↓
append
 ↓
repeat
```

---

## Day 4 — Generation experiments

Run:

```text
max_tokens = 10
max_tokens = 50
max_tokens = 100
max_tokens = 500
```

Then vary input length:

```text
~100 tokens
~500 tokens
~1000 tokens
~2000 tokens
```

Record:

- input tokens
- output tokens
- total latency
- tokens/sec

Start thinking about:

```text
Prefill
Decode
```

---

## Day 5 — Build the benchmark

Create:

```text
benchmarks/
    benchmark_latency.py
```

Measure:

- TTFT
- TPOT
- total latency
- output tokens/sec
- GPU memory

Use CUDA events or appropriate synchronization when measuring GPU work.

Run multiple trials and use warm-up iterations.

Example experiment matrix:

| Input tokens | Output tokens |
|---:|---:|
| 100 | 100 |
| 500 | 100 |
| 1000 | 100 |
| 2000 | 100 |

Do not rely on a single timing measurement.

---

## Day 6 — Profile

Use:

- `nvidia-smi`
- PyTorch Profiler
- CUDA timing/events

Investigate:

```text
CPU preprocessing
GPU execution
attention
matmul
sampling
memory operations
```

Learn the difference between CPU timing and actual GPU execution timing.

Checkpoint:

> Where does my inference time go?

---

## Day 7 — Write the baseline report

Create/update:

```text
README.md
notes/inference.md
```

Document:

### Architecture

```text
Prompt
 ↓
Tokenizer
 ↓
Input IDs
 ↓
Transformer
 ↓
Logits
 ↓
Sampler
 ↓
Next token
 ↓
Repeat
```

### Benchmark

Record:

```text
Model:
GPU:
dtype:
batch size:
context length:
output length:

TTFT:
TPOT:
tokens/sec:
VRAM:
```

Answer:

1. What is TTFT?
2. What is TPOT?
3. What is prefill?
4. What is decode?
5. Why does generation require repeated forward passes?
6. Why does longer context consume more memory?
7. Why can GPU utilization be low?

---

# Week 2 — Turn the script into an inference engine

## Day 8 — Refactor

Separate responsibilities:

```text
ModelRunner
Generator
Sampler
Tokenizer
Config
```

Suggested structure:

```text
mini-inference-engine/
├── src/
│   ├── model_runner.py
│   ├── generate.py
│   ├── sampling.py
│   └── config.py
├── benchmarks/
├── tests/
├── notes/
└── README.md
```

Target API:

```python
runner = ModelRunner(...)
generator = Generator(runner, ...)

result = generator.generate(
    prompt="Explain CAP theorem",
    max_tokens=100,
)
```

---

## Day 9 — Add KV cache

Understand why recomputing all previous K/V states during every decode step is wasteful.

Target flow:

```text
Prefill
  ↓
K/V for prompt
  ↓
KV cache
  ↓
Decode new token
  ↓
new K/V
  ↓
append to cache
  ↓
next decode
```

Initially use the model's supported `past_key_values` / cache mechanism.

Do NOT implement paged attention yet.

---

## Day 10 — Benchmark KV cache

Run the same workload:

```text
WITHOUT KV CACHE
vs
WITH KV CACHE
```

Compare:

- TTFT
- TPOT
- total latency
- tokens/sec
- VRAM

Document why the numbers changed.

---

## Day 11 — HTTP API

Add FastAPI:

```http
POST /generate
```

Request:

```json
{
  "prompt": "Explain CAP theorem",
  "max_tokens": 100,
  "temperature": 0.7
}
```

Response:

```json
{
  "text": "...",
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 100
  }
}
```

---

## Day 12 — Streaming

Implement token streaming:

```text
token 1
token 2
token 3
token 4
...
```

Understand why perceived responsiveness depends heavily on TTFT even when total generation time is unchanged.

---

## Day 13 — Concurrency experiment

Send:

```text
1 request
2 requests
4 requests
8 requests
```

Do not implement continuous batching yet.

Measure:

- requests/sec
- output tokens/sec
- TTFT
- p95 latency
- GPU utilization
- VRAM

Use the experiment to identify why naive serving does not scale cleanly.

---

## Day 14 — Phase 1 assessment

You should be able to explain this architecture:

```text
Client
  ↓
FastAPI
  ↓
Generator
  ↓
ModelRunner
  ↓
Prefill
  ↓
KV Cache
  ↓
Decode
  ↓
Transformer
  ↓
Logits
  ↓
Sampler
  ↓
Token
  └────────→ Decode
```

You should confidently explain:

- TTFT
- TPOT
- prefill
- decode
- KV cache
- GPU memory
- GPU utilization
- batching
- sampling
- why inference becomes a systems problem under concurrency

---

# Daily schedule

Target **2.5–3 hours/day**:

```text
45 min  → theory
90 min  → coding
30 min  → benchmark/profile
15 min  → notes
```

Maintain:

```text
notes/inference.md
```

After every session answer:

1. What did I learn?
2. What surprised me?
3. What bottleneck did I find?
4. What experiment did I run?
5. What changed?
6. Why?

---

# Phase 1 rules

Do NOT work on these yet:

- Tensor parallelism
- Multi-GPU inference
- CUDA kernel implementation
- FlashAttention implementation
- Quantization
- PagedAttention
- Speculative decoding
- Triton kernels
- Kubernetes

Those belong to later phases.

The goal is to become extremely comfortable with **one model on one GPU** first.

---

# Success criteria

At the end of Phase 1 you should have:

```text
[x] Open-weight model loaded from Hugging Face
[x] Manual autoregressive generation
[x] Sampling
[x] Benchmark harness
[x] TTFT measurement
[x] TPOT measurement
[x] GPU memory measurement
[x] Profiling
[x] Clean inference runtime structure
[x] KV-cache generation
[x] FastAPI endpoint
[x] Streaming
[x] Concurrency experiment
[x] README with benchmark results
```

## Most important principle

For every optimization:

```text
Implement
   ↓
Benchmark
   ↓
Profile
   ↓
Explain
   ↓
Document
```

Do not optimize because a framework says an optimization is good.

Measure it yourself.

---

# What comes after Phase 1

Phase 2 should focus on:

```text
Continuous batching
        ↓
Request scheduler
        ↓
Paged KV cache
        ↓
Memory-aware scheduling
        ↓
Preemption
```

That is where the project starts resembling a real inference runtime such as vLLM.
