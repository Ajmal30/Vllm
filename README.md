# LLM Inference Engineering & Benchmarking with TinyLlama

This repository contains a Google Colab notebook (`.ipynb`) that serves as a hands-on guide to understanding and benchmarking Large Language Model (LLM) inference, focusing on custom generation implementations and performance analysis using PyTorch and Hugging Face Transformers.

## Project Overview

The primary goal of this project is to demystify the LLM inference process, moving beyond high-level API calls to a granular understanding of how models generate text and how their performance can be accurately measured and optimized. We use a small, open-weight causal language model, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, to facilitate rapid experimentation on a single GPU (e.g., NVIDIA Tesla T4).

## Key Topics & Implementations

1.  **LLM Environment Setup & Inspection:**
    *   Verification of CUDA availability and GPU memory.
    *   Installation of necessary libraries (`transformers`, `accelerate`, `sentencepiece`).
    *   Loading `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model and tokenizer in `bfloat16` precision.
    *   Detailed inspection of model parameters, data types, device placement, configuration, and GPU memory usage.

2.  **Causal Transformer Forward Pass Deep Dive:**
    *   Step-by-step explanation of a single inference forward pass within a causal transformer, detailing tensor shapes (`B`, `S`, `H`) at each stage (embeddings, RoPE, Multi-Head Attention with GQA, KV Cache, MLP, LM Head).

3.  **Custom Autoregressive Text Generation:**
    *   Manual implementation of the core autoregressive text generation loop, demonstrating how to manage `input_ids` and `past_key_values` (KV cache) for efficient token-by-token generation.
    *   **Decoding Strategies Explored:**
        *   **Greedy Decoding:** Selecting the token with the highest probability at each step.
        *   **Temperature Sampling:** Introducing randomness by scaling logits before softmax.
        *   **Top-K Sampling:** Constraining sampling to the top `k` most probable tokens.
        *   **Top-P (Nucleus) Sampling:** Dynamically selecting a minimum set of tokens whose cumulative probability exceeds `p`.
    *   In-depth explanation of the algorithm, tensor operations, and how each method transforms the logits/probability distribution.
    *   Comprehensive inspection prints at each generation iteration (sequence length, logits shape, selected token ID, decoded token, GPU device).

4.  **Accurate GPU Inference Benchmarking:**
    *   Explanation of why ordinary Python time measurements (`time.time()`) are misleading for asynchronous CUDA operations.
    *   Demonstration of correct GPU timing using `torch.cuda.synchronize()` and `torch.cuda.Event`.
    *   Implementation of a benchmark function to measure key inference metrics:
        *   **Time To First Token (TTFT):** Latency until the first token is generated.
        *   **Time Per Output Token (TPOT):** Average time for subsequent tokens.
        *   **Total Generation Latency:** End-to-end time for complete response.
        *   **Output Tokens Per Second (TPS):** Throughput of generated tokens.
        *   Latency Percentiles (`p50`, `p95`, `p99`) for consistency analysis.
    *   Incorporation of warm-up runs and multiple measurement runs for reliable results.

5.  **Experiment: Output Length vs. Latency:**
    *   Designed an experiment to analyze how varying output lengths (e.g., 10, 50, 100, 500 tokens) impact TTFT, TPOT, Total Latency, and Output TPS.
    *   Hypothesized and observed behavior of these metrics with increasing generation length.

## Getting Started

### Prerequisites

*   **Hardware:** An NVIDIA GPU (e.g., Tesla T4, A100, V100) with CUDA support is required for GPU benchmarking. The notebook is designed for Google Colab environments that provide a GPU.
*   **Software:** Python 3.8+ and standard data science libraries.

### Installation

The notebook handles most installations. If running locally, ensure you have:

```bash
pip install torch transformers accelerate sentencepiece numpy
```

### Running the Notebook

1.  **Open in Google Colab:** Click the "Open in Colab" badge (if available) or upload the `.ipynb` file to your Google Drive and open it with Colab.
2.  **Connect to GPU Runtime:** Ensure your Colab runtime is set to GPU (Runtime > Change runtime type > Hardware accelerator: GPU).
3.  **Execute Cells:** Run the notebook cells sequentially. Each section builds upon the previous one, introducing concepts and code implementations incrementally.

## Usage & Experimentation

*   **Explore Code:** Examine the custom generation loops to understand the mechanics of greedy, temperature, top-k, and top-p sampling.
*   **Modify Parameters:** Experiment with different `temperature`, `top_k`, and `top_p` values in the custom generation functions and benchmarks to observe their effects on generation quality and latency.
*   **Benchmark Your Own Model:** Adapt the benchmarking utility to evaluate the performance of other Hugging Face causal language models on your hardware.
*   **Extend Experiments:** Design further experiments, e.g., to analyze the impact of prompt length, batch size, or different hardware on inference metrics.

## Key Takeaways

*   LLM inference is an iterative, token-by-token process, where the KV cache is crucial for efficiency.
*   Sampling methods (temperature, top-k, top-p) allow fine-grained control over the creativity and coherence of generated text.
*   Accurate GPU benchmarking requires careful use of CUDA synchronization primitives (events) due to asynchronous execution.
*   TTFT is dominated by initial prompt processing, while TPOT governs streaming speed. Total latency scales linearly with output length.
*   Output TPS initially increases with length before plateauing, as fixed overheads become less significant.
