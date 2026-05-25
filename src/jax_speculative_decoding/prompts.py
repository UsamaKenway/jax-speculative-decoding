from __future__ import annotations

from pathlib import Path

DEFAULT_PROMPTS = [
    "Explain speculative decoding for transformer inference in practical terms.",
    "Write a concise benchmark plan for comparing GPU inference throughput.",
    "Summarize why memory bandwidth matters for autoregressive decoding.",
    "Give three reasons same-family draft models can improve acceptance rate.",
    "Translate this short note into a crisp engineering status update: the benchmark is compiling kernels, loading model weights, and writing JSON metrics.",
    "Compare greedy decoding and sampling for language model inference, focusing on latency and reproducibility.",
    "Draft a short interview answer about why PCIe is acceptable when two GPUs exchange only token IDs.",
    "Explain how KV cache size changes with sequence length, batch size, number of layers, and attention head dimensions.",
    "Write a Python function that merges overlapping intervals and explain its time complexity.",
    "Solve this: a service receives 1,200 requests per second and each request uses 18 ms of GPU time; estimate required parallelism.",
    "Summarize the tradeoffs between tensor parallelism and pipeline parallelism for inference serving.",
    "Given a list of integers, explain how to find the longest increasing subsequence using dynamic programming.",
    "Write a short postmortem for an incident where model latency doubled after a deployment.",
    "Explain why batching improves throughput but can hurt time-to-first-token in interactive inference.",
    "Design a small experiment to compare BF16 and FP16 inference stability on an RTX 3090.",
    "Convert this product requirement into engineering tasks: add streaming responses, per-request tracing, and timeout handling.",
    "A transformer has 32 layers, 32 attention heads, and head dimension 128. Estimate KV cache elements for 2,048 tokens.",
    "Explain rotary position embeddings to a software engineer who knows attention but has not implemented RoPE.",
    "Write pseudocode for speculative decoding with one draft model and one target model.",
    "Compare greedy decoding, top-k sampling, and nucleus sampling for a chatbot used in customer support.",
    "Summarize a research paper abstract about model compression into five bullet points.",
    "Given two sorted arrays, write an algorithm to find their median without fully merging them.",
    "Explain how you would debug a CUDA out-of-memory error during long-context inference.",
    "Create a test plan for validating that two tokenizers produce identical token IDs for a model family.",
    "Describe how XLA compilation changes the first-request latency profile for a JAX service.",
    "Write a concise explanation of memory bandwidth roofline analysis for autoregressive decoding.",
    "A model server has p50 latency of 90 ms and p99 latency of 420 ms. Suggest three investigations.",
    "Explain the difference between prefill and decode phases in transformer inference.",
    "Draft a README section for reproducing benchmark results on a dual-GPU workstation.",
    "Given a buggy cache update in a decoder loop, explain how stale future KV positions can corrupt generation.",
    "Write a small SQL query to count daily active users from an events table.",
    "Explain why measuring asynchronous GPU programs requires explicit synchronization.",
    "Propose a resource allocation rule for choosing draft model size in speculative decoding.",
]


def make_prompt(tokenizer, input_len: int, text: str | None = None) -> str:
    """Create a prompt whose tokenized length is close to input_len."""
    if input_len <= 0:
        raise ValueError("input_len must be positive")

    seed = text or DEFAULT_PROMPTS[0]
    pieces = [seed]
    while True:
        candidate = "\n".join(pieces)
        token_count = len(tokenizer(candidate, add_special_tokens=False).input_ids)
        if token_count >= input_len:
            ids = tokenizer(candidate, add_special_tokens=False).input_ids[:input_len]
            return tokenizer.decode(ids, skip_special_tokens=True)
        pieces.append(seed)


def make_prompt_batch(tokenizer, input_len: int, batch_size: int, text: str | None = None) -> list[str]:
    prompts = []
    for idx in range(batch_size):
        seed = text or DEFAULT_PROMPTS[idx % len(DEFAULT_PROMPTS)]
        prompts.append(make_prompt(tokenizer, input_len, seed))
    return prompts


def load_prompt_seeds(path: str | Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS

    text = Path(path).read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if len(blocks) == 1:
        blocks = [line.strip() for line in text.splitlines() if line.strip()]
    if not blocks:
        raise ValueError(f"No prompts found in {path}")
    return blocks


def make_prompt_samples(
    tokenizer,
    input_len: int,
    num_samples: int,
    *,
    text: str | None = None,
    prompt_file: str | Path | None = None,
) -> list[str]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if text is not None:
        seeds = [text]
    else:
        seeds = load_prompt_seeds(prompt_file)

    return [make_prompt(tokenizer, input_len, seeds[idx % len(seeds)]) for idx in range(num_samples)]
