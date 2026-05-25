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
