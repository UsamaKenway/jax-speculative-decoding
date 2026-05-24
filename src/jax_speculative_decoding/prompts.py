from __future__ import annotations

DEFAULT_PROMPTS = [
    "Explain speculative decoding for transformer inference in practical terms.",
    "Write a concise benchmark plan for comparing GPU inference throughput.",
    "Summarize why memory bandwidth matters for autoregressive decoding.",
    "Give three reasons same-family draft models can improve acceptance rate.",
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
