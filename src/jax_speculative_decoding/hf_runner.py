from __future__ import annotations

import time

from .hf_loader import load_causal_lm
from .prompts import make_prompt_batch
from .results import BenchmarkResult
from .timing import tokens_per_second


def run_hf_baseline(
    *,
    model_id: str,
    device_index: int,
    input_len: int,
    output_len: int,
    batch_size: int,
    prompt: str | None = None,
) -> BenchmarkResult:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is not available")

    device = torch.device(f"cuda:{device_index}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = make_prompt_batch(tokenizer, input_len, batch_size, prompt)
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    prompt_tokens = int(encoded.input_ids.shape[1])

    model = load_causal_lm(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    model.eval()

    with torch.inference_mode():
        warmup = model(**encoded, use_cache=True)
        next_token = torch.argmax(warmup.logits[:, -1, :], dim=-1, keepdim=True)
        _ = model(input_ids=next_token, past_key_values=warmup.past_key_values, use_cache=True)
        torch.cuda.synchronize(device)

        out = model(**encoded, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]
        start = time.perf_counter()
        for _ in range(output_len):
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
        torch.cuda.synchronize(device)
        elapsed_s = time.perf_counter() - start

    total_tokens = output_len * batch_size
    return BenchmarkResult(
        name="hf_ar",
        model=model_id,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=total_tokens,
        elapsed_s=elapsed_s,
        tokens_per_second=tokens_per_second(total_tokens, elapsed_s),
        metadata={"device": str(device)},
    )
