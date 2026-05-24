from __future__ import annotations

import time

from .hf_loader import load_qwen2_jax_params, load_tokenizer, select_jax_device
from .jax_qwen import init_kv_cache, make_forward, make_greedy_step
from .prompts import make_prompt_batch
from .results import BenchmarkResult
from .timing import block_until_ready, tokens_per_second


def _encode_batch(tokenizer, prompts: list[str], device):
    import jax
    import jax.numpy as jnp

    encoded = tokenizer(prompts, return_tensors="np", padding=True, add_special_tokens=False)
    return jax.device_put(jnp.asarray(encoded.input_ids, dtype=jnp.int32), device)


def run_jax_autoregressive_benchmark(
    *,
    model_id: str,
    device_index: int,
    input_len: int,
    output_len: int,
    batch_size: int,
    max_model_len: int,
    prompt: str | None = None,
) -> BenchmarkResult:
    import jax
    import jax.numpy as jnp

    device = select_jax_device(device_index)
    tokenizer = load_tokenizer(model_id)
    prompts = make_prompt_batch(tokenizer, input_len, batch_size, prompt)
    input_ids = _encode_batch(tokenizer, prompts, device)
    prompt_tokens = int(input_ids.shape[1])

    config, params = load_qwen2_jax_params(model_id, device=device, dtype=jnp.bfloat16)
    cache = init_kv_cache(config, batch_size, max_model_len, dtype=jnp.bfloat16, device=device)
    forward = make_forward(config)
    step = make_greedy_step(config)

    compile_start = time.perf_counter()
    logits, cache = forward(params, input_ids, cache, jnp.asarray(0, dtype=jnp.int32))
    logits, cache = block_until_ready((logits, cache))
    cache_index = jnp.asarray(prompt_tokens, dtype=jnp.int32)
    token, next_logits, cache = step(params, logits[:, -1, :], cache, cache_index)
    token, next_logits, cache = block_until_ready((token, next_logits, cache))
    compile_s = time.perf_counter() - compile_start

    cache = init_kv_cache(config, batch_size, max_model_len, dtype=jnp.bfloat16, device=device)
    logits, cache = forward(params, input_ids, cache, jnp.asarray(0, dtype=jnp.int32))
    logits, cache = block_until_ready((logits, cache))
    next_logits = logits[:, -1, :]
    cache_index = jnp.asarray(prompt_tokens, dtype=jnp.int32)

    start = time.perf_counter()
    for _ in range(output_len):
        token, next_logits, cache = step(params, next_logits, cache, cache_index)
        cache_index = cache_index + 1
    block_until_ready((token, next_logits, cache))
    elapsed_s = time.perf_counter() - start
    total_tokens = output_len * batch_size

    return BenchmarkResult(
        name="jax_ar",
        model=model_id,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=total_tokens,
        elapsed_s=elapsed_s,
        tokens_per_second=tokens_per_second(total_tokens, elapsed_s),
        compile_s=compile_s,
        metadata={
            "device": str(device),
            "max_model_len": max_model_len,
            "jax_backend": jax.default_backend(),
        },
    )
