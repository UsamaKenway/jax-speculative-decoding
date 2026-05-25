from __future__ import annotations

import time

from .hf_loader import load_qwen2_jax_params, load_tokenizer, select_jax_device
from .jax_qwen import init_kv_cache, make_forward, make_greedy_step
from .prompts import make_prompt_batch, make_prompt_samples
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
    prompt_file: str | None = None,
    num_samples: int = 1,
) -> BenchmarkResult:
    import jax
    import jax.numpy as jnp

    device = select_jax_device(device_index)
    tokenizer = load_tokenizer(model_id)
    if num_samples == 1:
        prompt_batches = [make_prompt_batch(tokenizer, input_len, batch_size, prompt)]
    else:
        samples = make_prompt_samples(
            tokenizer,
            input_len,
            num_samples,
            text=prompt,
            prompt_file=prompt_file,
        )
        prompt_batches = [[sample] * batch_size for sample in samples]
    first_input_ids = _encode_batch(tokenizer, prompt_batches[0], device)
    prompt_tokens = int(first_input_ids.shape[1])

    config, params = load_qwen2_jax_params(model_id, device=device, dtype=jnp.bfloat16)
    forward = make_forward(config)
    step = make_greedy_step(config)

    compile_start = time.perf_counter()
    cache = init_kv_cache(config, batch_size, max_model_len, dtype=jnp.bfloat16, device=device)
    logits, cache = forward(params, first_input_ids, cache, jnp.asarray(0, dtype=jnp.int32))
    logits, cache = block_until_ready((logits, cache))
    cache_index = jnp.asarray(prompt_tokens, dtype=jnp.int32)
    token, next_logits, cache = step(params, logits[:, -1, :], cache, cache_index)
    token, next_logits, cache = block_until_ready((token, next_logits, cache))
    compile_s = time.perf_counter() - compile_start

    total_elapsed_s = 0.0
    total_tokens = 0
    sample_stats = []
    for sample_idx, prompts in enumerate(prompt_batches):
        input_ids = _encode_batch(tokenizer, prompts, device)
        sample_prompt_tokens = int(input_ids.shape[1])
        cache = init_kv_cache(config, batch_size, max_model_len, dtype=jnp.bfloat16, device=device)
        logits, cache = forward(params, input_ids, cache, jnp.asarray(0, dtype=jnp.int32))
        logits, cache = block_until_ready((logits, cache))
        next_logits = logits[:, -1, :]
        cache_index = jnp.asarray(sample_prompt_tokens, dtype=jnp.int32)

        start = time.perf_counter()
        for _ in range(output_len):
            token, next_logits, cache = step(params, next_logits, cache, cache_index)
            cache_index = cache_index + 1
        block_until_ready((token, next_logits, cache))
        sample_elapsed_s = time.perf_counter() - start
        sample_tokens = output_len * batch_size
        total_elapsed_s += sample_elapsed_s
        total_tokens += sample_tokens
        sample_stats.append(
            {
                "sample_index": sample_idx,
                "prompt_tokens": sample_prompt_tokens,
                "output_tokens": sample_tokens,
                "elapsed_s": sample_elapsed_s,
                "tokens_per_second": tokens_per_second(sample_tokens, sample_elapsed_s),
                "prompt_preview": prompts[0][:120],
            }
        )

    return BenchmarkResult(
        name="jax_ar",
        model=model_id,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=total_tokens,
        elapsed_s=total_elapsed_s,
        tokens_per_second=tokens_per_second(total_tokens, total_elapsed_s),
        compile_s=compile_s,
        metadata={
            "device": str(device),
            "max_model_len": max_model_len,
            "jax_backend": jax.default_backend(),
            "num_samples": num_samples,
            "prompt_file": prompt_file,
            "sample_stats": sample_stats,
        },
    )
