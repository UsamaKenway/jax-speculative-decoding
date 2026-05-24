from __future__ import annotations

import time

from .hf_loader import load_qwen2_jax_params, load_tokenizer, select_jax_device
from .jax_qwen import init_kv_cache, make_forward
from .prompts import make_prompt
from .results import BenchmarkResult
from .spec_decode import make_draft_k, make_verify_greedy
from .timing import block_until_ready, tokens_per_second


def run_speculative_benchmark(
    *,
    target_model_id: str,
    draft_model_id: str,
    target_device_index: int,
    draft_device_index: int,
    k: int,
    input_len: int,
    output_len: int,
    max_model_len: int,
    prompt: str | None = None,
) -> BenchmarkResult:
    import jax
    import jax.numpy as jnp

    if k <= 0:
        raise ValueError("k must be positive")

    target_device = select_jax_device(target_device_index)
    draft_device = select_jax_device(draft_device_index)
    tokenizer = load_tokenizer(target_model_id)
    prompt_text = make_prompt(tokenizer, input_len, prompt)
    encoded = tokenizer(prompt_text, return_tensors="np", add_special_tokens=False)
    target_input_ids = jax.device_put(jnp.asarray(encoded.input_ids, dtype=jnp.int32), target_device)
    draft_input_ids = jax.device_put(jnp.asarray(encoded.input_ids, dtype=jnp.int32), draft_device)
    prompt_tokens = int(target_input_ids.shape[1])
    if prompt_tokens + output_len + k + 1 > max_model_len:
        raise ValueError(
            "max_model_len is too small for prompt + output + speculative lookahead: "
            f"{prompt_tokens} + {output_len} + {k} + 1 > {max_model_len}"
        )

    target_config, target_params = load_qwen2_jax_params(
        target_model_id, device=target_device, dtype=jnp.bfloat16
    )
    draft_config, draft_params = load_qwen2_jax_params(
        draft_model_id, device=draft_device, dtype=jnp.bfloat16
    )

    target_forward = make_forward(target_config)
    draft_forward = make_forward(draft_config)
    draft_k = make_draft_k(draft_config, k)
    verify = make_verify_greedy(target_config, k)

    def prefill():
        target_cache = init_kv_cache(target_config, 1, max_model_len, dtype=jnp.bfloat16, device=target_device)
        draft_cache = init_kv_cache(draft_config, 1, max_model_len, dtype=jnp.bfloat16, device=draft_device)
        target_logits, target_cache = target_forward(
            target_params, target_input_ids, target_cache, jnp.asarray(0, dtype=jnp.int32)
        )
        draft_logits, draft_cache = draft_forward(
            draft_params, draft_input_ids, draft_cache, jnp.asarray(0, dtype=jnp.int32)
        )
        target_logits, target_cache, draft_logits, draft_cache = block_until_ready(
            (target_logits, target_cache, draft_logits, draft_cache)
        )
        return target_logits[:, -1, :], target_cache, draft_logits[:, -1, :], draft_cache

    compile_start = time.perf_counter()
    target_next_logits, target_cache, draft_next_logits, draft_cache = prefill()
    base_index = jnp.asarray(prompt_tokens, dtype=jnp.int32)
    draft_tokens, _, draft_next_logits_after_k, draft_cache_after_k = draft_k(
        draft_params, draft_next_logits, draft_cache, base_index
    )
    draft_tokens_on_target = jax.device_put(draft_tokens, target_device)
    accepted_count, target_tokens, target_next_logits_after_k, target_cache_after_k, accept = verify(
        target_params, target_next_logits, target_cache, base_index, draft_tokens_on_target
    )
    block_until_ready(
        (
            accepted_count,
            target_tokens,
            target_next_logits_after_k,
            target_cache_after_k,
            draft_next_logits_after_k,
            draft_cache_after_k,
            accept,
        )
    )
    compile_s = time.perf_counter() - compile_start

    target_next_logits, target_cache, draft_next_logits, draft_cache = prefill()
    cache_index = prompt_tokens
    emitted = 0
    proposed = 0
    accepted_total = 0
    pcie_transfer_s = 0.0

    start = time.perf_counter()
    last_token = None
    while emitted < output_len:
        index_value = jnp.asarray(cache_index, dtype=jnp.int32)
        draft_tokens, _, draft_next_logits_after_k, draft_cache_after_k = draft_k(
            draft_params, draft_next_logits, draft_cache, index_value
        )

        transfer_start = time.perf_counter()
        draft_tokens_on_target = jax.device_put(draft_tokens, target_device)
        block_until_ready(draft_tokens_on_target)
        pcie_transfer_s += time.perf_counter() - transfer_start

        accepted_count, target_tokens, target_next_logits_after_k, target_cache_after_k, accept = verify(
            target_params, target_next_logits, target_cache, index_value, draft_tokens_on_target
        )
        accepted_count = int(jax.device_get(accepted_count))
        proposed += k
        accepted_total += accepted_count

        if accepted_count == k:
            target_cache = target_cache_after_k
            draft_cache = draft_cache_after_k
            target_next_logits = target_next_logits_after_k
            draft_next_logits = draft_next_logits_after_k
            cache_index += k
            emitted += min(k, output_len - emitted)
            last_token = draft_tokens[:, -1]
            continue

        replacement = target_tokens[:, accepted_count : accepted_count + 1]
        replacement_on_draft = jax.device_put(replacement, draft_device)

        target_logits, target_cache = target_forward(
            target_params,
            replacement,
            target_cache_after_k,
            jnp.asarray(cache_index + accepted_count, dtype=jnp.int32),
        )
        draft_logits, draft_cache = draft_forward(
            draft_params,
            replacement_on_draft,
            draft_cache_after_k,
            jnp.asarray(cache_index + accepted_count, dtype=jnp.int32),
        )
        target_logits, target_cache, draft_logits, draft_cache = block_until_ready(
            (target_logits, target_cache, draft_logits, draft_cache)
        )
        target_next_logits = target_logits[:, -1, :]
        draft_next_logits = draft_logits[:, -1, :]
        cache_index += accepted_count + 1
        emitted += min(accepted_count + 1, output_len - emitted)
        last_token = replacement[:, 0]

    block_until_ready((last_token, target_next_logits, draft_next_logits, target_cache, draft_cache))
    elapsed_s = time.perf_counter() - start

    return BenchmarkResult(
        name="jax_speculative_greedy",
        model=target_model_id,
        draft_model=draft_model_id,
        k=k,
        batch_size=1,
        prompt_tokens=prompt_tokens,
        output_tokens=emitted,
        elapsed_s=elapsed_s,
        tokens_per_second=tokens_per_second(emitted, elapsed_s),
        compile_s=compile_s,
        acceptance_rate=(accepted_total / proposed) if proposed else 0.0,
        proposed_tokens=proposed,
        accepted_draft_tokens=accepted_total,
        pcie_transfer_s=pcie_transfer_s,
        metadata={
            "target_device": str(target_device),
            "draft_device": str(draft_device),
            "max_model_len": max_model_len,
            "acceptance": "greedy_prefix_match",
        },
    )
