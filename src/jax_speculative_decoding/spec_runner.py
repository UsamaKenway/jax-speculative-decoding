from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .hf_loader import load_qwen2_jax_params, load_tokenizer, select_jax_device
from .jax_qwen import init_kv_cache, make_forward
from .prompts import make_prompt_samples
from .results import BenchmarkResult
from .spec_decode import make_draft_k, make_verify_greedy
from .timing import block_until_ready, tokens_per_second


@dataclass
class LoadedSpeculativeModels:
    target_model_id: str
    draft_model_id: str
    tokenizer: Any
    target_device: Any
    draft_device: Any
    target_config: Any
    draft_config: Any
    target_params: Any
    draft_params: Any
    target_forward: Any
    draft_forward: Any


def load_speculative_models(
    *,
    target_model_id: str,
    draft_model_id: str,
    target_device_index: int,
    draft_device_index: int,
) -> LoadedSpeculativeModels:
    import jax.numpy as jnp

    target_device = select_jax_device(target_device_index)
    draft_device = select_jax_device(draft_device_index)
    tokenizer = load_tokenizer(target_model_id)
    target_config, target_params = load_qwen2_jax_params(
        target_model_id, device=target_device, dtype=jnp.bfloat16
    )
    draft_config, draft_params = load_qwen2_jax_params(
        draft_model_id, device=draft_device, dtype=jnp.bfloat16
    )
    return LoadedSpeculativeModels(
        target_model_id=target_model_id,
        draft_model_id=draft_model_id,
        tokenizer=tokenizer,
        target_device=target_device,
        draft_device=draft_device,
        target_config=target_config,
        draft_config=draft_config,
        target_params=target_params,
        draft_params=draft_params,
        target_forward=make_forward(target_config),
        draft_forward=make_forward(draft_config),
    )


def _encode_prompt(models: LoadedSpeculativeModels, prompt_text: str):
    import jax
    import jax.numpy as jnp

    encoded = models.tokenizer(prompt_text, return_tensors="np", add_special_tokens=False)
    input_ids = jnp.asarray(encoded.input_ids, dtype=jnp.int32)
    return (
        jax.device_put(input_ids, models.target_device),
        jax.device_put(input_ids, models.draft_device),
        int(input_ids.shape[1]),
    )


def run_speculative_benchmark_loaded(
    models: LoadedSpeculativeModels,
    *,
    k: int,
    input_len: int,
    output_len: int,
    max_model_len: int,
    prompt: str | None = None,
    prompt_file: str | None = None,
    num_samples: int = 1,
) -> BenchmarkResult:
    import jax
    import jax.numpy as jnp

    if k <= 0:
        raise ValueError("k must be positive")

    prompt_texts = make_prompt_samples(
        models.tokenizer,
        input_len,
        num_samples,
        text=prompt,
        prompt_file=prompt_file,
    )
    draft_k = make_draft_k(models.draft_config, k)
    verify = make_verify_greedy(models.target_config, k)

    def prefill(target_input_ids, draft_input_ids):
        target_cache = init_kv_cache(
            models.target_config, 1, max_model_len, dtype=jnp.bfloat16, device=models.target_device
        )
        draft_cache = init_kv_cache(
            models.draft_config, 1, max_model_len, dtype=jnp.bfloat16, device=models.draft_device
        )
        target_logits, target_cache = models.target_forward(
            models.target_params, target_input_ids, target_cache, jnp.asarray(0, dtype=jnp.int32)
        )
        draft_logits, draft_cache = models.draft_forward(
            models.draft_params, draft_input_ids, draft_cache, jnp.asarray(0, dtype=jnp.int32)
        )
        target_logits, target_cache, draft_logits, draft_cache = block_until_ready(
            (target_logits, target_cache, draft_logits, draft_cache)
        )
        return target_logits[:, -1, :], target_cache, draft_logits[:, -1, :], draft_cache

    compile_s = 0.0
    total_elapsed_s = 0.0
    total_emitted = 0
    total_proposed = 0
    total_accepted = 0
    total_pcie_transfer_s = 0.0
    first_prompt_tokens = 0

    for sample_idx, prompt_text in enumerate(prompt_texts):
        target_input_ids, draft_input_ids, prompt_tokens = _encode_prompt(models, prompt_text)
        if sample_idx == 0:
            first_prompt_tokens = prompt_tokens
        if prompt_tokens + output_len + k + 1 > max_model_len:
            raise ValueError(
                "max_model_len is too small for prompt + output + speculative lookahead: "
                f"{prompt_tokens} + {output_len} + {k} + 1 > {max_model_len}"
            )

        if sample_idx == 0:
            compile_start = time.perf_counter()
            target_next_logits, target_cache, draft_next_logits, draft_cache = prefill(
                target_input_ids, draft_input_ids
            )
            base_index = jnp.asarray(prompt_tokens, dtype=jnp.int32)
            draft_tokens, _, draft_next_logits_after_k, draft_cache_after_k = draft_k(
                models.draft_params, draft_next_logits, draft_cache, base_index
            )
            draft_tokens_on_target = jax.device_put(draft_tokens, models.target_device)
            accepted_count, target_tokens, target_next_logits_after_k, target_cache_after_k, accept = verify(
                models.target_params, target_next_logits, target_cache, base_index, draft_tokens_on_target
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

        target_next_logits, target_cache, draft_next_logits, draft_cache = prefill(
            target_input_ids, draft_input_ids
        )
        cache_index = prompt_tokens
        emitted = 0
        pcie_transfer_s = 0.0

        start = time.perf_counter()
        last_token = None
        while emitted < output_len:
            remaining = output_len - emitted
            proposed_this_step = min(k, remaining)
            index_value = jnp.asarray(cache_index, dtype=jnp.int32)
            draft_tokens, _, draft_next_logits_after_k, draft_cache_after_k = draft_k(
                models.draft_params, draft_next_logits, draft_cache, index_value
            )

            transfer_start = time.perf_counter()
            draft_tokens_on_target = jax.device_put(draft_tokens, models.target_device)
            block_until_ready(draft_tokens_on_target)
            pcie_transfer_s += time.perf_counter() - transfer_start

            accepted_count, target_tokens, target_next_logits_after_k, target_cache_after_k, _ = verify(
                models.target_params, target_next_logits, target_cache, index_value, draft_tokens_on_target
            )
            accepted_count = int(jax.device_get(accepted_count))
            total_proposed += proposed_this_step
            total_accepted += min(accepted_count, proposed_this_step)

            if accepted_count == k:
                target_cache = target_cache_after_k
                draft_cache = draft_cache_after_k
                target_next_logits = target_next_logits_after_k
                draft_next_logits = draft_next_logits_after_k
                cache_index += k
                emitted += proposed_this_step
                last_token = draft_tokens[:, proposed_this_step - 1]
                continue

            replacement = target_tokens[:, accepted_count : accepted_count + 1]
            replacement_on_draft = jax.device_put(replacement, models.draft_device)

            target_logits, target_cache = models.target_forward(
                models.target_params,
                replacement,
                target_cache_after_k,
                jnp.asarray(cache_index + accepted_count, dtype=jnp.int32),
            )
            draft_logits, draft_cache = models.draft_forward(
                models.draft_params,
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
            emitted += min(accepted_count + 1, remaining)
            last_token = replacement[:, 0]

        block_until_ready((last_token, target_next_logits, draft_next_logits, target_cache, draft_cache))
        total_elapsed_s += time.perf_counter() - start
        total_emitted += emitted
        total_pcie_transfer_s += pcie_transfer_s

    return BenchmarkResult(
        name="jax_speculative_greedy",
        model=models.target_model_id,
        draft_model=models.draft_model_id,
        k=k,
        batch_size=1,
        prompt_tokens=first_prompt_tokens,
        output_tokens=total_emitted,
        elapsed_s=total_elapsed_s,
        tokens_per_second=tokens_per_second(total_emitted, total_elapsed_s),
        compile_s=compile_s,
        acceptance_rate=(total_accepted / total_proposed) if total_proposed else 0.0,
        proposed_tokens=total_proposed,
        accepted_draft_tokens=total_accepted,
        pcie_transfer_s=total_pcie_transfer_s,
        metadata={
            "target_device": str(models.target_device),
            "draft_device": str(models.draft_device),
            "max_model_len": max_model_len,
            "acceptance": "greedy_prefix_match",
            "num_samples": num_samples,
            "prompt_file": prompt_file,
            "target_loaded_once": True,
            "draft_loaded_once": True,
        },
    )


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
    prompt_file: str | None = None,
    num_samples: int = 1,
) -> BenchmarkResult:
    models = load_speculative_models(
        target_model_id=target_model_id,
        draft_model_id=draft_model_id,
        target_device_index=target_device_index,
        draft_device_index=draft_device_index,
    )
    return run_speculative_benchmark_loaded(
        models,
        k=k,
        input_len=input_len,
        output_len=output_len,
        max_model_len=max_model_len,
        prompt=prompt,
        prompt_file=prompt_file,
        num_samples=num_samples,
    )
