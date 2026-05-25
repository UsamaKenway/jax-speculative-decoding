from __future__ import annotations

from pathlib import Path

from .hf_loader import load_qwen2_jax_params, load_tokenizer, select_jax_device
from .jax_qwen import make_forward
from .results import BenchmarkResult
from .spec_runner import LoadedSpeculativeModels, run_speculative_benchmark_loaded


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_scaling_sweep(
    *,
    target_model_id: str,
    draft_model_ids: list[str],
    ks: list[int],
    target_device_index: int,
    draft_device_index: int,
    input_len: int,
    output_len: int,
    max_model_len: int,
    prompt: str | None = None,
    prompt_file: str | None = None,
    num_samples: int = 1,
) -> list[BenchmarkResult]:
    import jax.numpy as jnp

    results: list[BenchmarkResult] = []
    target_device = select_jax_device(target_device_index)
    draft_device = select_jax_device(draft_device_index)
    tokenizer = load_tokenizer(target_model_id)
    target_config, target_params = load_qwen2_jax_params(
        target_model_id, device=target_device, dtype=jnp.bfloat16
    )
    target_forward = make_forward(target_config)

    for draft_model_id in draft_model_ids:
        draft_config, draft_params = load_qwen2_jax_params(
            draft_model_id, device=draft_device, dtype=jnp.bfloat16
        )
        models = LoadedSpeculativeModels(
            target_model_id=target_model_id,
            draft_model_id=draft_model_id,
            tokenizer=tokenizer,
            target_device=target_device,
            draft_device=draft_device,
            target_config=target_config,
            draft_config=draft_config,
            target_params=target_params,
            draft_params=draft_params,
            target_forward=target_forward,
            draft_forward=make_forward(draft_config),
        )
        for k in ks:
            result = run_speculative_benchmark_loaded(
                models,
                k=k,
                input_len=input_len,
                output_len=output_len,
                max_model_len=max_model_len,
                prompt=prompt,
                prompt_file=prompt_file,
                num_samples=num_samples,
            )
            result.metadata["scaling_target_loaded_once"] = True
            result.metadata["scaling_draft_reused_across_k"] = True
            results.append(result)
    return results


def maybe_plot_scaling(path: str | None, results: list[BenchmarkResult]) -> None:
    if path is None:
        return

    import matplotlib.pyplot as plt
    import pandas as pd

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([result.to_dict() for result in results])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for draft_model, group in df.groupby("draft_model"):
        group = group.sort_values("k")
        label = str(draft_model).split("/")[-1]
        axes[0].plot(group["k"], group["tokens_per_second"], marker="o", label=label)
        axes[1].plot(group["k"], group["acceptance_rate"], marker="o", label=label)

    axes[0].set_title("Speculative Throughput")
    axes[0].set_xlabel("K draft tokens")
    axes[0].set_ylabel("tokens/sec")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Greedy Acceptance Rate")
    axes[1].set_xlabel("K draft tokens")
    axes[1].set_ylabel("accepted / proposed")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out, dpi=160)
