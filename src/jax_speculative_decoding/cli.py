from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from .results import print_result, write_result
from .scaling import (
    maybe_plot_scaling,
    maybe_plot_speedup,
    parse_csv_ints,
    parse_csv_strings,
    run_scaling_sweep,
)


DEFAULT_TARGET = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"


def cmd_env(_: argparse.Namespace) -> int:
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    try:
        import jax

        print(f"JAX: {jax.__version__}")
        print(f"JAX backend: {jax.default_backend()}")
        print("JAX devices:")
        for device in jax.devices():
            print(f"  - {device}")
    except Exception as exc:
        print(f"JAX unavailable: {exc}")

    try:
        import torch

        print(f"Torch: {torch.__version__}")
        print(f"Torch CUDA available: {torch.cuda.is_available()}")
        print(f"Torch CUDA devices: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / (1024**3)
            print(f"  - cuda:{idx} {props.name} ({total_gb:.1f} GB)")
    except Exception as exc:
        print(f"Torch unavailable: {exc}")
    return 0


def cmd_vllm(args: argparse.Namespace) -> int:
    from .vllm_runner import run_vllm_baseline

    result = run_vllm_baseline(
        model_id=args.model,
        cuda_visible_devices=args.cuda_visible_devices,
        input_len=args.input_len,
        output_len=args.output_len,
        num_prompts=args.num_prompts,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prompt=args.prompt,
    )
    print_result(result)
    write_result(args.out, result)
    return 0


def cmd_hf(args: argparse.Namespace) -> int:
    from .hf_runner import run_hf_baseline

    result = run_hf_baseline(
        model_id=args.model,
        device_index=args.device,
        input_len=args.input_len,
        output_len=args.output_len,
        batch_size=args.batch_size,
        prompt=args.prompt,
    )
    print_result(result)
    write_result(args.out, result)
    return 0


def cmd_jax_ar(args: argparse.Namespace) -> int:
    from .jax_runner import run_jax_autoregressive_benchmark

    result = run_jax_autoregressive_benchmark(
        model_id=args.model,
        device_index=args.device,
        input_len=args.input_len,
        output_len=args.output_len,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        num_samples=args.num_samples,
    )
    print_result(result)
    write_result(args.out, result)
    return 0


def cmd_speculative(args: argparse.Namespace) -> int:
    from .spec_runner import run_speculative_benchmark

    result = run_speculative_benchmark(
        target_model_id=args.target_model,
        draft_model_id=args.draft_model,
        target_device_index=args.target_device,
        draft_device_index=args.draft_device,
        k=args.k,
        input_len=args.input_len,
        output_len=args.output_len,
        max_model_len=args.max_model_len,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        num_samples=args.num_samples,
    )
    print_result(result)
    write_result(args.out, result)
    return 0


def cmd_scaling(args: argparse.Namespace) -> int:
    results = run_scaling_sweep(
        target_model_id=args.target_model,
        draft_model_ids=parse_csv_strings(args.draft_models),
        ks=parse_csv_ints(args.ks),
        target_device_index=args.target_device,
        draft_device_index=args.draft_device,
        input_len=args.input_len,
        output_len=args.output_len,
        max_model_len=args.max_model_len,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        num_samples=args.num_samples,
    )
    for result in results:
        print_result(result)
    write_result(args.out, results)
    maybe_plot_scaling(args.plot, results, ar_baseline_tokens_per_second=args.ar_baseline_tok_s)
    maybe_plot_speedup(args.speedup_plot, results, ar_baseline_tokens_per_second=args.ar_baseline_tok_s)
    return 0


def cmd_consistency(args: argparse.Namespace) -> int:
    from .scaling import run_scaling_sweep
    from .spec_runner import run_speculative_benchmark

    standalone = run_speculative_benchmark(
        target_model_id=args.target_model,
        draft_model_id=args.draft_model,
        target_device_index=args.target_device,
        draft_device_index=args.draft_device,
        k=args.k,
        input_len=args.input_len,
        output_len=args.output_len,
        max_model_len=args.max_model_len,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        num_samples=args.num_samples,
    )
    scaling = run_scaling_sweep(
        target_model_id=args.target_model,
        draft_model_ids=[args.draft_model],
        ks=[args.k],
        target_device_index=args.target_device,
        draft_device_index=args.draft_device,
        input_len=args.input_len,
        output_len=args.output_len,
        max_model_len=args.max_model_len,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        num_samples=args.num_samples,
    )[0]
    acceptance_delta = abs((standalone.acceptance_rate or 0.0) - (scaling.acceptance_rate or 0.0))
    throughput_delta = abs(standalone.tokens_per_second - scaling.tokens_per_second)
    throughput_rel_delta = throughput_delta / max(standalone.tokens_per_second, 1e-9)
    matching_counts = (
        standalone.output_tokens == scaling.output_tokens
        and standalone.accepted_draft_tokens == scaling.accepted_draft_tokens
        and standalone.proposed_tokens == scaling.proposed_tokens
    )
    acceptance_passed = acceptance_delta <= args.acceptance_tolerance
    throughput_passed = throughput_rel_delta <= args.throughput_tolerance
    data_integrity_passed = acceptance_passed and matching_counts
    passed = data_integrity_passed and (throughput_passed or not args.strict_throughput)
    payload = {
        "passed": passed,
        "data_integrity_passed": data_integrity_passed,
        "acceptance_passed": acceptance_passed,
        "matching_counts": matching_counts,
        "throughput_passed": throughput_passed,
        "throughput_warning": data_integrity_passed and not throughput_passed,
        "strict_throughput": args.strict_throughput,
        "acceptance_delta": acceptance_delta,
        "throughput_delta": throughput_delta,
        "throughput_relative_delta": throughput_rel_delta,
        "acceptance_tolerance": args.acceptance_tolerance,
        "throughput_tolerance": args.throughput_tolerance,
        "standalone": standalone.to_dict(),
        "scaling": scaling.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if passed else 1


def add_common_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--out", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen2.5 inference benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="Print available JAX and Torch devices")
    env.set_defaults(func=cmd_env)

    vllm = sub.add_parser("vllm-baseline", help="Run a vLLM autoregressive throughput baseline")
    vllm.add_argument("--model", default=DEFAULT_TARGET)
    vllm.add_argument("--cuda-visible-devices", default="1")
    vllm.add_argument("--num-prompts", type=int, default=50)
    vllm.add_argument("--dtype", default="bfloat16")
    vllm.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    add_common_generation_args(vllm)
    vllm.set_defaults(func=cmd_vllm)

    hf = sub.add_parser("hf-baseline", help="Run a PyTorch/HuggingFace greedy AR baseline")
    hf.add_argument("--model", default=DEFAULT_TARGET)
    hf.add_argument("--device", type=int, default=1)
    hf.add_argument("--batch-size", type=int, default=1)
    add_common_generation_args(hf)
    hf.set_defaults(func=cmd_hf)

    jax_ar = sub.add_parser("jax-ar", help="Run the JAX greedy AR benchmark")
    jax_ar.add_argument("--model", default=DEFAULT_TARGET)
    jax_ar.add_argument("--device", type=int, default=1)
    jax_ar.add_argument("--batch-size", type=int, default=1)
    jax_ar.add_argument("--prompt-file", default=None)
    jax_ar.add_argument("--num-samples", type=int, default=1)
    add_common_generation_args(jax_ar)
    jax_ar.set_defaults(func=cmd_jax_ar)

    spec = sub.add_parser("speculative", help="Run greedy speculative decoding across two devices")
    spec.add_argument("--target-model", default=DEFAULT_TARGET)
    spec.add_argument("--draft-model", default=DEFAULT_DRAFT)
    spec.add_argument("--target-device", type=int, default=1)
    spec.add_argument("--draft-device", type=int, default=0)
    spec.add_argument("--k", type=int, default=5)
    spec.add_argument("--prompt-file", default=None)
    spec.add_argument("--num-samples", type=int, default=1)
    add_common_generation_args(spec)
    spec.set_defaults(func=cmd_speculative)

    scaling = sub.add_parser("scaling", help="Sweep K and draft model size")
    scaling.add_argument("--target-model", default=DEFAULT_TARGET)
    scaling.add_argument("--draft-models", default="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct")
    scaling.add_argument("--ks", default="1,3,5,8,10")
    scaling.add_argument("--target-device", type=int, default=1)
    scaling.add_argument("--draft-device", type=int, default=0)
    scaling.add_argument("--plot", default=None)
    scaling.add_argument("--speedup-plot", default=None)
    scaling.add_argument("--ar-baseline-tok-s", type=float, default=None)
    scaling.add_argument("--prompt-file", default=None)
    scaling.add_argument("--num-samples", type=int, default=1)
    add_common_generation_args(scaling)
    scaling.set_defaults(func=cmd_scaling)

    consistency = sub.add_parser("consistency-check", help="Compare standalone speculative against single-point scaling")
    consistency.add_argument("--target-model", default=DEFAULT_TARGET)
    consistency.add_argument("--draft-model", default=DEFAULT_DRAFT)
    consistency.add_argument("--target-device", type=int, default=1)
    consistency.add_argument("--draft-device", type=int, default=0)
    consistency.add_argument("--k", type=int, default=5)
    consistency.add_argument("--prompt-file", default=None)
    consistency.add_argument("--num-samples", type=int, default=4)
    consistency.add_argument("--acceptance-tolerance", type=float, default=0.0)
    consistency.add_argument("--throughput-tolerance", type=float, default=0.05)
    consistency.add_argument("--strict-throughput", action="store_true")
    add_common_generation_args(consistency)
    consistency.set_defaults(func=cmd_consistency)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
