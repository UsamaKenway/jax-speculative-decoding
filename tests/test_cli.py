from jax_speculative_decoding.cli import build_parser


def test_env_command_parses():
    args = build_parser().parse_args(["env"])
    assert args.command == "env"


def test_speculative_defaults_parse():
    args = build_parser().parse_args(["speculative", "--k", "3"])
    assert args.k == 3
    assert args.target_device == 1
    assert args.draft_device == 0


def test_consistency_check_parses():
    args = build_parser().parse_args(
        ["consistency-check", "--k", "5", "--num-samples", "8", "--strict-throughput"]
    )
    assert args.command == "consistency-check"
    assert args.k == 5
    assert args.num_samples == 8
    assert args.strict_throughput is True


def test_scaling_speedup_plot_parses():
    args = build_parser().parse_args(["scaling", "--speedup-plot", "results/speedup.png"])
    assert args.speedup_plot == "results/speedup.png"
