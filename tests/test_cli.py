from jax_speculative_decoding.cli import build_parser


def test_env_command_parses():
    args = build_parser().parse_args(["env"])
    assert args.command == "env"


def test_speculative_defaults_parse():
    args = build_parser().parse_args(["speculative", "--k", "3"])
    assert args.k == 3
    assert args.target_device == 1
    assert args.draft_device == 0
