from jax_speculative_decoding.results import BenchmarkResult


def test_result_to_dict_contains_core_metrics():
    result = BenchmarkResult(
        name="unit",
        model="target",
        output_tokens=10,
        elapsed_s=2.0,
        tokens_per_second=5.0,
    )
    payload = result.to_dict()
    assert payload["name"] == "unit"
    assert payload["tokens_per_second"] == 5.0
