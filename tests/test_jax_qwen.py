import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jax_speculative_decoding.jax_qwen import Qwen2JAXConfig, init_kv_cache, make_forward


def _tiny_config():
    return Qwen2JAXConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        max_position_embeddings=16,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )


def _tiny_params(dtype):
    return {
        "embed_tokens": jnp.ones((8, 4), dtype=dtype),
        "norm": jnp.ones((4,), dtype=dtype),
        "lm_head": jnp.ones((8, 4), dtype=dtype),
        "layers": (
            {
                "input_layernorm": jnp.ones((4,), dtype=dtype),
                "post_attention_layernorm": jnp.ones((4,), dtype=dtype),
                "self_attn": {
                    "q_proj": jnp.ones((4, 4), dtype=dtype),
                    "k_proj": jnp.ones((2, 4), dtype=dtype),
                    "v_proj": jnp.ones((2, 4), dtype=dtype),
                    "o_proj": jnp.ones((4, 4), dtype=dtype),
                },
                "mlp": {
                    "gate_proj": jnp.ones((8, 4), dtype=dtype),
                    "up_proj": jnp.ones((8, 4), dtype=dtype),
                    "down_proj": jnp.ones((4, 8), dtype=dtype),
                },
            },
        ),
    }


def test_tiny_qwen_forward_shapes():
    config = _tiny_config()
    params = _tiny_params(jnp.float32)
    cache = init_kv_cache(config, batch_size=1, max_seq_len=6, dtype=jnp.float32)
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)

    logits, new_cache = make_forward(config)(params, input_ids, cache, jnp.asarray(0, dtype=jnp.int32))

    assert logits.shape == (1, 3, 8)
    assert new_cache["key"][0].shape == (1, 6, 1, 2)


def test_tiny_qwen_forward_accepts_bfloat16_cache():
    config = _tiny_config()
    params = _tiny_params(jnp.bfloat16)
    cache = init_kv_cache(config, batch_size=1, max_seq_len=6, dtype=jnp.bfloat16)
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)

    logits, new_cache = make_forward(config)(params, input_ids, cache, jnp.asarray(0, dtype=jnp.int32))

    assert logits.shape == (1, 3, 8)
    assert new_cache["key"][0].dtype == jnp.bfloat16
