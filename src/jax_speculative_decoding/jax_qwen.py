from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any


@dataclass(frozen=True)
class Qwen2JAXConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf_config(cls, config: Any) -> "Qwen2JAXConfig":
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return cls(
            vocab_size=int(config.vocab_size),
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.intermediate_size),
            num_hidden_layers=int(config.num_hidden_layers),
            num_attention_heads=int(config.num_attention_heads),
            num_key_value_heads=int(config.num_key_value_heads),
            head_dim=int(head_dim),
            max_position_embeddings=int(config.max_position_embeddings),
            rope_theta=float(getattr(config, "rope_theta", 1000000.0)),
            rms_norm_eps=float(config.rms_norm_eps),
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


def _jax():
    import jax

    return jax


def _jnp():
    import jax.numpy as jnp

    return jnp


def init_kv_cache(
    config: Qwen2JAXConfig,
    batch_size: int,
    max_seq_len: int,
    *,
    dtype: Any | None = None,
    device: Any | None = None,
) -> dict[str, tuple[Any, ...]]:
    jax = _jax()
    jnp = _jnp()
    dtype = dtype or jnp.bfloat16
    shape = (batch_size, max_seq_len, config.num_key_value_heads, config.head_dim)
    keys = tuple(jnp.zeros(shape, dtype=dtype) for _ in range(config.num_hidden_layers))
    values = tuple(jnp.zeros(shape, dtype=dtype) for _ in range(config.num_hidden_layers))
    cache = {"key": keys, "value": values}
    return jax.device_put(cache, device) if device is not None else cache


def _linear(x, weight, bias=None):
    jnp = _jnp()
    y = jnp.einsum("...d,od->...o", x, weight)
    if bias is not None:
        y = y + bias
    return y


def _rms_norm(x, weight, eps: float):
    jax = _jax()
    jnp = _jnp()
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(variance + eps)
    return (x * weight).astype(weight.dtype)


def _rotate_half(x):
    jnp = _jnp()
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def _apply_rope(q, k, position_ids, config: Qwen2JAXConfig):
    jnp = _jnp()
    q_dtype = q.dtype
    k_dtype = k.dtype
    inv_freq = 1.0 / (
        config.rope_theta
        ** (jnp.arange(0, config.head_dim, 2, dtype=jnp.float32) / config.head_dim)
    )
    freqs = jnp.einsum("bt,d->btd", position_ids.astype(jnp.float32), inv_freq)
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    cos = jnp.cos(emb)[:, :, None, :]
    sin = jnp.sin(emb)[:, :, None, :]
    q_rot = ((q * cos) + (_rotate_half(q) * sin)).astype(q_dtype)
    k_rot = ((k * cos) + (_rotate_half(k) * sin)).astype(k_dtype)
    return q_rot, k_rot


def _repeat_kv(x, repeats: int):
    if repeats == 1:
        return x
    jnp = _jnp()
    return jnp.repeat(x, repeats=repeats, axis=2)


def _attention_mask(seq_len: int, max_seq_len: int, cache_index):
    jnp = _jnp()
    query_positions = cache_index + jnp.arange(seq_len)
    key_positions = jnp.arange(max_seq_len)
    valid = (key_positions[None, :] < cache_index + seq_len) & (
        key_positions[None, :] <= query_positions[:, None]
    )
    return valid[None, None, :, :]


def _optional(params: dict[str, Any], key: str):
    return params[key] if key in params else None


def qwen2_forward(
    params: dict[str, Any],
    config: Qwen2JAXConfig,
    input_ids,
    cache: dict[str, tuple[Any, ...]],
    cache_index,
):
    jax = _jax()
    jnp = _jnp()

    batch_size, seq_len = input_ids.shape
    hidden = jnp.take(params["embed_tokens"], input_ids, axis=0)
    position_ids = cache_index + jnp.arange(seq_len)
    position_ids = jnp.broadcast_to(position_ids[None, :], (batch_size, seq_len))
    new_keys = list(cache["key"])
    new_values = list(cache["value"])
    kv_repeats = config.num_attention_heads // config.num_key_value_heads

    for layer_idx, layer in enumerate(params["layers"]):
        residual = hidden
        x = _rms_norm(hidden, layer["input_layernorm"], config.rms_norm_eps)

        attn = layer["self_attn"]
        q = _linear(x, attn["q_proj"], _optional(attn, "q_proj_bias"))
        k = _linear(x, attn["k_proj"], _optional(attn, "k_proj_bias"))
        v = _linear(x, attn["v_proj"], _optional(attn, "v_proj_bias"))
        q = q.reshape(batch_size, seq_len, config.num_attention_heads, config.head_dim)
        k = k.reshape(batch_size, seq_len, config.num_key_value_heads, config.head_dim)
        v = v.reshape(batch_size, seq_len, config.num_key_value_heads, config.head_dim)
        q, k = _apply_rope(q, k, position_ids, config)

        key_cache = jax.lax.dynamic_update_slice(
            cache["key"][layer_idx], k, (0, cache_index, 0, 0)
        )
        value_cache = jax.lax.dynamic_update_slice(
            cache["value"][layer_idx], v, (0, cache_index, 0, 0)
        )
        new_keys[layer_idx] = key_cache
        new_values[layer_idx] = value_cache

        full_k = _repeat_kv(key_cache, kv_repeats)
        full_v = _repeat_kv(value_cache, kv_repeats)
        attn_scores = jnp.einsum("bthd,bshd->bhts", q, full_k)
        attn_scores = attn_scores / jnp.sqrt(jnp.asarray(config.head_dim, dtype=jnp.float32))
        mask = _attention_mask(seq_len, full_k.shape[1], cache_index)
        attn_scores = jnp.where(mask, attn_scores, jnp.asarray(-1e30, dtype=attn_scores.dtype))
        attn_weights = jax.nn.softmax(attn_scores.astype(jnp.float32), axis=-1).astype(hidden.dtype)
        attn_out = jnp.einsum("bhts,bshd->bthd", attn_weights, full_v)
        attn_out = attn_out.reshape(batch_size, seq_len, config.hidden_size)
        hidden = residual + _linear(attn_out, attn["o_proj"], _optional(attn, "o_proj_bias"))

        residual = hidden
        x = _rms_norm(hidden, layer["post_attention_layernorm"], config.rms_norm_eps)
        mlp = layer["mlp"]
        gate = jax.nn.silu(_linear(x, mlp["gate_proj"]))
        up = _linear(x, mlp["up_proj"])
        hidden = residual + _linear(gate * up, mlp["down_proj"])

    hidden = _rms_norm(hidden, params["norm"], config.rms_norm_eps)
    lm_head = params.get("lm_head", params["embed_tokens"])
    logits = _linear(hidden, lm_head).astype(jnp.float32)
    return logits, {"key": tuple(new_keys), "value": tuple(new_values)}


def make_forward(config: Qwen2JAXConfig):
    @partial(_jax().jit, donate_argnames=("cache",))
    def forward(params, input_ids, cache, cache_index):
        return qwen2_forward(params, config, input_ids, cache, cache_index)

    return forward


def make_greedy_step(config: Qwen2JAXConfig):
    @partial(_jax().jit, donate_argnames=("cache",))
    def step(params, next_logits, cache, cache_index):
        jnp = _jnp()
        token = jnp.argmax(next_logits, axis=-1).astype(jnp.int32)
        logits, cache = qwen2_forward(params, config, token[:, None], cache, cache_index)
        return token, logits[:, -1, :], cache

    return step
