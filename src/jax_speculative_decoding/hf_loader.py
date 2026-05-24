from __future__ import annotations

from typing import Any

from .jax_qwen import Qwen2JAXConfig


def _jax_modules():
    import jax
    import jax.numpy as jnp

    return jax, jnp


def select_jax_device(index: int):
    import jax

    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        devices = jax.devices()
    if index >= len(devices):
        raise ValueError(f"Requested JAX device {index}, but only found {len(devices)} devices: {devices}")
    return devices[index]


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_hf_config(model_id: str) -> Qwen2JAXConfig:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    if getattr(config, "model_type", None) != "qwen2":
        raise ValueError(f"Expected a Qwen2-family model, got model_type={config.model_type!r}")
    return Qwen2JAXConfig.from_hf_config(config)


def load_causal_lm(model_id: str, *, dtype: Any, **kwargs):
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def _to_jax(tensor, *, device: Any, dtype: Any):
    jax, jnp = _jax_modules()
    import torch

    if tensor.dtype == torch.bfloat16:
        array = tensor.detach().cpu().float().numpy()
    else:
        array = tensor.detach().cpu().numpy()
    return jax.device_put(jnp.asarray(array, dtype=dtype), device)


def _required(state: dict[str, Any], name: str):
    if name not in state:
        raise KeyError(f"Missing expected weight {name}")
    return state[name]


def _optional_linear_bias(state: dict[str, Any], name: str, *, device: Any, dtype: Any):
    if name not in state:
        return None
    return _to_jax(state[name], device=device, dtype=dtype)


def state_dict_to_jax_qwen_params(
    state: dict[str, Any],
    config: Qwen2JAXConfig,
    *,
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "embed_tokens": _to_jax(_required(state, "model.embed_tokens.weight"), device=device, dtype=dtype),
        "norm": _to_jax(_required(state, "model.norm.weight"), device=device, dtype=dtype),
    }
    if "lm_head.weight" in state:
        params["lm_head"] = _to_jax(state["lm_head.weight"], device=device, dtype=dtype)

    layers = []
    for idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{idx}"
        attn_prefix = f"{prefix}.self_attn"
        mlp_prefix = f"{prefix}.mlp"
        layer = {
            "input_layernorm": _to_jax(
                _required(state, f"{prefix}.input_layernorm.weight"), device=device, dtype=dtype
            ),
            "post_attention_layernorm": _to_jax(
                _required(state, f"{prefix}.post_attention_layernorm.weight"), device=device, dtype=dtype
            ),
            "self_attn": {
                "q_proj": _to_jax(_required(state, f"{attn_prefix}.q_proj.weight"), device=device, dtype=dtype),
                "k_proj": _to_jax(_required(state, f"{attn_prefix}.k_proj.weight"), device=device, dtype=dtype),
                "v_proj": _to_jax(_required(state, f"{attn_prefix}.v_proj.weight"), device=device, dtype=dtype),
                "o_proj": _to_jax(_required(state, f"{attn_prefix}.o_proj.weight"), device=device, dtype=dtype),
            },
            "mlp": {
                "gate_proj": _to_jax(_required(state, f"{mlp_prefix}.gate_proj.weight"), device=device, dtype=dtype),
                "up_proj": _to_jax(_required(state, f"{mlp_prefix}.up_proj.weight"), device=device, dtype=dtype),
                "down_proj": _to_jax(_required(state, f"{mlp_prefix}.down_proj.weight"), device=device, dtype=dtype),
            },
        }
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            bias = _optional_linear_bias(state, f"{attn_prefix}.{proj}.bias", device=device, dtype=dtype)
            if bias is not None:
                layer["self_attn"][f"{proj}_bias"] = bias
        layers.append(layer)

    params["layers"] = tuple(layers)
    return params


def load_qwen2_jax_params(model_id: str, *, device: Any, dtype: Any | None = None):
    jax, jnp = _jax_modules()
    import torch
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    config = Qwen2JAXConfig.from_hf_config(hf_config)
    dtype = dtype or jnp.bfloat16
    model = load_causal_lm(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    params = state_dict_to_jax_qwen_params(model.state_dict(), config, device=device, dtype=dtype)
    del model
    return config, jax.device_put(params, device)
