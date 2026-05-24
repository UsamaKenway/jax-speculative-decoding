from __future__ import annotations

from functools import partial

from .jax_qwen import Qwen2JAXConfig, qwen2_forward


def make_draft_k(config: Qwen2JAXConfig, k: int):
    import jax
    import jax.numpy as jnp

    @partial(jax.jit, donate_argnames=("cache",))
    def draft_k(params, next_logits, cache, cache_index):
        tokens = []
        token_logprobs = []
        for offset in range(k):
            log_probs = jax.nn.log_softmax(next_logits.astype(jnp.float32), axis=-1)
            token = jnp.argmax(log_probs, axis=-1).astype(jnp.int32)
            logprob = jnp.take_along_axis(log_probs, token[:, None], axis=-1)[:, 0]
            tokens.append(token)
            token_logprobs.append(logprob)
            logits, cache = qwen2_forward(params, config, token[:, None], cache, cache_index + offset)
            next_logits = logits[:, -1, :]
        return (
            jnp.stack(tokens, axis=1),
            jnp.stack(token_logprobs, axis=1),
            next_logits,
            cache,
        )

    return draft_k


def make_verify_greedy(config: Qwen2JAXConfig, k: int):
    import jax
    import jax.numpy as jnp

    @partial(jax.jit, donate_argnames=("cache",))
    def verify(params, next_logits, cache, cache_index, draft_tokens):
        logits, verified_cache = qwen2_forward(params, config, draft_tokens, cache, cache_index)
        first_target = jnp.argmax(next_logits, axis=-1).astype(jnp.int32)
        if k == 1:
            target_tokens = first_target[:, None]
        else:
            rest_targets = jnp.argmax(logits[:, :-1, :], axis=-1).astype(jnp.int32)
            target_tokens = jnp.concatenate((first_target[:, None], rest_targets), axis=1)

        accept = target_tokens == draft_tokens
        reject = jnp.logical_not(accept[0])
        first_reject = jnp.argmax(reject.astype(jnp.int32))
        accepted_count = jnp.where(jnp.all(accept[0]), k, first_reject).astype(jnp.int32)
        return accepted_count, target_tokens, logits[:, -1, :], verified_cache, accept

    return verify
