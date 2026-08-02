import numpy as np

def stable_softmax(logits):
    """Compute a numerically stable softmax over the last axis of logits"""
    max = np.max(logits, axis=-1, keepdims=True) # (..., 1), need to keep 1 for last broadcasting
    return (np.exp(logits - max)) / (np.sum(np.exp(logits - max), axis=-1, keepdims=True))

def apply_temperature(logits, temperature):
    """Scale logits by 1 / temperature; if temperature <= 0, return logits unchanged (greedy)"""
    if temperature <= 0:
        return logits
    else:
        return logits / temperature

def top_k_filter(logits, k):
    """Mask logits outside the top-k per row to -inf"""
    V = logits.shape[-1]
    if k >= V:
        return logits
    threshold = np.partition(logits, -k, axis=-1)[..., -k:][..., 0:1] # keep as (..., 1) so it broadcasts against logits
    mask = logits >= threshold
    return np.where(mask, logits, -np.inf)

def top_p_filter(logits, p):
    """Keep smallest set of tokens whose cumulative prob >= p, mask the rest to -inf"""
    probs = stable_softmax(logits)
    idx = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, idx, axis=-1)
    probs_sum = np.cumsum(sorted_probs, axis=-1)
    shifted_sum = np.insert(probs_sum[..., :-1], 0, 0, axis=-1)
    mask = p > shifted_sum
    new_arr = np.full_like(logits, False, dtype=bool)
    np.put_along_axis(new_arr, idx, mask, axis=-1)
    return np.where(new_arr, logits, -np.inf)

def sample_from_probs(probs, rng):
    """Draw a single token id from the categorical distribution probs using rng"""
    return int(rng.choice(len(probs), p=probs))

def greedy_select(logits):
    """Return the index of the maximum logit (ties -> lowest index)"""
    return int(np.argmax(logits))
