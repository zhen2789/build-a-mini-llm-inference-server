import numpy as np
from sampling import stable_softmax

def embed_tokens(token_ids, embedding_matrix):
    """Return the (T, D) embedding rows for each token id in token_ids"""
    return embedding_matrix[token_ids]
  
def linear_projection(x, weight, bias=None):
    """Apply y = x @ weight + bias, with bias optional and broadcasting over leading axes"""
    if bias is None:
        return x @ weight
    if bias is not None:
        return x @ weight + bias
        
def init_kv_cache(max_seq_len, d_model):
    """Allocate empty K and V buffers and a length counter for a single sequence"""
    K = np.zeros((max_seq_len, d_model), dtype=np.float32)
    V = np.zeros((max_seq_len, d_model), dtype=np.float32)
    return {'K': K, 'V': V, 'length': 0}
    
def append_kv(cache, k_new, v_new):
    """Write k_new and v_new into the cache starting at cache['length'] and bump length"""
    L = cache['length']
    t = k_new.shape[0]
    cache['K'][L:L+t] = k_new
    cache['V'][L:L+t] = v_new
    cache['length'] = L + t
    return cache
    
def causal_attention(q, k, v, is_causal=True):
    """Scaled dot-product attention with optional causal mask, returns (Tq, D)"""
    Tq, D = q.shape[0], q.shape[-1]
    Tk = k.shape[0]
    scores = (q @ k.T) / np.sqrt(D) # (Tq, Tk), square root of D is for scaling
    if is_causal is True:
        mask = np.triu(np.ones((Tq, Tk), dtype=bool), k=1 + (Tk - Tq)) # gives upper-triangular mask of disallowed positions (True)
        scores[mask] = -np.inf 
    weights = stable_softmax(scores)
    return weights @ v # (Tq, D)
    
def model_prefill(token_ids, params):
    """Embed tokens, project Q/K/V, fill the KV cache, run causal attention, return last-position logits"""
    x = embed_tokens(token_ids, params['embedding']) # (T, D)
    q = linear_projection(x, params['Wq'], bias=None) # (T, D)
    k = linear_projection(x, params['Wk'], bias=None) # (T, D)
    v = linear_projection(x, params['Wv'], bias=None) # (T, D)
    D = params['embedding'].shape[1]
    cache = init_kv_cache(params.get('max_seq_len', 2048), D)
    appended_cache = append_kv(cache, k, v)
    attn_out = causal_attention(q, k, v, is_causal = True) # (T, D)
    out_proj = linear_projection(attn_out, params['Wo']) # (T, D)
    logits = linear_projection(out_proj[-1], params['W_out']) # (V, )
    return logits, appended_cache
    
def model_decode_step(token_id, cache, params):
    """Advance generation by one token using the existing KV cache."""
    x = embed_tokens([token_id], params['embedding']) # (1, D)
    q = linear_projection(x, params['Wq'], bias=None) # (1, D)
    k_new = linear_projection(x, params['Wk'], bias=None) # (1, D)
    v_new = linear_projection(x, params['Wv'], bias=None) # (1, D)
    append_kv(cache, k_new, v_new)
    K = cache['K'][:cache['length']] # (T_total, D) -> extracts all historical tokens plus one new token
    V = cache['V'][:cache['length']] # (T_total, D)
    attn_out = causal_attention(q, K, V, is_causal=False) # (1, D)
    out_proj = linear_projection(attn_out, params['Wo']) # (1, D)
    logits = linear_projection(out_proj, params['W_out']) # (1, V)
    return logits[0], cache
