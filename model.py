"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
def stable_softmax(logits):
    # TODO: compute a numerically stable softmax over the last axis of logits.
    max = np.max(logits, axis=-1, keepdims=True) # (..., 1), need to keep 1 for last broadcasting
    return (np.exp(logits - max)) / (np.sum(np.exp(logits - max), axis=-1, keepdims=True))
    pass

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    # TODO: scale logits by 1 / temperature; if temperature <= 0, return logits unchanged (greedy).
    if temperature <= 0:
        return logits
    else:
        return logits / temperature
    pass

# Step 3 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    """Mask logits outside the top-k per row to -inf."""
    # TODO: keep only the k largest logits along the last axis, set the rest to -inf
    V = logits.shape[-1]
    if k >= V:
        return logits
    threshold = np.partition(logits, -k, axis=-1)[..., -k:][..., 0:1] # 0:1 for matching dimensions
    mask = logits >= threshold
    return np.where(mask, logits, -np.inf)
    pass

# Step 4 - top_p_filter
def top_p_filter(logits, p):
    # TODO: keep smallest set of tokens whose cumulative prob >= p, mask the rest to -inf.
    probs = stable_softmax(logits)
    idx = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, idx, axis=-1)
    probs_sum = np.cumsum(sorted_probs, axis=-1)
    shifted_sum = np.insert(probs_sum[..., :-1], 0, 0, axis=-1)
    mask = p > shifted_sum
    new_arr = np.full_like(logits, False, dtype=bool)
    np.put_along_axis(new_arr, idx, mask, axis=-1)
    return np.where(new_arr, logits, -np.inf)
    pass

# Step 5 - sample_from_probs
def sample_from_probs(probs, rng):
    # TODO: draw a single token id from the categorical distribution probs using rng
    return int(rng.choice(len(probs), p=probs))
    pass

# Step 6 - greedy_select
def greedy_select(logits):
    # TODO: return the index of the maximum logit (ties -> lowest index).
    return int(np.argmax(logits))
    pass

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    # TODO: build a character-level vocab; specials get the lowest ids, then sorted unique chars.
    id_to_token = special_tokens.copy()
    chars = sorted(set("".join(corpus)))
    for char in chars:
        if char not in id_to_token:
            id_to_token.append(char)
    token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
    return {'token_to_id': token_to_id, 'id_to_token': id_to_token}
    pass

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    # TODO: encode text into token ids using vocab, optionally prepending <bos>.
    ids = []
    if add_bos is True and '<bos>' in vocab['token_to_id']:
        ids.append(vocab['token_to_id'].get('<bos>'))
    for c in text:
        if c in vocab['token_to_id']:
            ids.append(vocab['token_to_id'].get(c))
        elif '<unk>' in vocab['token_to_id']:
            ids.append(vocab['token_to_id'].get('<unk>'))
        else:
            continue
    return ids
    pass

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):
    # TODO: convert token ids back into a string using vocab['id_to_token'], optionally skipping specials.
    tokens = []
    for id in token_ids:
        if skip_special is True and vocab['id_to_token'][id].startswith('<') and vocab['id_to_token'][id].endswith('>'):
            continue
        else:
            tokens.append(vocab['id_to_token'][int(id)])
    result = ''.join(tokens)
    return result
    pass

# Step 10 - embed_tokens
import numpy as np

def embed_tokens(token_ids, embedding_matrix):
    # TODO: return the (T, D) embedding rows for each token id in token_ids
    return embedding_matrix[token_ids]
    pass

# Step 11 - linear_projection
def linear_projection(x, weight, bias=None):
    # TODO: Apply y = x @ weight + bias, with bias optional and broadcasting over leading axes.
    if bias is None:
        return x @ weight
    if bias is not None:
        return x @ weight + bias
    pass

# Step 12 - init_kv_cache
import numpy as np

def init_kv_cache(max_seq_len, d_model):
    # TODO: allocate empty K and V buffers and a length counter for a single sequence
    K = np.zeros((max_seq_len, d_model), dtype=np.float32)
    V = np.zeros((max_seq_len, d_model), dtype=np.float32)
    return {'K': K, 'V': V, 'length': 0}
    pass

# Step 13 - append_kv
import numpy as np

def append_kv(cache, k_new, v_new):
    # TODO: write k_new and v_new into the cache starting at cache['length'] and bump length.
    L = cache['length']
    t = k_new.shape[0]
    cache['K'][L:L+t] = k_new
    cache['V'][L:L+t] = v_new
    cache['length'] = L + t
    return cache
    pass

# Step 14 - causal_attention
import numpy as np

def causal_attention(q, k, v, is_causal=True):
    # TODO: scaled dot-product attention with optional causal mask, returns (Tq, D)
    Tq, D = q.shape[0], q.shape[-1]
    Tk = k.shape[0]
    scores = (q @ k.T) / np.sqrt(D) # (Tq, Tk), square root of D is for scaling
    if is_causal is True:
        mask = np.triu(np.ones((Tq, Tk), dtype=bool), k=1 + (Tk - Tq))
        scores[mask] = -np.inf 
        # np.triu gives upper-triangular mask of disallowed positions
        # np.ones, dtype=bool makes positions in upper-triangular mask True
        # k=1 + (Tk - Tq) ensures queries don't attend to keys (need help with this explanation)
    weights = stable_softmax(scores)
    return weights @ v # (Tq, D)
    pass

# Step 15 - model_prefill
def model_prefill(token_ids, params):
    # TODO: embed tokens, project Q/K/V, fill the KV cache, run causal attention, return last-position logits.
    x = embed_tokens(token_ids, params['embedding']) # (T, D)
    q = linear_projection(x, params['Wq'], bias=None) # (T, D)
    k = linear_projection(x, params['Wk'], bias=None) # (T, D)
    v = linear_projection(x, params['Wv'], bias=None) # (T, D)
    D = params['embedding'].shape[1]
    cache = init_kv_cache(params['max_seq_len'], D)
    appended_cache = append_kv(cache, k, v)
    attn_out = causal_attention(q, k, v, is_causal = True) # (T, D)
    out_proj = linear_projection(attn_out, params['Wo']) # (T, D)
    logits = linear_projection(out_proj[-1], params['W_out']) # (V, )
    return logits, appended_cache
    pass

# Step 16 - model_decode_step
def model_decode_step(token_id, cache, params):
    """Advance generation by one token using the existing KV cache."""
    # TODO: advance generation by one token using the existing KV cache and return next-token logits
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
    pass

# Step 17 - blocks_needed
def blocks_needed(num_tokens, block_size):
    # TODO: return the number of fixed-size blocks needed to store num_tokens tokens.
    if num_tokens == 0:
        return 0
    return (num_tokens + block_size - 1) // block_size
    pass

# Step 18 - init_block_allocator
def init_block_allocator(num_blocks, block_size, d_model):
    # TODO: build the paged KV allocator dict with K_blocks, V_blocks, free_list, seq_tables, and config.
    K_blocks = np.zeros((num_blocks, block_size, d_model), dtype=np.float32)
    V_blocks = np.zeros((num_blocks, block_size, d_model), dtype=np.float32)
    free_list = list(range(num_blocks))
    seq_tables = {}
    return {
        'K_blocks': K_blocks,
        'V_blocks': V_blocks,
        'free_list': free_list,
        'block_size': block_size,
        'num_blocks': num_blocks,
        'd_model': d_model,
        'seq_tables': seq_tables
    }
    pass

# Step 19 - allocate_block
def allocate_block(allocator, seq_id):
    # TODO: pop one free block id and append it to allocator['seq_tables'][seq_id]; raise RuntimeError if OOM.
    if len(allocator['free_list']) == 0:
        raise RuntimeError('KV cache OOM')
    popped_id = allocator['free_list'].pop()
    if seq_id not in allocator['seq_tables']:
        allocator['seq_tables'].setdefault(seq_id, [])
    allocator['seq_tables'][seq_id].append(popped_id)
    return popped_id
    pass

# Step 20 - free_block
def free_block(allocator, block_id):
    # TODO: return block_id to allocator['free_list']
    return allocator['free_list'].append(block_id)
    pass

# Step 21 - append_to_paged_cache
def append_to_paged_cache(allocator, seq_id, k_new, v_new):
    """Write t new K/V rows into the sequence's paged blocks, allocating as needed."""
    # TODO: append k_new/v_new (t, d_model) rows into seq_id's paged blocks, growing the block table when needed.
    if 'seq_lengths' not in allocator:
        allocator['seq_lengths'] = {}
    L = allocator['seq_lengths'].get(seq_id, 0)
    t = k_new.shape[0]
    block_count = blocks_needed(L + t, allocator['block_size'])
    if seq_id not in allocator['seq_tables']:
        allocate_block(allocator, seq_id)
    while len(allocator['seq_tables'][seq_id]) < block_count:
        allocate_block(allocator, seq_id)
    for i in range(t):
        token_pos = L + i
        block_idx = token_pos // allocator['block_size']
        token_slot = token_pos % allocator['block_size']
        block_id = allocator['seq_tables'][seq_id][block_idx]
        allocator['K_blocks'][block_id, token_slot] = k_new[i]
        allocator['V_blocks'][block_id, token_slot] = v_new[i]
    allocator['seq_lengths'][seq_id] = L + t
    pass

# Step 22 - gather_kv_from_blocks
def gather_kv_from_blocks(allocator, seq_id):
    # TODO: reconstruct contiguous (length, d_model) K and V from the sequence's paged blocks.
    length = allocator['seq_lengths'][seq_id]
    block_ids = allocator['seq_tables'][seq_id]
    d_model = allocator['d_model']
    block_size = allocator['block_size']
    K = np.zeros((length, d_model), dtype=np.float32)
    V = np.zeros((length, d_model), dtype=np.float32)
    for i, bid in enumerate(block_ids):
        start = i * block_size
        end = min((i+1) * block_size, length) # global row index where block ends
        n = min(block_size, length - i * block_size) # number of valid rows
        K[start:end] = allocator['K_blocks'][bid, :n]
        V[start:end] = allocator['V_blocks'][bid, :n]
    return K, V
    pass

# Step 23 - paged_attention_step
def paged_attention_step(q, allocator, seq_id):
    # TODO: gather K, V for seq_id from the paged allocator and run causal attention with q
    k, v = gather_kv_from_blocks(allocator, seq_id) # (length, d_model)
    output = causal_attention(q, k, v, is_causal=True)
    return output # (1, d_model)
    pass

# Step 24 - free_sequence_blocks
def free_sequence_blocks(allocator, seq_id):
    # TODO: release all blocks owned by seq_id and remove its entry from seq_tables.
    if seq_id in allocator['seq_tables']:
        block_ids = list(allocator['seq_tables'][seq_id])
        for block_id in block_ids:
            free_block(allocator, block_id)
        del allocator['seq_tables'][seq_id]
        return None
    pass

# Step 25 - kv_blocks_in_use
def kv_blocks_in_use(allocator):
    # TODO: report allocator usage as {'used': int, 'free': int, 'total': int}.
    free = len(allocator['free_list'])
    total = allocator['num_blocks']
    used = total - free
    return {
        'used': used,
        'free': free,
        'total': total
    }
    pass

# Step 26 - make_request
def make_request(request_id, prompt_token_ids, max_new_tokens, sampling_params):
    # TODO: package the request id, prompt tokens, generation budget, and sampling params into a dict.
    return {
        'request_id': request_id,
        'prompt_token_ids': prompt_token_ids.copy(),
        'max_new_tokens': max_new_tokens,
        'sampling_params': sampling_params
    }
    pass

# Step 27 - init_sequence_state
def init_sequence_state(request, params):
    # TODO: Initialize per-sequence state by running prefill and storing cache/logits.
    last_logits, cache = model_prefill(request['prompt_token_ids'], params)
    return {
        'request_id': request['request_id'],
        'prompt_token_ids': request['prompt_token_ids'].copy(),
        'generated_token_ids': [],
        'generated': [],
        'cache': cache,
        'last_logits': last_logits,
        'done': False,
        'sampling_params': request['sampling_params'],
        'max_new_tokens': request['max_new_tokens']
    }
    pass

# Step 28 - sequence_decode_step (not yet solved)
# TODO: implement

# Step 29 - is_sequence_done (not yet solved)
# TODO: implement

# Step 30 - generate_single_sequence (not yet solved)
# TODO: implement

# Step 31 - build_batch_step_input (not yet solved)
# TODO: implement

# Step 32 - batched_decode_step (not yet solved)
# TODO: implement

# Step 33 - static_batch_generate (not yet solved)
# TODO: implement

# Step 34 - has_free_capacity (not yet solved)
# TODO: implement

# Step 35 - continuous_batch_step (not yet solved)
# TODO: implement

# Step 36 - run_continuous_batching (not yet solved)
# TODO: implement

# Step 37 - priority_queue_push (not yet solved)
# TODO: implement

# Step 38 - priority_queue_pop (not yet solved)
# TODO: implement

# Step 39 - select_admissions (not yet solved)
# TODO: implement

# Step 40 - preempt_sequence (not yet solved)
# TODO: implement

# Step 41 - schedule_step (not yet solved)
# TODO: implement

# Step 42 - format_stream_chunk (not yet solved)
# TODO: implement

# Step 43 - submit_request (not yet solved)
# TODO: implement

# Step 44 - drive_until_complete (not yet solved)
# TODO: implement

# Step 45 - collect_request_output (not yet solved)
# TODO: implement

# Step 46 - build_completion_response (not yet solved)
# TODO: implement

# Step 47 - time_to_first_token (not yet solved)
# TODO: implement

# Step 48 - inter_token_latency (not yet solved)
# TODO: implement

# Step 49 - aggregate_throughput (not yet solved)
# TODO: implement

# Step 50 - latency_percentiles (not yet solved)
# TODO: implement

# Step 51 - run_throughput_latency_benchmark (not yet solved)
# TODO: implement

