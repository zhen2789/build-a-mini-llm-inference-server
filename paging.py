import numpy as np
from model import causal_attention

def blocks_needed(num_tokens, block_size):
    """Return the number of fixed-size blocks needed to store num_tokens tokens"""
    if num_tokens == 0:
        return 0
    return (num_tokens + block_size - 1) // block_size

def init_block_allocator(num_blocks, block_size, d_model):
    """Build the paged KV allocator dict with K_blocks, V_blocks, free_list, seq_tables, and config"""
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

def allocate_block(allocator, seq_id):
    """Pop one free block id and append it to allocator['seq_tables'][seq_id]; raise RuntimeError if OOM"""
    if len(allocator['free_list']) == 0:
        raise RuntimeError('KV cache OOM')
    popped_id = allocator['free_list'].pop()
    if seq_id not in allocator['seq_tables']:
        allocator['seq_tables'].setdefault(seq_id, [])
    allocator['seq_tables'][seq_id].append(popped_id)
    return popped_id

def free_block(allocator, block_id):
    """Return block_id to allocator['free_list']"""
    return allocator['free_list'].append(block_id)

def append_to_paged_cache(allocator, seq_id, k_new, v_new):
    """Write t new K/V rows into the sequence's paged blocks, allocating as needed."""
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

def gather_kv_from_blocks(allocator, seq_id):
    """Reconstruct contiguous (length, d_model) K and V from the sequence's paged blocks"""
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

def paged_attention_step(q, allocator, seq_id):
    """Gather K, V for seq_id from the paged allocator and run causal attention with q"""
    k, v = gather_kv_from_blocks(allocator, seq_id) # (length, d_model)
    output = causal_attention(q, k, v, is_causal=True)
    return output # (1, d_model)

def free_sequence_blocks(allocator, seq_id):
    """Release all blocks owned by seq_id and remove its entry from seq_tables"""
    if seq_id in allocator['seq_tables']:
        block_ids = list(allocator['seq_tables'][seq_id])
        for block_id in block_ids:
            free_block(allocator, block_id)
        del allocator['seq_tables'][seq_id]
        return None

def kv_blocks_in_use(allocator):
    """Report allocator usage as {'used': int, 'free': int, 'total': int}"""
    free = len(allocator['free_list'])
    total = allocator['num_blocks']
    used = total - free
    return {
        'used': used,
        'free': free,
        'total': total
    }
