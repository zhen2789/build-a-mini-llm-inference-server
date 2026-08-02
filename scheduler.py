import heapq
from sampling import stable_softmax, apply_temperature, top_k_filter, top_p_filter, sample_from_probs, greedy_select
from model import embed_tokens, linear_projection
from paging import blocks_needed, append_to_paged_cache, paged_attention_step, free_sequence_blocks
from sequence import init_sequence_state, is_sequence_done

def has_free_capacity(allocator, required_blocks):
    """Return True iff allocator has at least required_blocks free blocks"""
    if len(allocator['free_list']) >= required_blocks:
        return True
    else:
        return False

def continuous_batch_step(params, running, allocator, sampling_config):
    """Advance every active sequence in `running` by one decoded token using the paged allocator"""
    for seq in running:
        if seq['done'] == True:
            continue
        elif seq['done'] == False:
            last_tok = seq['token_ids'][-1:]
            x = embed_tokens(last_tok, params['embedding'])
            q = x @ params['Wq']
            k = x @ params['Wk']
            v = x @ params['Wv']
            seq_id = seq['request_id']
            append_to_paged_cache(allocator, seq_id, k, v)
            attn_output = paged_attention_step(q, allocator, seq_id) # (1, D)
            out_proj = linear_projection(attn_output, params['Wo'], bias=None) # (1, D)
            logits = linear_projection(out_proj, params['W_out'], bias=None) # (V, )
            if sampling_config.get('greedy', False) is True or sampling_config.get('temperature', 1.0) <= 0:
                next_token_id = int(greedy_select(logits[0]))
                seq['token_ids'].append(next_token_id)
                seq['generated'].append(next_token_id)
            else:
                logits = apply_temperature(logits[0], sampling_config['temperature'])
                if sampling_config.get('top_k', 0) > 0:
                    logits = top_k_filter(logits, sampling_config['top_k'])
                if sampling_config.get('top_p', 1.0) < 1.0:
                    logits = top_p_filter(logits, sampling_config['top_p'])
                probs = stable_softmax(logits)
                next_token_id = sample_from_probs(probs, sampling_config['rng'])
                seq['token_ids'].append(next_token_id)
                seq['generated'].append(next_token_id)
            seq['length'] += 1
            seq['done'] = is_sequence_done(seq, sampling_config.get('eos_token_id', -1))
    return running

def run_continuous_batching(params, requests, allocator, sampling_config, max_steps):
    """Drive the continuous-batching loop: admit, decode, retire finished sequences"""
    waiting = requests.copy()
    running = []
    completed = []
    for step in range(max_steps):
        while len(running) < sampling_config.get('max_running', len(requests)) and waiting:
            request = waiting.pop(0)
            seq = init_sequence_state(request, params)
            running.append(seq)
        running = continuous_batch_step(params, running, allocator, sampling_config)
        for sequence in running:
            if is_sequence_done(sequence, sampling_config.get('eos_token_id', -1)) is True:
                free_sequence_blocks(allocator, sequence['request_id'])
                output = {'request_id': sequence['request_id'], 'output_ids': list(sequence['generated'])}
                completed.append(output)
                running = [s for s in running if not is_sequence_done(s, sampling_config.get('eos_token_id', -1))]
    for remaining_seq in running:
        free_sequence_blocks(allocator, remaining_seq['request_id'])
        remaining_output = {'request_id': remaining_seq['request_id'], 'output_ids': list(remaining_seq['generated'])}
        completed.append(remaining_output)
    return completed

def priority_queue_push(heap, priority, request):
    """Push (priority, counter, request) onto the min-heap with stable tie-breaking"""
    counter = len(heap)
    heapq.heappush(heap, (priority, counter, request))
    return heap

def priority_queue_pop(heap):
    """Pop and return the request with the smallest priority from the min-heap, or None if empty"""
    if len(heap) == 0:
        return None
    priority, counter, request = heapq.heappop(heap)
    return request

def select_admissions(waiting_heap, allocator, block_size, max_admit):
    """Pop requests from the waiting priority queue and admit as many as the allocator can host"""
    admitted = []
    available = len(allocator['free_list'])
    while len(admitted) < max_admit:
        request = priority_queue_pop(waiting_heap)
        if request is None:
            break
        needed = blocks_needed(len(request['prompt_token_ids']), block_size)
        if needed <= available:
            admitted.append(request)
            available -= needed
        else:
            waiting_heap = priority_queue_push(waiting_heap, request['priority'], request)
            break
    return admitted

def preempt_sequence(sequence, allocator, waiting_heap):
    """Free the sequence's KV blocks and re-queue its request on the waiting heap"""
    free_sequence_blocks(allocator, sequence['request_id'])
    waiting_heap = priority_queue_push(waiting_heap, sequence['priority'], sequence)
    return sequence

def schedule_step(waiting_heap, running, allocator, block_size, max_running):
    """Preempt over-capacity sequences, then admit from the waiting heap up to max_running"""
    while len(running) > max_running:
        sequence = running.pop()
        sequence = preempt_sequence(sequence, allocator, waiting_heap)
    slots = max(0, max_running - len(running))
    admitted = select_admissions(waiting_heap, allocator, block_size, slots)
    return {'running': running, 'newly_admitted': admitted}
