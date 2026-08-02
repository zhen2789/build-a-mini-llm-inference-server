import numpy as np
from sampling import stable_softmax, apply_temperature, top_k_filter, top_p_filter, sample_from_probs, greedy_select
from model import model_prefill, model_decode_step

def make_request(request_id, prompt_token_ids, max_new_tokens, sampling_params):
    """Package the request id, prompt tokens, generation budget, and sampling params into a dict"""
    return {
        'request_id': request_id,
        'prompt_token_ids': prompt_token_ids.copy(),
        'max_new_tokens': max_new_tokens,
        'sampling_params': sampling_params
    }

def init_sequence_state(request, params):
    """Initialize per-sequence state by running prefill and storing cache/logits"""
    last_logits, cache = model_prefill(request['prompt_token_ids'], params)
    return {
        'request_id': request['request_id'],
        'prompt_token_ids': request['prompt_token_ids'].copy(),
        'token_ids': request['prompt_token_ids'].copy(),
        'length': len(request['prompt_token_ids']),
        'generated_token_ids': [],
        'generated': [],
        'cache': cache,
        'last_logits': last_logits,
        'done': False,
        'sampling_params': request.get('sampling_params', {}),
        'max_new_tokens': request['max_new_tokens']
    }

def sequence_decode_step(state, params, rng):
    """Sample next token from state['last_logits'], advance cache via model_decode_step, append token"""
    sp = state['sampling_params']
    sp.get('greedy', False)
    if sp.get('greedy', False) is True or sp.get('temperature', 1.0) <= 0:
        next_token_id = greedy_select(state['last_logits'])
    else:
        last_logits = apply_temperature(state['last_logits'], sp['temperature'])
        if sp.get('top_k', 0) > 0:
            last_logits = top_k_filter(last_logits, sp['top_k'])
        if sp.get('top_p', 1.0) < 1.0:
            last_logits = top_p_filter(last_logits, sp['top_p'])
        probs = stable_softmax(last_logits)
        next_token_id = sample_from_probs(probs, rng)
    new_logits, cache = model_decode_step(next_token_id, state['cache'], params)
    state['cache'] = cache
    state['last_logits'] = new_logits
    state['generated'].append(next_token_id)
    return next_token_id, state

def is_sequence_done(state, eos_token_id):
    """Return True if state has hit max_new_tokens budget or last generated token is EOS"""
    if len(state['generated']) > 0:
        if len(state['generated']) >= state['max_new_tokens'] or state['generated'][-1] == eos_token_id:
            return True
        else:
            return False
    else:
        return False

def generate_single_sequence(request, params, eos_token_id, rng):
    """Drive end-to-end generation for one request and return only the generated token ids"""
    seq_state = init_sequence_state(request, params) # initialize per-sequence state from prompt
    done_condition = is_sequence_done(seq_state, eos_token_id)
    while done_condition is False:
        next_token_id, state = sequence_decode_step(seq_state, params, rng) # keep producing tokens until sequence is finished
        done_condition = is_sequence_done(seq_state, eos_token_id)
    return list(seq_state['generated']) # return list of generated token ids not including prompt tokens

def build_batch_step_input(sequences):
    """Collect the last token id from each non-done sequence into a (B,) int64 array"""
    active_indices = []
    id_list = []
    for i, seq in enumerate(sequences):
        if seq['done'] == False:
            active_indices.append(i)
            id_list.append(seq['token_ids'][-1])
    input_ids = np.array(id_list, dtype=np.int64)
    return {
        'active_indices': active_indices,
        'input_ids': input_ids
    }

def batched_decode_step(params, sequences, sampling_config):
    """Run one synchronized decode step across active sequences"""
    batch = build_batch_step_input(sequences)
    for idx, tok in zip(batch['active_indices'], batch['input_ids']):
        seq = sequences[idx]
        logits, _ = model_decode_step(int(tok), seq['kv_cache'], params)
        if sampling_config.get('greedy', False) is True or sampling_config.get('temperature', 1.0) <= 0:
            next_token_id = int(greedy_select(logits))
        else:
            logits = apply_temperature(logits, sampling_config['temperature'])
            if sampling_config.get('top_k', 0) > 0:
                logits = top_k_filter(logits, sampling_config['top_k'])
            if sampling_config.get('top_p', 1.0) < 1.0:
                logits = top_p_filter(logits, sampling_config['top_p'])
            probs = stable_softmax(logits)
            next_token_id = sample_from_probs(probs, sampling_config['rng'])
        seq['token_ids'].append(next_token_id)
    return sequences

def static_batch_generate(params, requests, sampling_config, max_new_tokens):
    """Run prefill for all requests, then iterate batched decode steps until each
    sequence hits its per-request budget or the global max_new_tokens cap"""
    output = []
    for request in requests:
        state = init_sequence_state(request, params)
        cap = min(state['max_new_tokens'], max_new_tokens)
        for i in range(max_new_tokens):
            if sampling_config.get('greedy', False) is True or sampling_config.get('temperature', 1.0) <= 0:
                next_token_id = int(greedy_select(state['last_logits']))
                state['generated_token_ids'].append(next_token_id)
            else:
                logits = apply_temperature(state['last_logits'], sampling_config['temperature'])
                if sampling_config.get('top_k', 0) > 0:
                    logits = top_k_filter(logits, sampling_config['top_k'])
                if sampling_config.get('top_p', 1.0) < 1.0:
                    logits = top_p_filter(logits, sampling_config['top_p'])
                probs = stable_softmax(logits)
                next_token_id = sample_from_probs(probs, sampling_config['rng'])
                state['generated_token_ids'].append(next_token_id)
            if len(state['generated_token_ids']) >= cap:
                break
            else:
                logits, cache = model_decode_step(next_token_id, state['cache'], params)
                state['last_logits'] = logits
        record = {
            'request_id': request['request_id'],
            'output_ids': state['generated_token_ids']
        }
        output.append(record)
    return output
