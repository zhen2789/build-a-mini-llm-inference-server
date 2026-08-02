from tokenizer import encode_prompt, decode_tokens
from model import model_prefill
from paging import free_sequence_blocks
from sequence import make_request, init_sequence_state, is_sequence_done
from scheduler import continuous_batch_step, priority_queue_push, schedule_step

def format_stream_chunk(request_id, token_id, token_text, finished):
    """Package a streaming token event into a chunk dict with keys request_id, token_id, text, finished"""
    return {
        'request_id': request_id,
        'token_id': token_id,
        'text': token_text,
        'finished': finished
    }

def submit_request(server_state, prompt, max_new_tokens, priority, vocab):
    """Encode the prompt, build a request record, push it on the waiting heap, return new id"""
    n = int(server_state['next_request_id'])
    new_id = f"req-{n}"
    if vocab['id_to_token'][0] == '<bos>':
        prompt_token_ids = encode_prompt(prompt, vocab, add_bos=True)
    else:
        prompt_token_ids = encode_prompt(prompt, vocab, add_bos=False)
    request = make_request(new_id, prompt_token_ids, max_new_tokens, {})
    server_state['waiting_heap'] = priority_queue_push(server_state['waiting_heap'], priority, request)
    server_state['next_request_id'] += 1
    return new_id

def drive_until_complete(server_state, params, allocator, sampling_config, vocab, max_steps):
    """Run the scheduler/prefill/decode loop until queues are empty or max_steps is hit"""
    server_state.setdefault('waiting_heap', [])
    server_state.setdefault('running', [])
    server_state.setdefault('completed', {})
    server_state.setdefault('streams', {})
    for i in range(max_steps):
        if len(server_state['waiting_heap']) == 0 and len(server_state['running']) == 0:
            break
        schedule = schedule_step(server_state['waiting_heap'], server_state['running'], allocator, allocator.get('block_size', 4), server_state.get('max_running', 8))
        server_state['running'] = schedule['running']
        for request in schedule['newly_admitted']:
            seq_dict = init_sequence_state(request, params)
            server_state['running'].append(seq_dict)
        prev_lens = {seq['request_id']: len(seq['generated']) for seq in server_state['running']}
        if len(server_state['running']) != 0:
            server_state['running'] = continuous_batch_step(params, server_state['running'], allocator, sampling_config)
        for seq in server_state['running']:
            start_idx = prev_lens[seq['request_id']]
            seq_is_done = is_sequence_done(seq, sampling_config.get('eos_token_id', -1))
            for new_token in seq['generated'][start_idx:]:
                text = decode_tokens([new_token], vocab, skip_special=True)
                stream_chunk = format_stream_chunk(seq['request_id'], new_token, text, seq_is_done)
                if seq['request_id'] not in server_state['streams']:
                    server_state['streams'][seq['request_id']] = []
                server_state['streams'][seq['request_id']].append(stream_chunk)
            if is_sequence_done(seq, sampling_config.get('eos_token_id', -1)) is True:
                free_sequence_blocks(allocator, seq['request_id'])
                server_state['completed'][seq['request_id']] = {
                    'output_ids': seq['generated'],
                    'chunks': server_state['streams'][seq['request_id']]
                    }
        server_state['running'] = [s for s in server_state['running'] if not is_sequence_done(s, sampling_config.get('eos_token_id', -1))]
    return list(server_state['completed'].values())

def collect_request_output(server_state, request_id):
    """Look up the completed record for request_id and return its output_ids and chunks"""
    if request_id not in server_state.get('completed', {}):
        return None
    return {
        'request_id': request_id,
        'output_ids': server_state['completed'][request_id]['output_ids'],
        'chunks': server_state['completed'][request_id]['chunks'] 
    }

def build_completion_response(server_state, request_id, vocab):
    """Build the final OpenAI-style completion dict from the completed record"""
    record = collect_request_output(server_state, request_id)
    if record is None:
        return None
    text = decode_tokens(record['output_ids'], vocab, skip_special=True)
    return {
        'request_id': request_id,
        'text': text,
        'output_ids': record['output_ids'].copy(),
        'finish_reason': server_state['completed'][request_id].get('finish_reason', 'stop')
    }
