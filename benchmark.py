import time
from server import submit_request, drive_until_complete

def time_to_first_token(events):
    """Compute per-request TTFT from a list of timestamped serving events"""
    submit_times = {}
    token_times = {}
    time_diff = {}
    for request in events:
        if request['event'] == 'submit':
            submit_times[request['request_id']] = request['time']
        if request['event'] == 'token':
            req_id = request['request_id']
            if req_id not in token_times or request['time'] < token_times[req_id]:
                token_times[req_id] = request['time']
    for req in submit_times:
        if req in token_times:
            time_diff[req] = token_times[req] - submit_times[req]
    return time_diff

def inter_token_latency(events):
    """Compute mean inter-token latency per request from token-event timestamps"""
    token_times = {}
    mean_itl = {}
    for request in events:
        if request['request_id'] not in token_times and request['event'] == 'token':
            token_times[request['request_id']] = [request['time']]
        elif request['request_id'] in token_times and request['event'] == 'token':
            token_times[request['request_id']].append(request['time'])
    for token_req_id, times in token_times.items():
        times.sort()
        if len(times) < 2:
            mean_itl[token_req_id] = 0.0
        else:
            differences = [b - a for a, b in zip(times, times[1:])]
            mean = sum(differences) / len(differences)
            mean_itl[token_req_id] = mean
    return mean_itl

def aggregate_throughput(events, total_time):
    """Count tokens and finished requests in events and divide by total_time"""
    total_tokens = []
    total_requests = []
    for event in events:
        if event['type'] == 'finish':
            total_requests.append(event['request_id'])
        if event['type'] == 'first_token' or event['type'] == 'token':
            total_tokens.append(event['type'])
    return {
        'tokens_per_second': len(total_tokens) / total_time,
        'requests_per_second': len(total_requests) / total_time,
        'total_tokens': len(total_tokens),
        'total_requests': len(total_requests)
    }

def latency_percentiles(latencies, percentiles):
    """Return a dict mapping each percentile in `percentiles` to the corresponding latency value"""
    output = {}
    sorted_latencies = sorted(latencies)
    if len(sorted_latencies) == 0:
        return {float(p): 0.0 for p in percentiles}
    else:
        for p in percentiles:
            idx = (len(sorted_latencies) - 1) * (p / 100)
            lower_bound = int(idx)
            if len(sorted_latencies) > 1:
                upper_bound = min(lower_bound + 1, len(sorted_latencies) - 1)
                output[float(p)] = sorted_latencies[lower_bound] + (idx - lower_bound) * (sorted_latencies[upper_bound] - sorted_latencies[lower_bound])
            else:
                output[float(p)] = sorted_latencies[lower_bound]
    return output

def run_throughput_latency_benchmark(params, allocator, vocab, prompts, sampling_config, max_new_tokens, max_steps):
    """Submit prompts, drive the server, and reduce events into TTFT/ITL/throughput/percentile metrics"""
    server_state = {'running': [], 'waiting_heap': [], 'next_request_id': 0, 'stream_buffer': [], 'completed': {}}
    t0 = time.perf_counter()
    events = []
    for prompt in prompts:
        new_id = submit_request(server_state, prompt, max_new_tokens, 0, vocab)
        now = time.perf_counter()
        submit_event = {'request_id': new_id, 'event': 'submit', 'type': 'submit', 'time': now - t0}
        events.append(submit_event)
    processed_counts = {}
    for i in range(max_steps):
        chunk_list = drive_until_complete(server_state, params, allocator, sampling_config, vocab, max_steps=1)
        for req_id, chunks in server_state['streams'].items():
            already_processed = processed_counts.get(req_id, 0)
            first_tok_flag = (already_processed == 0)
            for new_chunk in server_state['streams'][req_id][already_processed:]:
                now = time.perf_counter()
                if first_tok_flag:
                    token_event = {'request_id': req_id, 'event': 'token', 'type': 'first_token', 'time': now - t0}
                    events.append(token_event)
                    first_tok_flag = False
                else:
                    token_event = {'request_id': req_id, 'event': 'token', 'type': 'token', 'time': now - t0}
                    events.append(token_event)
                if new_chunk['finished'] is True:
                    finish_event = {'request_id': req_id, 'event': 'finish', 'type': 'finish', 'time': now - t0}
                    events.append(finish_event)
            processed_counts[req_id] = len(server_state['streams'][req_id])
    now = time.perf_counter()
    total_time = now - t0
    latencies = list(time_to_first_token(events).values())
    return {
        'ttft': time_to_first_token(events),
        'itl': inter_token_latency(events),
        'throughput': aggregate_throughput(events, total_time),
        'percentiles': latency_percentiles(latencies, sampling_config.get('percentiles', [50, 90, 99])),
        'total_time': total_time
    }
