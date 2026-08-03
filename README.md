# Mini LLM Inference Server

A complete LLM inference stack from scratch, including sampling, tokenization, a tiny transformer with KV caching, a paged attention allocator, continuous batching with scheduling, a streaming serving API, and a throughput/latency benchmark harness.

## Overview

This project builds a mini large-language model (LLM) inference server, which hosts a small transformer architecture built from scratch with random weights and processes text generation requests, mirroring the core architecture of production serving systems like vLLM.

## Architecture

The server samples from probabilities based on computed logits, which are raw, unnormalized score vectors, through greedy search, temperature scaling, and/or top-k/top-p filters.

A tiny vocabulary is then built with prompt encoding (turning text into token IDs) and decoding (turning token IDs into text) in a process called tokenization.

A tiny transformer with causal attention and decode forward passes then takes in the prompt. It utilizes a paged KV cache, which stores past computations for reference in future steps. The cache relates the context and label of the text (key) with the actual content inside it (value). Every new token computes attention scores against all cached keys to find similarities, so the cache helps avoid recomputing K/V projections for tokens already seen. The KV blocks are stored in an allocator that tracks them and their usage.

The model also utilizes continuous batching, which synchronizes decode steps through multiple sequences and turns them into text based on the KV cache. Scheduling checks capacities and queues or admits sequences in the continuous batching based on their priority. It can also preempt or queue them in a separate waiting heap. Since sequences finish at different times, naive batching will waste compute waiting for slower sequences. Continuous batching lets a finished sequence's slot get replaced by a new one immediately, instead of the whole batch waiting for the slowest sequence, keeping the GPU/compute busy.

As tokens are generated, the server emits streaming chunks per request. Once a request finishes, its chunks and output are collected into a completion response, which is then fed into the benchmark harness by measuring TTFT, inter-token latency, throughput, and latency percentiles.

## Structure

- `sampling.py` - decoding math: stable softmax, temperature scaling, top-k/top-p filtering, sampling, greedy-select
- `tokenizer.py` - tiny vocabulary, encode/decode between text and token IDs
- `model.py` - transformer (embeddings, linear projections, KV cache, causal attention) with prefill and decode forward passes
- `paging.py` - KV allocator with paged appends/attention, a free list, gather, and usage tracking
- `sequence.py` - models per-request sequence state and drives generation, batching multiple sequences through synchronized decode steps
- `scheduler.py` - checks capacity and continuously queues, admits, or preempts sequences based on priority
- `server.py` - exposes a request interface with streaming chunks, submission, a driver loop, output collection, and completion responses
- `benchmark.py` - measures TTFT, inter-token latency, throughput, and latency percentiles
- `main.py` - entry point that connects modules together and runs sampling, tokenizer, single-sequence, server, and benchmark demos

## How to Run

Install the dependency:
```bash
pip install numpy
```
Then run:
```bash
python main.py
```
(Use `py main.py` instead if `python` isn't recognized in your system.)

## Sample Output

Running `python main.py` produces output like:
```
[sampling] greedy=4, sampled=4, probs=[0.22270014 0.         0.         0.         0.77729986]
[vocab] size=31, bos=1, eos=2
[tokenize] prompt='hello world' ids=[1, 12, 9, 16, 16, 19, 4, 27, 19, 22, 16, 8] roundtrip='hello world'
[single] generated ids=[13, 7, 12, 13, 12, 7] text='ichihc'
[allocator] blocks=32, block_size=8, usage={'used': 0, 'free': 32, 'total': 32}
[server] submitted requests: ['req-0', 'req-1', 'req-2', 'req-3']
[server] req-0: tokens=[12, 24, 16, 24, 6] text='htltb'
[server] req-1: tokens=[2, 2, 18, 25, 2] text='nu'
[server] req-2: tokens=[14, 10, 10, 7, 22] text='jffcr'
[server] req-3: tokens=[10, 6, 6, 1, 9] text='fbbe'
[allocator] post-run usage={'used': 0, 'free': 32, 'total': 32}
[bench] wall=0.0031s report keys=['ttft', 'itl', 'throughput', 'percentiles', 'total_time']
  ttft: {'req-0': 0.0008104720000000065, 'req-1': 0.0008103500000000152, 'req-2': 0.0008086520000000208}
  itl: {'req-0': 0.000523266750000001, 'req-1': 0.000522682999999996, 'req-2': 0.0005224547499999982}
  throughput: {'tokens_per_second': 4953.928465272916, 'requests_per_second': 990.7856930545832, 'total_tokens': 15, 'total_requests': 3}
  percentiles: {50.0: 0.0008103500000000152, 90.0: 0.0008104476000000083, 99.0: 0.0008104695600000067}
  total_time: 0.0030
```
Note: since the transformer uses randomly initialized (untrained) weights, generated text is not meant to be coherent. The demo is meant to validate the serving infrastructure (sampling, batching, caching, scheduling), not language modeling quality.

## Background

This project was built by working through Deep-ML's (deep-ml.com) curriculum. Deep-ML is an ML education website that provides a scaffold of what to do for projects like these. I worked through it step-by-step, and beyond just filling in the scaffold, I also debugged integration/shape mismatches across functions and restructured the code into modules.

Specifically, I fixed: 
- A shape bug of 2D `(1, V)` vs 1D `(V, )` logits
- A missing dict key (`eos_token_id`)
- A type mismatch (`completed`/`streams` dict vs list) across two functions
- A duplicate-computation bug where `model_prefill` was called twice
- Built a token-counter/streaming logic for the benchmark harness that wasn't in the original scaffold

Docstrings are adapted from the original scaffold's TODO comments. I cleaned them up and made them consistent across modules.
