# mini-llm-inference-server

A complete LLM inference stack from scratch, including sampling, tokenization, a tiny transformer with KV caching, a paged attention allocator, continuous batching with scheduling, a streaming serving API, and a throughput/latency benchmark harness.

## Overview

This project builds a mini large-language model (LLM) inference server, which hosts a small transformer architecture built from scratch with random weights and processes text generation requests, mirroring the core architecture of production serving systems like vLLM.

## Architecture

The server samples from probabilities based on computed logits, which are raw, unnormalized score vectors, through greedy search, temperature scaling, and/or top-k/top-p filters.

A tiny vocabulary is then built with prompt encoding (turning text into token IDs) and decoding (turning token IDs into text) in a process called tokenization.

A tiny transformer with causal attention and decode forward passes then takes in the prompt. It utilizes a paged KV cache, which stores past computations for reference in future steps. The cache relates the context and label of the text (key) with the actual content inside it (value). Every new token computes attention scores against all cached keys to find similarities, so the cache helps avoid recomputing K/V projections for tokens already seen. The KV blocks are stored in an allocator that tracks them and their usage.

The model also utilizes continuous batching, which synchronizes decode steps through multiple sequences and turns them into text based on the KV cache. Scheduling checks capacities and queues or admits sequences in the continuous batching based on their priority. It can also preempt or queue them in a separate waiting heap. Since sequences finish at different times, naive batching will waste compute waiting for slower sequences. Continuous batching lets a finished sequence's slot get replaced by a new one immediately, instead of the whole batch waiting for the slowest sequence, keeping the GPU/compute busy.

As tokens are generated, the server emits streaming chunks per request. Once a request finishes, its chunks and output are collected into a completion response, which is then fed into the benchmark harness by measuring TTFT, inter-token latency, throughput, and latency percentiles.
