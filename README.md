# nano-vllm-runtime

This repository stores my runtime-oriented work based on `nano-vllm`.

At the current stage, this project is still the imported baseline from `nano-vllm` and does not yet contain architectural upgrades over the original implementation.

The purpose of this repository is to iteratively build and benchmark a small LLM inference runtime around topics such as:

- serving scheduler behavior
- chunked prefill
- KV cache and block management
- runtime benchmarking and profiling

## Current Status

- baseline imported from `nano-vllm`
- local baseline tag: `v0-baseline`
- current focus: serving benchmark design and runtime architecture iteration

## Notes

- this is a personal runtime project for learning, benchmarking, and interview-oriented engineering work
- it is not an official `vLLM` implementation
