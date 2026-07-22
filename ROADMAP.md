# Roadmap

This roadmap separates work we can do now from work that depends on the official
Kimi K3 release.

## Phase 0: Current Scaffold

Status: done.

- C synthetic MoE runtime
- packed int4 expert file format
- per-layer LRU cache
- usage stats and hot-expert preload
- batch-union route deduplication
- metadata probe
- converter planning skeleton
- regression tests for model-independent streaming behavior

## Phase 1: Work We Can Do Before Kimi K3 Downloads

These tasks do not require full Kimi K3 weights:

- split the single C runtime into clearer modules
- add a local safetensors metadata/index reader
- improve the converter skeleton so it can validate tensor maps without loading
  tensors
- add GitHub Actions CI for `make test`
- add issue templates and benchmark templates
- build a tiny Python oracle for the synthetic MoE fixture
- add a documented memory-budget planner
- add a benchmark harness for disk read patterns
- research Kimi K2/K2.5/K2.6 public configs as architectural precedent
- document every assumption that must be rechecked when K3 lands

## Phase 2: Metadata-Only Kimi K3 Lock

Start this as soon as K3 publishes metadata, before downloading full weights.

- fetch only config, tokenizer, modeling files, and safetensors index
- run `make probe`
- run `python3 tools/kimi_convert_skeleton.py --snapshot <snapshot>`
- lock architecture fields:
  - model type
  - hidden size
  - layer count
  - sparse layer schedule
  - expert count
  - routed experts per token
  - shared expert behavior
  - router score function
  - attention/KV format
  - context-extension method
  - tokenizer/chat template
- write the real Kimi K3 adapter plan

## Phase 3: First Real Kimi K3 Execution

Start this only after the model can be downloaded legally and safely.

- download a minimal shard subset if possible
- implement exact tensor-name mapping
- convert a tiny subset into the project container
- implement real dense path
- implement real attention/KV path
- implement real router and expert path
- compare against official Transformers with teacher forcing
- fix correctness before optimizing speed

## Phase 4: Colibri-Class Streaming

Once correctness is established:

- pack each routed expert into one contiguous read region
- add async expert I/O
- add direct I/O where useful
- add learned route-history hot cache
- add RAM/VRAM/disk planner
- add prefetch/lookahead experiments
- add profiling output
- benchmark cold, warm, and pinned runs

## Phase 5: Usable Runtime

After correctness and basic speed:

- tokenizer-driven CLI chat
- OpenAI-compatible local server
- streaming responses
- queue limits
- model doctor command
- model plan command
- release packages
- documentation for hardware expectations

## Phase 6: Optional Backends

These are important, but not required for the first faithful CPU runtime:

- CUDA dense/resident expert tier
- CUDA residual pipeline
- Metal backend for Apple Silicon
- NUMA placement policies
- grammar-constrained decoding
- speculative decoding if K3 provides a compatible draft head

## What Completes The Project

The project is complete when it can:

- convert official Kimi K3 weights into a streaming container
- run Kimi K3 locally without requiring all experts in RAM or VRAM
- match official Transformers on a teacher-forced correctness suite
- generate text through a CLI
- report memory placement and cache statistics
- document realistic speed on at least one small-RAM machine and one larger
  host
