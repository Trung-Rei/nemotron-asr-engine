# How the engine got fast: an optimization journey

*[Tiếng Việt](optimization-journey.vi.md)*

This page tells the story in the order it happened: what was measured, what it showed, what changed, and the ideas that did not work. It is meant as a practical guide for anyone who wants to serve a streaming model to many concurrent users from Python. The current numbers are in [benchmarks.md](benchmarks.md), and the resulting design is in [architecture.md](architecture.md).

## 0. Start from a reference you trust

Before optimizing, the model was ported from NeMo to plain PyTorch ([`research/nemotron_lite`](../research/nemotron_lite)):

- It reads the checkpoint's `model.safetensors` directly.
- It needs no NeMo, Lightning, Hydra or transformers at run time.

The port was checked at three levels:

| Level | Check | Result |
|---|---|---|
| Layer by layer | NeMo's intermediate tensors dumped on a padded batch, compared with the port's | match within 1e-6 |
| Transcripts | 500 utterances offline | WER equal to NeMo, ~99% identical hypotheses |
| Streaming | streaming vs offline output | 100% identical |

The remaining tiny differences came from one source: NeMo's fused Triton subsampling kernel. With it turned off, every layer matches.

**Lesson:** a bit-exact reference turns every later optimization into a yes/no question ("are the tokens still identical?"). Without one, every speed-up is followed by an argument about whether accuracy moved.

## 1. Profile first: the step was bound by Python, not the GPU

Batching streams together is the obvious first step. A profile of one batched streaming step showed a surprise: the step cost the same at batch 1 and at batch 32. The GPU was not the limit.

| Finding | Detail |
|---|---|
| Launch count | ~1,750 kernel launches per step. Issuing the 24 layers from Python took longer than running them. |
| Slowest kernel | The 1×1 "pointwise" convolutions of the Conformer conv module took the largest share of GPU time. They ran through cuDNN about 10× slower than the same math as a matrix multiply. |
| Host syncs | Greedy decoding synced with the host for nearly every emitted token (`int(k[i])` in a loop). |

**Lesson:** profile before choosing what to optimize. Look at both CPU time per step and GPU kernel time per step. If they differ a lot, you know which side is the bottleneck.

## 2. Three fixes to the model code

1. **Pointwise conv → `F.linear`.** Same weights, reshaped. This was the biggest GPU-side win for one line of code.
2. **Slot-major cache pools and a CUDA graph per batch bucket.**
   - Each stream gets a fixed slot in pre-allocated cache tensors, so a batch is a gather → compute → scatter over slot indices.
   - With static shapes per bucket, the encoder can be captured as a CUDA graph and replayed with one launch.
3. **Greedy decoding without per-token syncs.** Labels stay on the GPU, and the host syncs once per symbol round instead of once per token.

Tokens stayed identical. The step became GPU-bound for the first time.

## 3. Fused attention

Relative-position attention in the reference was several matmuls plus a "shift" trick, with K/V concatenated from cache and chunk each step. The fused version:

- computes Q, K and V with one GEMM;
- precomputes the shifted position keys, so the position term becomes an additive attention bias;
- calls PyTorch's `scaled_dot_product_attention` (memory-efficient backend) once over cache + chunk keys;
- writes new K/V into a cache buffer that is rolled during the scatter, with no separate copy.

Strict-fp32 parity: 0 token differences. With TF32 matmuls, a handful of tokens differ in a few streams out of thousands of tokens. That is the expected numerical noise of TF32, not a bug.

## 4. From a benchmark loop to an engine: asyncio API and a GPU worker thread

A synchronous benchmark loop is the ideal case: it owns the process. A real application has an asyncio event loop that receives audio from hundreds of connections. The engine therefore became:

- an **asyncio API**: `open_stream`, `push`, `end_turn`, `async for event in stream`;
- a single **worker thread** that owns the GPU.

The first version worked, but it was slower than the benchmark loop at the same load. CPU use was ~100% of one core.

**Diagnosis: the GIL.** Every PyTorch call releases the GIL and must reacquire it. When the event loop is busy handling packets, the worker waits for the GIL before each op, and the GPU waits for the worker. The worker's time was measured as wall time minus thread CPU time, per phase. The waits sat exactly on the GPU's critical path: submitting the step and the greedy loop.

## 5. Pipeline the worker

1. **One shared inbox** (a deque) for every command from the event loop. The worker handles only streams with news, instead of scanning every stream each round.
2. **Pinned staging buffers.** Host-to-device copies became non-blocking.
3. **Prepare batch N+1 while the GPU runs batch N.**
4. **Move work off the host.** Log-mel and preemphasis moved to the GPU, batched. Each audio packet is now one copy into a per-stream linear buffer, and chunk windows are copied straight into staging.
5. **Admission control.** `open_stream` raises `EngineOverloaded` when the oldest waiting chunk is older than `max_lag_ms`. An overloaded engine refuses new streams instead of letting every stream's latency explode.

Throughput rose, but the greedy decode was still host-driven: many small ops, each one a GIL handoff.

## 6. The whole step in one CUDA graph

The decisive change was to put **everything** in the step graph: preemphasis, log-mel, subsampling, cache gather, 24 layers, projector, cache scatter and the greedy decode.

The hard part was greedy RNN-T, which has data-dependent control flow: a variable number of tokens per frame and per stream. The solution is a **per-stream frame pointer**. At each step, every stream either emits one token or advances one frame. Twelve such branch-free steps run in the graph, and a small continuation graph runs when a stream still has frames left. For each stream, the computation is the same sequence as the frame-by-frame loop, so tokens stay identical.

One more detail: padding rows in a batch point to a **scratch slot**, so their writes never hit a real stream.

After this change, the host issues about 6 PyTorch calls plus one graph replay per step. The GIL wait on the critical path almost disappeared, and 20 ms and 100 ms packets gave the same step rate. Contention no longer decided throughput.

## 7. Pick the batch when the GPU frees up

The pipeline from step 5 chose batch N+1 *while* batch N ran. Chunks that became ready during that step therefore waited a full extra step.

The final worker loop does this instead:

1. While the GPU runs, poll the step's CUDA event and drain the inbox in between.
2. When the event fires, collect.
3. Then pick the next batch, prepare it, and submit.

A chunk now waits for at most the step in flight. Median chunk lag dropped noticeably, and process CPU fell by about half, because the worker no longer spins inside `cudaStreamSynchronize`.

## What did not work

| Idea | Why it was dropped |
|---|---|
| A fixed number of greedy symbols per frame inside the graph | Simple, but real speech averages several tokens per chunk, and almost every batch had a frame with more symbols than the cap. The frame-pointer design is exact and nearly as cheap. |
| Draining the inbox on the event loop, with a lock per stream | Measured the same as draining on the worker while the GPU runs, but shares mutable state between threads. Reverted for simplicity. |
| A separate thread for draining | One more thread competing for the same GIL, for work that is already off the critical path. |
| Greedy and encoder on two CUDA streams | Once greedy is ~2 ms inside the graph there is little to overlap, and concurrent graph replays on a shared GPU are a known source of faults. |
| Attaching py-spy to the live process | It killed the container. Wrapping the worker's methods from outside (wall vs thread-CPU time) plus CUDA events was enough to find the GIL waits. |

## Where it stands, and what is next

**Calling the engine directly,** it is GPU-bound. The 24 Conformer layers take most of the step; [architecture.md](architecture.md#where-one-step-spends-its-time) has the breakdown.

**Through the WebSocket server,** the event loop is the bottleneck. It handles every WebSocket frame (50 per second per stream with 20 ms packets) and competes with the worker for the GIL. Larger packets help a lot.

Ideas for the next steps:

1. Split the WebSocket front end into its own process(es). Send audio batched per ~10 ms to the engine process.
2. Remove the cache gather/scatter: attention reads K/V in place from the slot pool (paged-attention style).
3. Write a small-query attention kernel. With 4 queries per stream, SDPA spends far longer than reading K/V would take.
4. Try a bf16 encoder (WER must be re-checked) and `max_batch=128`.
5. Avoid tiny first-chunk batches, and pre-copy windows of chunks that are already complete.
