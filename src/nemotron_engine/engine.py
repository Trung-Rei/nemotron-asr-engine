"""Asynchronous multi-stream streaming ASR.

    engine = await AsyncNemotronEngine.create(EngineConfig(max_streams=512))
    stream = engine.open_stream(sample_rate=8000, encoding="mulaw")   # EngineFull when no slot is free
    await stream.push(packet)          # any packet size, returns at once
    stream.end_turn()                  # the client's endpointer decided the utterance is over
    async for ev in stream: ...        # TranscriptEvent: "partial" per chunk with new tokens, one "final" per turn
    await stream.close()
    await engine.aclose()

Threads: the event loop only decodes packets and appends them, with turn markers, to one engine inbox. One
worker thread owns the GPU. While the GPU runs a batch (one CUDA graph replay for the whole step: log-mel,
encoder, greedy RNN-T), the worker polls the step's CUDA event and drains the inbox in between, so stream state is
current when the step ends. Then it collects the tokens, posts them to the loop in one `call_soon_threadsafe`,
picks the next batch from the streams whose next chunk is complete (oldest first, same phase, up to max_batch),
writes their sample windows into a pinned staging buffer and submits. Picking right when the GPU frees up, rather
than while it is still busy, means a chunk waits for at most the step in flight. Text is detokenized on the loop.

Admission: `open_stream` raises `EngineOverloaded` (an `EngineFull`) while the oldest chunk waiting for the GPU
is older than `max_lag_ms`, so a saturated engine refuses new streams instead of letting every stream's latency grow.

Turns: `end_turn(pad=True)` pads the stream with zeros to the end of the current chunk, so the final follows the
decode of that chunk instead of waiting for up to one more chunk of real audio. The stream keeps its caches; the
next turn continues after the padding. Tokens of chunks up to the boundary belong to the ending turn.
"""

from __future__ import annotations

import asyncio
import collections
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .audio import AudioDecoder
from .batch import BatchStepper
from .model import NemotronModel
from .stream_audio import Chunking, StreamAudio

_LANG_TAG = re.compile(r"\s*<[a-z]{2,3}(?:-[A-Z]{2})?>\s*")


class EngineFull(RuntimeError):
    """No free stream slot."""


class EngineOverloaded(EngineFull):
    """The engine is behind (chunk lag above EngineConfig.max_lag_ms); try again later."""


class EngineClosed(RuntimeError):
    """The engine or the stream is closed (or the engine's worker failed)."""


@dataclass
class EngineConfig:
    lookahead: int = 3  # chunk = (lookahead + 1) * 80 ms; the model supports 0, 3, 6, 13
    max_streams: int = 512
    max_batch: int = 64
    language: str = "auto"  # language prompt ("auto", "en-US", "vi-VN", ...); open_stream(language=...) overrides it
    model_dir: str | None = None  # None: download from the HF hub
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    matmul_precision: str = "high"  # TF32 matmuls
    cuda_graphs: bool = True
    greedy_steps: int = 12  # greedy RNN-T steps in the step graph (a stream needs chunk frames + its tokens);
    # streams that need more run on in continuation graphs of `chunk` steps
    idle_wait_s: float = 0.002  # worker sleep when no chunk is ready (a push wakes it earlier)
    poll_wait_s: float = 0.0002  # while the GPU runs a step and the inbox is empty
    max_lag_ms: float | None = 200.0  # refuse new streams while the chunk lag is above this; None: never
    record_chunks: bool = False  # keep (t_ready, t_start, t_emit, batch) per decoded chunk in engine.chunk_log


@dataclass
class TranscriptEvent:
    kind: str  # "partial" | "final"
    turn: int
    text: str  # full text of the turn so far
    delta: str  # appended since the previous event of this turn (the full text if it was rewritten)
    tokens: list[int]  # tokens of the turn so far
    audio_ms: float  # stream audio (incl. flush padding) decoded up to
    t_ready: float  # perf_counter when the chunk's audio was complete (end_turn time for a flushed chunk)
    t_emit: float  # perf_counter when the worker produced the result


class AsrStream:
    """One live audio stream (one client connection), created by `AsyncNemotronEngine.open_stream`."""

    def __init__(self, engine: "AsyncNemotronEngine", slot: int, decoder: AudioDecoder):
        self._engine, self.slot, self._decoder = engine, slot, decoder
        self._state: _State | None = None  # worker-side state, set by open_stream
        self._events: asyncio.Queue = asyncio.Queue()
        self._closed = asyncio.Event()
        self._closing = False
        self.turn = 0
        self._tokens: list[int] = []
        self._text = ""

    # ---------------------------------------------------------------- input
    def push_nowait(self, data) -> None:
        """Queue a packet: bytes in the stream's encoding, or a float32 / int16 array."""
        self._check_open()
        x = self._decoder(data)
        if x.size:
            self._send("audio", x)

    async def push(self, data) -> None:
        self.push_nowait(data)

    def end_turn(self, pad: bool = True) -> None:
        """The user turn is over; a final event follows. pad=False waits for real audio to complete the chunk."""
        self._check_open()
        self._send("end", pad)

    async def close(self) -> None:
        """End an open turn, wait for its final, release the slot. The event iterator then stops."""
        if not self._closing:
            self._closing = True
            self._send("close")
        await self._closed.wait()

    def _check_open(self) -> None:
        if self._closing:
            raise EngineClosed("stream is closed")

    def _send(self, kind: str, arg=None) -> None:
        self._engine._post((self._state, kind, time.perf_counter(), arg))

    # ---------------------------------------------------------------- output
    def __aiter__(self):
        return self

    async def __anext__(self) -> TranscriptEvent:
        ev = await self._events.get()
        if ev is None:
            raise StopAsyncIteration
        if isinstance(ev, BaseException):
            raise ev
        return ev

    def _on_tokens(self, kind: str, new: list[int], audio_ms: float, t_ready: float, t_emit: float) -> None:
        if new:
            self._tokens.extend(new)
            text = _LANG_TAG.sub(" ", self._engine.tokenizer.decode(self._tokens)).strip()
        else:
            text = self._text
        delta = text[len(self._text) :] if text.startswith(self._text) else text
        self._events.put_nowait(TranscriptEvent(kind, self.turn, text, delta, list(self._tokens), audio_ms, t_ready, t_emit))
        self._text = text
        if kind == "final":
            self.turn += 1
            self._tokens, self._text = [], ""

    def _on_released(self, err: BaseException | None) -> None:
        if err is not None:
            self._events.put_nowait(err)
        self._events.put_nowait(None)
        self._closing = True
        self._closed.set()


@dataclass
class _State:
    """Worker-side state of one stream."""

    stream: AsrStream
    language: str | None
    audio: StreamAudio
    chunks_taken: int = 0  # chunks handed to a batch
    chunks_done: int = 0  # chunks decoded
    ready_t: collections.deque = field(default_factory=collections.deque)  # completion time per chunk not yet taken
    n_complete: int = 0
    real_end: int = 0  # samples up to the last real (non-padding) sample
    boundaries: collections.deque = field(default_factory=collections.deque)  # chunk counts that end turns
    last_boundary: int = 0
    closing: bool = False


@dataclass
class _Batch:
    states: list[_State]
    first: bool
    t_ready: list[float]
    width: int  # samples per window
    t_start: float = 0.0


class AsyncNemotronEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.chunk_log: list[tuple[float, float, float, int]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake = threading.Event()
        self._stop = False
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._free = list(range(cfg.max_streams - 1, -1, -1))  # slot numbers, guarded by _lock
        self._inbox: collections.deque = collections.deque()  # (state, kind, t, arg) from the loop, in order
        self._active: dict[int, _State] = {}  # slot -> state, worker thread only
        self._lag_ms = 0.0  # age of the oldest chunk waiting for the GPU, at the last pick
        self._out: list = []  # results of this worker round, posted to the loop at once
        self._stats = {"steps": 0, "chunks": 0, "last_batch": 0, "step_ms_ema": 0.0, "active": 0, "backlog": 0}

    # ---------------------------------------------------------------- public
    @classmethod
    async def create(cls, cfg: EngineConfig | None = None) -> "AsyncNemotronEngine":
        """Load the model and capture the CUDA graphs on the worker thread, then return a running engine."""
        self = cls(cfg or EngineConfig())
        self._loop = asyncio.get_running_loop()
        ready = self._loop.create_future()
        self._thread = threading.Thread(target=self._run, args=(ready,), name="nemotron-gpu", daemon=True)
        self._thread.start()
        await ready
        return self

    async def aclose(self) -> None:
        """Stop the worker; open streams end their iterators."""
        self._stop = True
        self._wake.set()
        await asyncio.to_thread(self._thread.join)

    def open_stream(self, language: str | None = None, sample_rate: int = 16000, encoding: str = "pcm16") -> AsrStream:
        if self._stop or self._error is not None:
            raise EngineClosed("engine is closed") from self._error
        if self.cfg.max_lag_ms is not None and self._lag_ms > self.cfg.max_lag_ms:
            raise EngineOverloaded(f"chunk lag {self._lag_ms:.0f} ms is above max_lag_ms={self.cfg.max_lag_ms:.0f}")
        decoder = AudioDecoder(sample_rate, encoding)  # validates the format before taking a slot
        with self._lock:
            if not self._free:
                raise EngineFull(f"all {self.cfg.max_streams} stream slots are in use")
            slot = self._free.pop()
        stream = AsrStream(self, slot, decoder)
        stream._state = _State(stream, language, StreamAudio(self.chunking))
        stream._send("open")
        return stream

    def stats(self) -> dict:
        return {**self._stats, "lag_ms": self._lag_ms, "greedy_continuations": self.stepper.continuations, "free_slots": len(self._free)}

    def _post(self, item) -> None:
        self._inbox.append(item)
        if not self._wake.is_set():  # set() takes a lock; skip it while the worker is awake anyway
            self._wake.set()

    # ---------------------------------------------------------------- worker thread
    def _run(self, ready: asyncio.Future) -> None:
        try:
            self._init_gpu()
        except BaseException as e:  # noqa: BLE001 - surface any init failure in create()
            self._error = e
            self._loop.call_soon_threadsafe(ready.set_exception, e)
            return
        self._loop.call_soon_threadsafe(ready.set_result, None)
        err, cur = None, None
        try:
            while not self._stop:
                if cur is not None:
                    while not self.stepper.ready():  # GPU busy: keep stream state current
                        self._wake.clear()
                        if self._inbox:
                            self._drain()
                            self._flush_out()  # finals of turns whose chunks are already decoded
                        else:
                            self._wake.wait(self.cfg.poll_wait_s)
                    self._finish(cur)
                    self._flush_out()
                self._wake.clear()
                self._drain()
                cur = self._prepare(self._pick())
                if cur is not None:
                    self._submit(cur)
                else:
                    self._flush_out()
                    self._wake.wait(self.cfg.idle_wait_s)
        except BaseException as e:  # noqa: BLE001 - end every stream instead of leaving it hanging
            self._error = err = e
            raise
        finally:
            failure = EngineClosed(f"engine worker failed: {err!r}") if err else None
            opening = [st for st, kind, *_ in list(self._inbox) if kind == "open"]
            for st in [*self._active.values(), *opening]:
                self._out.append((st.stream._on_released, failure))
            self._flush_out()

    def _init_gpu(self) -> None:
        cfg = self.cfg
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        self.model = NemotronModel.load(cfg.model_dir, cfg.device, cfg.dtype)
        self.tokenizer = self.model.tokenizer
        self.stepper = BatchStepper(self.model, cfg.lookahead, cfg.max_streams, cfg.max_batch, cfg.cuda_graphs, cfg.greedy_steps)
        mc = self.model.cfg
        self.chunking = Chunking(self.stepper.first_mel, self.stepper.later_mel, mc.hop_length, mc.n_fft)
        self.frame_ms = 1000.0 * mc.hop_length / mc.sample_rate
        # pinned staging (windows, slot numbers); filled only after the previous step's copies are done
        w = self.stepper.width[False]
        self._stage = (torch.empty(cfg.max_batch * w, pin_memory=True), torch.empty(cfg.max_batch, dtype=torch.long, pin_memory=True))
        self._stage_np = tuple(x.numpy() for x in self._stage)

    def _flush_out(self) -> None:
        if self._out:
            out, self._out = self._out, []
            self._loop.call_soon_threadsafe(_dispatch, out)

    def _drain(self) -> None:
        touched: dict[int, _State] = {}
        inbox = self._inbox
        while inbox:
            st, kind, t, arg = inbox.popleft()
            touched[st.stream.slot] = st
            if kind == "audio":
                st.audio.append(arg)
                st.real_end = st.audio.total
                self._mark_complete(st, t)
            elif kind == "end":
                self._end_turn(st, t, pad=arg, always=True)
            elif kind == "open":
                self.stepper.reset(st.stream.slot, self.cfg.language if st.language is None else st.language)
                self._active[st.stream.slot] = st
            else:  # close: end the open turn, if it has audio, and flush any turn still waiting for audio
                self._end_turn(st, t, pad=True, always=False)
                self._pad_to(st, st.last_boundary, t)
                st.closing = True
        for st in touched.values():
            self._settle(st)
        self._stats["active"] = len(self._active)

    def _mark_complete(self, st: _State, t: float) -> None:
        avail = self.chunking.frames_available(st.audio.total)
        while self.chunking.end(st.n_complete) <= avail:
            st.ready_t.append(t)
            st.n_complete += 1

    def _end_turn(self, st: _State, t: float, pad: bool, always: bool) -> None:
        n = max(self.chunking.chunks_covering(st.real_end), st.last_boundary)
        if n == st.last_boundary and not always:
            return  # no audio since the last turn ended
        if pad:
            self._pad_to(st, n, t)
        st.boundaries.append(n)
        st.last_boundary = n

    def _pad_to(self, st: _State, n: int, t: float) -> None:
        """Append zeros so that chunks 0..n-1 are complete."""
        if n > st.n_complete:
            st.audio.append(np.zeros(max(0, self.chunking.samples_for_chunks(n) - st.audio.total), np.float32))
            self._mark_complete(st, t)

    def _settle(self, st: _State, tokens: list[int] | None = None, t_ready: float = 0.0, t_emit: float = 0.0) -> None:
        """Queue the events due after `tokens` of the latest decoded chunk (None: no chunk was decoded):
        a final per turn boundary reached (the first one carries the tokens), else a partial if there are
        tokens; then release the slot once a closing stream has no turn left."""
        on_tokens = st.stream._on_tokens
        if tokens or (st.boundaries and st.boundaries[0] <= st.chunks_done):
            audio_ms = self.chunking.end(st.chunks_done - 1) * self.frame_ms if st.chunks_done else 0.0
        while st.boundaries and st.boundaries[0] <= st.chunks_done:
            st.boundaries.popleft()
            if tokens is None:
                t_ready = t_emit = time.perf_counter()
            self._out.append((on_tokens, "final", tokens or [], audio_ms, t_ready, t_emit))
            tokens = []
        if tokens:
            self._out.append((on_tokens, "partial", tokens, audio_ms, t_ready, t_emit))
        if st.closing and not st.boundaries and st.chunks_done == st.chunks_taken:
            del self._active[st.stream.slot]
            with self._lock:
                self._free.append(st.stream.slot)
            self._out.append((st.stream._on_released, None))

    def _pick(self) -> list[_State]:
        ready = [st for st in self._active.values() if st.n_complete > st.chunks_taken]
        self._stats["backlog"] = len(ready)
        if not ready:
            self._lag_ms = 0.0
            return []
        ready.sort(key=lambda st: st.ready_t[0])
        self._lag_ms = (time.perf_counter() - ready[0].ready_t[0]) * 1000
        first = ready[0].chunks_taken == 0
        return [st for st in ready if (st.chunks_taken == 0) == first][: self.cfg.max_batch]

    def _prepare(self, states: list[_State]) -> _Batch | None:
        """Write the batch's sample windows and slot numbers into the staging buffer."""
        if not states:
            return None
        first = states[0].chunks_taken == 0
        n = self.stepper.first_mel if first else self.stepper.later_mel
        width = self.stepper.width[first]
        xs, idx = self._stage_np
        xs = xs[: len(states) * width].reshape(len(states), width)
        for i, st in enumerate(states):
            st.audio.window(n, xs[i])
            idx[i] = st.stream.slot
            st.chunks_taken += 1
        return _Batch(states, first, [st.ready_t.popleft() for st in states], width)

    def _submit(self, bt: _Batch) -> None:
        """Queue the batch's step on the GPU; returns without waiting for it."""
        b = len(bt.states)
        xs, idx = self._stage
        bt.t_start = time.perf_counter()
        self.stepper.submit(xs[: b * bt.width].view(b, bt.width), idx[:b], bt.first)

    def _finish(self, bt: _Batch) -> None:
        new = self.stepper.collect()
        t_emit = time.perf_counter()
        for st, toks, t_ready in zip(bt.states, new, bt.t_ready):
            st.chunks_done += 1
            if self.cfg.record_chunks:
                self.chunk_log.append((t_ready, bt.t_start, t_emit, len(bt.states)))
            self._settle(st, toks, t_ready, t_emit)
        s = self._stats
        s["steps"] += 1
        s["chunks"] += len(bt.states)
        s["last_batch"] = len(bt.states)
        s["step_ms_ema"] = 0.9 * s["step_ms_ema"] + 0.1 * (t_emit - bt.t_start) * 1000


def _dispatch(out: list) -> None:
    """Event loop side: deliver one worker round of results, in order."""
    for fn, *args in out:
        fn(*args)
