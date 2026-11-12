"""WebSocket server for AsyncNemotronEngine: one process, one GPU, one engine.

    pip install "nemotron-asr-engine[server]"
    nemotron-asr-engine serve --port 8090
    # or: uvicorn nemotron_engine.server:app --host 0.0.0.0 --port 8090   (one worker: the engine owns the GPU)

Settings come from the environment (all optional):
    NEMOTRON_MODEL_DIR          local checkpoint dir (default: download from the HF hub)
    NEMOTRON_LANGUAGE           default language prompt: auto (default), en-US, vi-VN, ...
    NEMOTRON_MAX_STREAMS        concurrent streams (slots), default 512
    NEMOTRON_MAX_BATCH          default 64
    NEMOTRON_MAX_LAG_MS         refuse new streams above this chunk lag, default 200; "none" disables
    NEMOTRON_MATMUL_PRECISION   "high" (TF32, default) or "highest"
    NEMOTRON_CUDA_GRAPHS        1 (default) or 0: run each step eagerly (slower; for GPUs where graphs misbehave)

WS /v1/stream?language=en-US&sample_rate=8000&encoding=mulaw
    client -> server  binary frame: audio in the stream's encoding (pcm16 | mulaw | f32), any size
                      text frame:   {"type": "end_turn", "pad": true}   the client's endpointer ended the turn
                                    {"type": "close"}                   finish the open turn and close
    server -> client  {"type": "partial" | "final", "turn", "text", "delta", "audio_ms", "latency_ms"}
                      {"type": "closed"} after the last event; then the server closes the socket
    Socket close codes: 1013 engine full or overloaded (try again later), 1003 bad query parameters,
    1011 engine failure. A client that disconnects without "close" gets its open turn finished and dropped.

GET /  -> a small demo page: microphone in the browser, live partials and finals (browsers allow the microphone
    only on https or localhost; for a remote server, use an SSH tunnel: ssh -L 8090:localhost:8090 host)

GET /health -> engine stats (active streams, free slots, chunk lag, ...) and memory: GPU memory held by PyTorch
    (allocated / reserved / peak MB) and the process RSS (MB), to watch for growth under load
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .engine import AsyncNemotronEngine, EngineClosed, EngineConfig, EngineFull


def config_from_env() -> EngineConfig:
    env, cfg = os.environ.get, EngineConfig()
    lag = env("NEMOTRON_MAX_LAG_MS")
    return EngineConfig(
        model_dir=env("NEMOTRON_MODEL_DIR") or None,
        language=env("NEMOTRON_LANGUAGE", cfg.language),
        max_streams=int(env("NEMOTRON_MAX_STREAMS", cfg.max_streams)),
        max_batch=int(env("NEMOTRON_MAX_BATCH", cfg.max_batch)),
        max_lag_ms=cfg.max_lag_ms if lag is None else (None if lag.lower() == "none" else float(lag)),
        matmul_precision=env("NEMOTRON_MATMUL_PRECISION", cfg.matmul_precision),
        cuda_graphs=env("NEMOTRON_CUDA_GRAPHS", "1") != "0",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = await AsyncNemotronEngine.create(config_from_env())
    try:
        yield
    finally:
        await app.state.engine.aclose()


app = FastAPI(title="nemotron-asr-engine", lifespan=lifespan)


def _rss_mb() -> float:
    with open("/proc/self/status") as f:
        return next(int(l.split()[1]) for l in f if l.startswith("VmRSS:")) / 1024


@app.get("/", response_class=HTMLResponse)
async def demo() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health() -> dict:
    mb = 1 / 2**20
    return {**app.state.engine.stats(),
            "gpu_allocated_mb": torch.cuda.memory_allocated() * mb, "gpu_reserved_mb": torch.cuda.memory_reserved() * mb,
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() * mb, "rss_mb": _rss_mb()}


@app.websocket("/v1/stream")
async def stream(ws: WebSocket, language: str | None = None, sample_rate: int = 16000, encoding: str = "pcm16"):
    await ws.accept()
    try:
        st = app.state.engine.open_stream(language=language, sample_rate=sample_rate, encoding=encoding)
    except EngineFull as e:  # also EngineOverloaded
        await ws.close(code=1013, reason=str(e)[:120])
        return
    except EngineClosed as e:
        await ws.close(code=1011, reason=str(e)[:120])
        return
    except ValueError as e:
        await ws.close(code=1003, reason=str(e)[:120])
        return

    connected, close_code = True, 1000

    async def send_events() -> None:
        nonlocal connected, close_code
        try:
            async for ev in st:
                if connected:
                    await ws.send_text(json.dumps({
                        "type": ev.kind, "turn": ev.turn, "text": ev.text, "delta": ev.delta,
                        "audio_ms": round(ev.audio_ms, 1), "latency_ms": round((ev.t_emit - ev.t_ready) * 1000, 1),
                    }, ensure_ascii=False))
        except (WebSocketDisconnect, RuntimeError):  # client gone; the engine releases the stream on its own
            connected = False
        except EngineClosed:
            close_code = 1011

    sender = asyncio.create_task(send_events())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                connected = False
                break
            if msg.get("bytes") is not None:
                st.push_nowait(msg["bytes"])
                continue
            cmd = json.loads(msg.get("text") or "{}")
            kind = cmd.get("type") if isinstance(cmd, dict) else None
            if kind == "end_turn":
                st.end_turn(pad=bool(cmd.get("pad", True)))
            elif kind == "close":
                break
            else:
                close_code = 1003
                break
    except WebSocketDisconnect:
        connected = False
    except (ValueError, EngineClosed):  # bad JSON or audio payload; engine stopped
        close_code = 1003 if close_code == 1000 else close_code
    finally:
        await st.close()  # final of the open turn, then the slot is released and the event iterator ends
        await sender
        if connected:
            try:
                if close_code == 1000:
                    await ws.send_text(json.dumps({"type": "closed"}))
                await ws.close(code=close_code)
            except (WebSocketDisconnect, RuntimeError):
                pass  # the client closed first
