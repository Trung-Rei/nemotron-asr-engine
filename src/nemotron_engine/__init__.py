"""Real-time multi-stream streaming ASR for nvidia/nemotron-3.5-asr-streaming-0.6b.

    from nemotron_engine import AsyncNemotronEngine, EngineConfig
"""

from typing import TYPE_CHECKING

__version__ = "0.1.0"
__all__ = ["AsrStream", "AsyncNemotronEngine", "EngineClosed", "EngineConfig", "EngineFull", "EngineOverloaded", "TranscriptEvent"]

if TYPE_CHECKING:
    from .engine import (
        AsrStream,
        AsyncNemotronEngine,
        EngineClosed,
        EngineConfig,
        EngineFull,
        EngineOverloaded,
        TranscriptEvent,
    )


def __getattr__(name: str):
    # torch is imported on first use, so the CLI's --help and the audio helpers load without it
    if name in __all__:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module 'nemotron_engine' has no attribute {name!r}")
