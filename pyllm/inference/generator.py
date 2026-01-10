"""Streaming generator utilities."""

import asyncio
from typing import Generator, AsyncGenerator, Optional
from dataclasses import dataclass

from pyllm.inference.engine import InferenceEngine, GenerationConfig, Message


@dataclass
class StreamChunk:
    """A chunk of streamed text."""
    text: str
    finished: bool = False


# Sentinel to signal end of iteration (avoids StopIteration in async context)
_DONE = object()


def _safe_next(gen):
    """
    Safely get next item from generator.

    Returns _DONE sentinel instead of raising StopIteration,
    which doesn't work well with asyncio executors in Python 3.7+.
    """
    try:
        return next(gen)
    except StopIteration:
        return _DONE


class StreamingGenerator:
    """
    Async streaming generator wrapper.

    Wraps the synchronous generator for async contexts (FastAPI, etc.)
    """

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async generator for streaming tokens."""
        loop = asyncio.get_event_loop()

        # Run sync generator in executor
        gen = self.engine.generate(prompt, config)

        try:
            while True:
                # Get next token in thread pool using safe wrapper
                token = await loop.run_in_executor(None, _safe_next, gen)
                if token is _DONE:
                    yield StreamChunk(text="", finished=True)
                    break
                yield StreamChunk(text=token, finished=False)
        except Exception as e:
            yield StreamChunk(text=f"[Error: {e}]", finished=True)

    async def chat(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async chat generator."""
        loop = asyncio.get_event_loop()
        gen = self.engine.chat(messages, config)

        try:
            while True:
                # Get next token in thread pool using safe wrapper
                token = await loop.run_in_executor(None, _safe_next, gen)
                if token is _DONE:
                    yield StreamChunk(text="", finished=True)
                    break
                yield StreamChunk(text=token, finished=False)
        except Exception as e:
            yield StreamChunk(text=f"[Error: {e}]", finished=True)
