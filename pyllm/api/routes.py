"""FastAPI routes for LLM inference with streaming."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pyllm.core.config import Config
from pyllm.inference.engine import InferenceEngine, GenerationConfig, Message
from pyllm.inference.generator import StreamingGenerator
from pyllm.inference.templates import ChatTemplate, TemplateType

logger = logging.getLogger("pyllm.api")


# Request/Response models

class GenerateRequest(BaseModel):
    """Text generation request."""
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    max_new_tokens: int = 256
    stream: bool = True


class ChatMessage(BaseModel):
    """Chat message."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat completion request (OpenAI-compatible)."""
    model: str = "pyllm"
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = True


class ChatChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    """Chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create FastAPI application."""
    config = config or Config.load()

    app = FastAPI(
        title="PyLLM API",
        description="LLM Inference API with streaming support",
        version="1.8.3",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize engine (lazy loading)
    engine: Optional[InferenceEngine] = None
    generator: Optional[StreamingGenerator] = None
    chat_template = ChatTemplate(TemplateType.SIMPLE)

    def get_engine() -> InferenceEngine:
        nonlocal engine, generator
        if engine is None:
            engine = InferenceEngine(config.model)
            if config.model.path:
                engine.load(config.model.path)
            generator = StreamingGenerator(engine)
        return engine

    def get_generator() -> StreamingGenerator:
        get_engine()
        return generator

    # Health check
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model_loaded": engine is not None and engine.is_loaded,
            "timestamp": datetime.now().isoformat(),
        }

    # Model info
    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model.name,
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "pyllm",
                }
            ]
        }

    # Load model
    @app.post("/v1/models/load")
    async def load_model(model_path: str):
        """Load a model."""
        try:
            nonlocal engine, generator
            engine = InferenceEngine(config.model)
            engine.load(model_path)
            generator = StreamingGenerator(engine)
            return {"status": "loaded", "path": model_path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Text generation
    @app.post("/v1/generate")
    async def generate(request: GenerateRequest):
        """Generate text from prompt."""
        try:
            eng = get_engine()

            if not eng.is_loaded:
                raise HTTPException(status_code=400, detail="Model not loaded")

            gen_config = GenerationConfig(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                max_new_tokens=request.max_new_tokens,
            )

            if request.stream:
                return StreamingResponse(
                    stream_generate(get_generator(), request.prompt, gen_config),
                    media_type="text/event-stream",
                )
            else:
                result = eng.complete(request.prompt, gen_config)
                return {"text": result, "prompt": request.prompt}

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Chat completion (OpenAI-compatible)
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        """Chat completion endpoint (OpenAI-compatible)."""
        try:
            eng = get_engine()

            if not eng.is_loaded:
                raise HTTPException(status_code=400, detail="Model not loaded")

            # Convert messages
            messages = [
                Message(role=m.role, content=m.content)
                for m in request.messages
            ]

            gen_config = GenerationConfig(
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=request.max_tokens,
            )

            if request.stream:
                return StreamingResponse(
                    stream_chat(get_generator(), messages, gen_config, request.model),
                    media_type="text/event-stream",
                )
            else:
                # Non-streaming response
                prompt = chat_template.format(messages)
                result = eng.complete(prompt, gen_config)

                return ChatResponse(
                    id=f"chatcmpl-{int(datetime.now().timestamp())}",
                    created=int(datetime.now().timestamp()),
                    model=request.model,
                    choices=[
                        ChatChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=result),
                        )
                    ]
                )

        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


async def stream_generate(generator: StreamingGenerator, prompt: str, config: GenerationConfig):
    """Stream generation as SSE events."""
    try:
        async for chunk in generator.generate(prompt, config):
            if chunk.finished:
                yield f"data: [DONE]\n\n"
            else:
                data = json.dumps({"text": chunk.text})
                yield f"data: {data}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def stream_chat(
    generator: StreamingGenerator,
    messages: List[Message],
    config: GenerationConfig,
    model: str,
):
    """Stream chat completion as SSE events (OpenAI-compatible format)."""
    try:
        chat_id = f"chatcmpl-{int(datetime.now().timestamp())}"

        async for chunk in generator.chat(messages, config):
            if chunk.finished:
                # Final chunk
                data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield f"data: [DONE]\n\n"
            else:
                # Token chunk
                data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk.text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"

    except Exception as e:
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
