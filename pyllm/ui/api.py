"""API client for PyLLM UI."""

import requests
import json
from typing import Generator, Dict, Any, List, Optional

from pyllm.ui.config import get_api_url


class APIClient:
    """HTTP client for PyLLM API."""

    def __init__(self):
        self.base_url = get_api_url()

    def health(self) -> Optional[Dict[str, Any]]:
        """Check API health."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def list_models(self) -> Optional[Dict[str, Any]]:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> Generator[str, None, None]:
        """Stream chat completion."""
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                stream=True,
                timeout=(30, 600),  # 30s connect, 600s read (CPU inference is slow)
            )

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            pass

        except Exception as e:
            yield f"[Error: {e}]"

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> Generator[str, None, None]:
        """Stream text generation."""
        try:
            response = requests.post(
                f"{self.base_url}/v1/generate",
                json={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "stream": True,
                },
                stream=True,
                timeout=(30, 600),  # 30s connect, 600s read (CPU inference is slow)
            )

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "text" in chunk:
                                yield chunk["text"]
                        except json.JSONDecodeError:
                            pass

        except Exception as e:
            yield f"[Error: {e}]"


_client: Optional[APIClient] = None


def get_client() -> APIClient:
    global _client
    if _client is None:
        _client = APIClient()
    return _client
