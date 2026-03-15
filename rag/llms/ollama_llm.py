import json
import urllib.request
from rag.base import BaseLLM

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "qwen3.5:latest"  # "qwen3.5:4b"  # "mistral"
DEFAULT_EMBED_MODEL = "nomic-embed-text"  # fast local embedding model via Ollama


class OllamaLLM(BaseLLM):
    """
    Runs fully locally via Ollama.

    Prerequisites:
        brew install ollama            # or https://ollama.com
        ollama pull mistral
        ollama pull nomic-embed-text   # for embeddings
        ollama serve
    """

    def __init__(
        self,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.base_url = base_url
        self._check_server()

    def _check_server(self):
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
        except Exception:
            raise ConnectionError(f"Ollama server not reachable at {self.base_url}.\nStart it with: ollama serve")

    def _post(self, endpoint: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed using nomic-embed-text (or any Ollama embed model)."""
        embeddings = []
        for text in texts:
            result = self._post("/api/embeddings", {"model": self.embed_model, "prompt": text})
            embeddings.append(result["embedding"])
        return embeddings

    def chat(self, system: str, user: str) -> str:
        result = self._post(
            "/api/chat",
            {
                "model": self.chat_model,
                "stream": False,
                "think": False,  # disables thinking mode on qwen3, deepseek-r1, etc.
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return result["message"]["content"]
