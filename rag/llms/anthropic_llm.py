import os
import anthropic
from rag.base import BaseLLM

CHAT_MODEL = "claude-sonnet-4-5"

# Anthropic doesn't have a native embedding API.
# We use voyage-3 via the anthropic SDK (Voyage AI is Anthropic's embedding partner).
# Install: pip install anthropic[voyage]
EMBED_MODEL = "voyage-3"


class AnthropicLLM(BaseLLM):
    """
    Chat   : Anthropic Claude (claude-sonnet-4-5 by default)
    Embed  : Voyage AI via anthropic SDK (voyage-3 by default)

    Prerequisites:
        pip install anthropic[voyage]
        export ANTHROPIC_API_KEY=sk-ant-...
    """

    def __init__(
        self,
        chat_model: str = CHAT_MODEL,
        embed_model: str = EMBED_MODEL,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

        self.chat_model = chat_model
        self.embed_model = embed_model
        self.client = anthropic.Anthropic(api_key=api_key)

        # Voyage client (bundled with anthropic[voyage])
        try:
            import voyageai

            self.voyage = voyageai.Client(api_key=api_key)
        except ImportError:
            raise ImportError("Install Voyage for embeddings: pip install anthropic[voyage]\nOr: pip install voyageai")

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self.voyage.embed(texts, model=self.embed_model)
        return result.embeddings

    def chat(self, system: str, user: str) -> str:
        message = self.client.messages.create(
            model=self.chat_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text
