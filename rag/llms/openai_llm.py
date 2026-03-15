import os
from openai import OpenAI
from rag.base import BaseLLM

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


class OpenAILLM(BaseLLM):
    def __init__(self, embed_model: str = EMBED_MODEL, chat_model: str = CHAT_MODEL):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)
        self.embed_model = embed_model
        self.chat_model = chat_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.embed_model)
        return [item.embedding for item in response.data]

    def chat(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content