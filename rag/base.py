from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """All vector store backends must implement this interface."""

    @abstractmethod
    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        ...

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 3) -> list[str]:
        """Returns top_k matching document strings."""
        ...

    @abstractmethod
    def collection_exists(self) -> bool:
        ...


class BaseLLM(ABC):
    """All LLM backends must implement this interface."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def chat(self, system: str, user: str) -> str:
        ...