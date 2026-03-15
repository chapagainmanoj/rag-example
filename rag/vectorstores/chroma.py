import chromadb
from rag.base import BaseVectorStore

COLLECTION_NAME = "resume"


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, persist_path: str = ".chroma_db"):
        """
        Uses persistent ChromaDB so re-runs skip re-indexing.
        Pass persist_path=None for in-memory only.
        """
        if persist_path:
            self.client = chromadb.PersistentClient(path=persist_path)
        else:
            self.client = chromadb.Client()
        self._exists = False

    def collection_exists(self) -> bool:
        existing = [c.name for c in self.client.list_collections()]
        self._exists = COLLECTION_NAME in existing
        return self._exists

    def add(self, ids, embeddings, documents, metadatas) -> None:
        collection = self.client.get_or_create_collection(COLLECTION_NAME)
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, embedding: list[float], top_k: int = 3) -> list[str]:
        collection = self.client.get_or_create_collection(COLLECTION_NAME)
        results = collection.query(query_embeddings=[embedding], n_results=top_k)
        return results["documents"][0]