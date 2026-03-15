import os
from rag.base import BaseVectorStore

INDEX_NAME = "resume"
DIMENSION = 1536  # matches text-embedding-3-small; adjust for other models


class PineconeVectorStore(BaseVectorStore):
    def __init__(self):
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError:
            raise ImportError("Install pinecone: pip install pinecone-client")

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set.")

        self.pc = Pinecone(api_key=api_key)
        self._spec = ServerlessSpec(cloud="aws", region="us-east-1")
        self._index = None

    def _get_index(self):
        if self._index is None:
            self._index = self.pc.Index(INDEX_NAME)
        return self._index

    def collection_exists(self) -> bool:
        existing = [i.name for i in self.pc.list_indexes()]
        return INDEX_NAME in existing

    def add(self, ids, embeddings, documents, metadatas) -> None:
        if not self.collection_exists():
            from pinecone import ServerlessSpec
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=self._spec,
            )

        index = self._get_index()
        vectors = [
            {"id": id_, "values": emb, "metadata": {**meta, "text": doc}}
            for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas)
        ]
        index.upsert(vectors=vectors)

    def query(self, embedding: list[float], top_k: int = 3) -> list[str]:
        index = self._get_index()
        results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return [match["metadata"]["text"] for match in results["matches"]]