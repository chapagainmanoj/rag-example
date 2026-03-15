from rag.base import BaseVectorStore, BaseLLM

SYSTEM_PROMPT = (
    "You are a resume assistant. Answer using ONLY the provided context. "
    "Be concise. Do not explain your reasoning or repeat yourself."
)


class RAGPipeline:
    def __init__(self, vectorstore: BaseVectorStore, llm: BaseLLM):
        self.vectorstore = vectorstore
        self.llm = llm

    def index(self, chunks: list[dict]) -> None:
        if self.vectorstore.collection_exists():
            print("Collection already exists, skipping indexing.")
            return

        print("Embedding and indexing chunks...")
        texts = [c["content"] for c in chunks]
        embeddings = self.llm.embed(texts)

        self.vectorstore.add(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"section": c["section"]} for c in chunks],
        )
        print(f"Indexed {len(chunks)} chunks.\n")

    def query(self, question: str, top_k: int = 6) -> str:
        q_embedding = self.llm.embed([question])[0]
        docs = self.vectorstore.query(q_embedding, top_k=top_k)
        context = "\n\n---\n\n".join(docs)
        return self.llm.chat(system=SYSTEM_PROMPT, user=f"Context:\n{context}\n\nQuestion: {question}")
