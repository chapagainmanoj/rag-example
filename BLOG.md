# Building a RAG Pipeline from Scratch

**Retrieval-Augmented Generation (RAG)** is the technique behind most production AI assistants today. Instead of asking an LLM to recall facts from training, you give it relevant context at query time. The result is more accurate, grounded, and up-to-date answers.

This article walks through building one from scratch - every layer, every decision.

---

## Why RAG?

LLMs have two fundamental limitations:

- **Knowledge cutoff** - they don't know what happened after training
- **Context size** - you can't stuff an entire document library into a prompt

RAG solves both. You store your documents externally, retrieve only the relevant pieces per query, and pass those pieces as context to the LLM.

```
User question
     ↓
Embed question → vector
     ↓
Search vector store → top-k similar chunks
     ↓
LLM(system_prompt + chunks + question)
     ↓
Grounded answer
```

---

## Step 1: Chunking

Before you can embed anything, you need to split your documents into chunks. This is more important than most tutorials admit.

**Why not embed the whole document?**

- Embedding models have token limits (typically 512-8192 tokens)
- Large chunks dilute signal - a 10-page doc embedded as one vector loses specificity
- Retrieval precision drops - you want to fetch a paragraph, not a chapter

**Chunking strategies:**

| Strategy | Best for |
|---|---|
| Fixed size (e.g. 300 chars) | Simple, works everywhere |
| By sentence | Better semantic coherence |
| By section/heading | Structured docs like resumes, reports |
| Recursive | Long-form prose with nested structure |

For a resume, splitting by section headings (Work Experience, Skills, Education) gives the cleanest retrieval - each chunk represents a discrete topic.

```python
section_pattern = re.compile(
    r"\n(?=Work Experience|Education|Skills|Publications)",
    re.IGNORECASE
)
sections = section_pattern.split(text)
```

---

## Step 2: Embeddings

An embedding converts text into a dense float vector. Similar text produces similar vectors - that's what makes semantic search possible.

```
"Python developer" → [0.12, -0.34, 0.88, ...]  (1536 dimensions)
"skilled in Python" → [0.11, -0.31, 0.85, ...]  (very close)
"banana bread recipe" → [-0.72, 0.14, -0.22, ...] (far away)
```

**Model options:**

| Model | Provider | Notes |
|---|---|---|
| `text-embedding-3-small` | OpenAI | Best balance of cost/quality |
| `text-embedding-3-large` | OpenAI | Higher accuracy, 3x cost |
| `nomic-embed-text` | Ollama (local) | Free, runs on your machine |
| `embed-english-v3.0` | Cohere | Strong alternative to OpenAI |

**Important:** Use the same model for indexing and querying. Mixing models breaks similarity search.

```python
def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]
```

---

## Step 3: Vector Store

A vector store indexes your embeddings and lets you search by similarity (cosine distance or dot product) rather than exact match.

**Options by use case:**

| Store | Type | Best for |
|---|---|---|
| ChromaDB | Local/embedded | Dev, small datasets, no infra |
| Pinecone | Managed cloud | Production, scale, no ops |
| Qdrant | Self-hosted | Production + full control |
| FAISS | In-memory | Batch research, no persistence |
| pgvector | Postgres extension | Already using Postgres |

For development, ChromaDB with persistence is ideal:

```python
import chromadb

client = chromadb.PersistentClient(path=".chroma_db")
collection = client.get_or_create_collection("my_docs")

collection.add(
    ids=["chunk_0", "chunk_1"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["text of chunk 0", "text of chunk 1"],
    metadatas=[{"section": "Skills"}, {"section": "Experience"}]
)
```

Persistent storage means you only embed once - subsequent runs skip indexing entirely.

---

## Step 4: Retrieval

At query time, embed the question and find the most similar chunks:

```python
query_vector = embed(["What languages does Manoj know?"])[0]

results = collection.query(
    query_embeddings=[query_vector],
    n_results=5   # top_k - tune this to your chunk count
)

context = "\n---\n".join(results["documents"][0])
```

**Tuning top_k:**

- Too low: miss relevant chunks (especially when one topic is split across multiple chunks)
- Too high: flood the LLM with noise, degrade answer quality
- A good default: match or slightly exceed the number of chunks for the most common section

---

## Step 5: Generation

Pass the retrieved context to the LLM with a tight system prompt:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "Answer using ONLY the provided context. "
                "Be concise. Do not explain your reasoning."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
)
```

**System prompt tips:**

- "Answer using ONLY the provided context" - prevents hallucination
- "Be concise. Do not explain your reasoning." - critical for reasoning models (Qwen3, DeepSeek-R1) which tend to over-explain
- For Ollama/Qwen3 specifically, also pass `"think": False` to disable chain-of-thought mode

---

## Making It Modular

A production RAG pipeline should decouple the LLM backend from the vector store backend. Define abstract base classes:

```python
class BaseLLM(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def chat(self, system: str, user: str) -> str: ...

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, ids, embeddings, documents, metadatas) -> None: ...

    @abstractmethod
    def query(self, embedding, top_k) -> list[str]: ...
```

Now your pipeline only depends on these interfaces. Swapping OpenAI for Ollama, or ChromaDB for Pinecone, requires zero changes to pipeline logic.

---

## Common Pitfalls

**1. Re-embedding on every run**
Use persistent storage and check if the collection exists before indexing. Embeddings are expensive and slow.

**2. top_k too low**
If one section (e.g. Work Experience) splits into 6 chunks and you only fetch 3, you'll get incomplete answers. Set top_k to cover your densest section.

**3. Mixing embedding models**
Index with `text-embedding-3-small`, query with `text-embedding-3-large` = broken results. Always use the same model end-to-end.

**4. Reasoning models in verbose mode**
Qwen3, DeepSeek-R1, and similar models think out loud by default. Pass `"think": False` in the Ollama request or use `/no_think` system prompt prefix.

**5. Chunk size mismatch**
Chunks too small = lose context. Chunks too large = lose precision. For most document types, 200-400 tokens per chunk is the sweet spot.

---

## Full Stack Summary

```
PDF / Docs
    ↓  loader.py
Chunks (text + metadata)
    ↓  llm.embed()
Vectors (float arrays)
    ↓  vectorstore.add()
ChromaDB / Pinecone
    ↓  vectorstore.query()
Top-K Chunks
    ↓  llm.chat()
Grounded Answer
```

RAG is not magic - it's a retrieval system feeding a generation system. Get the retrieval right (chunking, top_k, embeddings) and the generation almost takes care of itself.