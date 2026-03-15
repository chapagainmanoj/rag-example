# Code Walkthrough: End-to-End Pipeline

Here's how all the pieces connect when you run `python main.py --llm ollama`:

---

## 1. `main.py` - Wires everything together

Parses CLI args (`--llm`, `--vectorstore`), instantiates the right backends, and calls the pipeline. Nothing smart here - just a composition root.

```
--llm ollama           → OllamaLLM()
--vectorstore chromadb → ChromaVectorStore()
                ↓
   RAGPipeline(vectorstore, llm)
```

---

## 2. `loader.py` - PDF → Chunks

```
PdfReader → raw text → split by section headings → sub-chunk if >600 chars
```

Each chunk is a dict: `{ id, section, content }`. The section label (`Work Experience`, `Skills`, etc.) is stored as metadata in the vector store for filtering later.

---

## 3. `pipeline.index()` - Embed + Store

```
chunks → llm.embed(texts) → list of float vectors
                                    ↓
                       vectorstore.add(ids, embeddings, docs, metadata)
```

Each chunk of text becomes a ~1536-dimensional vector. Similar text lands close together in that space.

---

## 4. `pipeline.query()` - Retrieve + Generate

```
question → llm.embed([question]) → query vector
                                        ↓
                          vectorstore.query(vector, top_k=6)
                          → returns 6 most similar text chunks
                                        ↓
                    llm.chat(system_prompt, context + question)
                                        ↓
                                    answer
```

The LLM never sees the whole resume - only the most relevant chunks. That's the core idea of RAG.

---

## 5. `base.py` - Why the ABCs matter

`BaseLLM` and `BaseVectorStore` define contracts with just 2-3 methods each. The pipeline only talks to these interfaces - it has no idea if it's talking to OpenAI, Anthropic, or Ollama, or whether the store is ChromaDB or Pinecone. Swapping backends = zero changes to pipeline logic.

---

## Full Flow Summary

```
PDF
 ↓  loader.py        → chunks (text + metadata)
 ↓  llm.embed()      → vectors (float arrays)
 ↓  vectorstore.add()→ indexed in ChromaDB / Pinecone
 ↓  vectorstore.query() → top-k relevant chunks
 ↓  llm.chat()       → grounded answer
```

| Step | File | Swappable? |
|---|---|---|
| Load + chunk | `loader.py` | No (shared) |
| Embed | `llms/*.py` | Yes - OpenAI / Anthropic / Ollama |
| Store + retrieve | `vectorstores/*.py` | Yes - ChromaDB / Pinecone |
| Orchestrate | `pipeline.py` | No (shared) |
| Chat | `llms/*.py` | Yes - same as embed |