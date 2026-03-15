### Installation
Resume RAG Pipeline - Modular Entry Point

```bash
pip install -r requirements.txt
```

### Usage

```bash
# Default: OpenAI + ChromaDB
python main.py

# Pinecone (needs PINECONE_API_KEY)
python main.py --vectorstore pinecone

# Fully local via Ollama
ollama pull mistral && ollama pull nomic-embed-text
python main.py --llm ollama

# Mix and match
python main.py --llm ollama --vectorstore pinecone
```
