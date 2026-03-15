import os
import argparse

from dotenv import load_dotenv

load_dotenv()


from rag.pipeline import RAGPipeline
from rag.vectorstores.chroma import ChromaVectorStore
from rag.vectorstores.pinecone import PineconeVectorStore
from rag.llms.openai_llm import OpenAILLM
from rag.llms.ollama_llm import OllamaLLM
from rag.llms.anthropic_llm import AnthropicLLM
from rag.loader import load_and_chunk

PDF_PATH = os.environ.get("PDF_PATH")

VECTORSTORE_MAP = {
    "chromadb": ChromaVectorStore,
    "pinecone": PineconeVectorStore,
}

LLM_MAP = {
    "openai": OpenAILLM,
    "ollama": OllamaLLM,
    "anthropic": AnthropicLLM,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Resume RAG Pipeline")
    parser.add_argument("--vectorstore", choices=VECTORSTORE_MAP.keys(), default="chromadb")
    parser.add_argument("--llm", choices=LLM_MAP.keys(), default="openai")
    parser.add_argument("--pdf", default=PDF_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Using vectorstore: {args.vectorstore} | llm: {args.llm}\n")

    # Init components
    vectorstore = VECTORSTORE_MAP[args.vectorstore]()
    llm = LLM_MAP[args.llm]()
    pipeline = RAGPipeline(vectorstore=vectorstore, llm=llm)

    # Load and index
    print(f"Loading {args.pdf}...")
    chunks = load_and_chunk(args.pdf)
    pipeline.index(chunks)

    # Example queries
    questions = [
        "What programming languages does Manoj know?",
        "How many years of experience does he have?",
        "Has he worked with Kubernetes?",
        "What companies has he worked at?",
    ]

    print("\n── RAG Query Results ─────────────────────────────────")
    for q in questions:
        print(f"\nQ: {q}")
        answer = pipeline.query(q)
        print(f"A: {answer}")


if __name__ == "__main__":
    main()
