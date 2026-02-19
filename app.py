import os
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBED_MODEL_NAME, TOP_K
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
SYSTEM_PROMPT = "당신은 제공된 문서(md) 컨텍스트만 근거로 답하는 도우미입니다. 모르면 모른다고 말하고, 근거가 되는 문서 부분을 함께 제시하세요."

def build_prompt(query: str, contexts: list[dict]) -> str:
    ctx_block = "\n\n".join(
        [f"[출처: {c['source_path']} | chunk:{c['chunk_index']}]\n{c['text']}" for c in contexts]
    )
    return f"""[질문]
{query}

[컨텍스트]
{ctx_block}"""

def retrieve(collection, embedder: SentenceTransformer, query: str, top_k: int):
    qvec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    res = collection.query(query_embeddings=[qvec], n_results=top_k, include=["metadatas", "documents", "distances"])
    contexts = []
    if res and "documents" in res and len(res["documents"]) > 0:
        docs = res["documents"][0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, m, dist in zip(docs, metas, dists):
            contexts.append({
                "text": doc or "",
                "source_path": m.get("source_path", "unknown") if isinstance(m, dict) else "unknown",
                "chunk_index": m.get("chunk_index", -1) if isinstance(m, dict) else -1,
                "score": float(dist) if dist is not None else 0.0,
            })
    return contexts

def generate(prompt: str) -> str:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    return response["message"]["content"].strip()

def main():
    print(f"Loading embedder: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Connecting Chroma: {CHROMA_PERSIST_DIR}")
    try:
        chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    except AttributeError:
        from chromadb.config import Settings as ChromaSettings
        chroma = chromadb.Client(ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_PERSIST_DIR,
            anonymized_telemetry=False,
        ))
    try:
        collection = chroma.get_collection(CHROMA_COLLECTION)
    except Exception:
        collection = chroma.get_or_create_collection(CHROMA_COLLECTION)

    print(f"Using Ollama model: {OLLAMA_MODEL}")

    print("\n=== MD RAG Chatbot (type 'exit' to quit) ===")
    while True:
        q = input("\nQ> ").strip()
        if not q or q.lower() == "exit":
            break

        contexts = retrieve(collection, embedder, q, TOP_K)
        prompt = build_prompt(q, contexts)

        ans = generate(prompt)

        print("\n--- Answer ---")
        print(ans)
        print("\n--- Retrieved Contexts ---")
        for i, c in enumerate(contexts, 1):
            print(f"{i}. score={c['score']:.4f} | {c['source_path']} | chunk={c['chunk_index']}")

if __name__ == "__main__":
    main()
