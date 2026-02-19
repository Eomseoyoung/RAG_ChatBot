import os
import glob
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any

from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBED_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR
)

@dataclass
class Chunk:
    text: str
    meta: Dict[str, Any]

def read_md_files(data_dir: str) -> List[Dict[str, Any]]:
    paths = glob.glob(os.path.join(data_dir, "**/*.md"), recursive=True)
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            docs.append({"path": p, "text": f.read()})
    return docs

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def build_chunks(docs: List[Dict[str, Any]]) -> List[Chunk]:
    out = []
    for d in docs:
        pieces = chunk_text(d["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, c in enumerate(pieces):
            out.append(
                Chunk(
                    text=c,
                    meta={
                        "source_path": d["path"],
                        "chunk_index": i,
                    },
                )
            )
    return out

def main():
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # Chroma client (persist to directory)
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    except AttributeError:
        # older chromadb versions
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.Client(ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_PERSIST_DIR,
            anonymized_telemetry=False,
        ))

    # 임베딩 차원 자동 확인
    test_vec = embedder.encode(["dim_check"], normalize_embeddings=True)
    dim = int(test_vec.shape[1])

    # 컬렉션 생성 또는 로드
    try:
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION, get_or_create=True)
    except TypeError:
        # older chromadb versions accept only name
        collection = client.get_or_create_collection(CHROMA_COLLECTION)

    docs = read_md_files(DATA_DIR)
    if not docs:
        raise RuntimeError(f"No .md files found in {DATA_DIR}")

    chunks = build_chunks(docs)
    print(f"Total chunks: {len(chunks)}")

    # 배치 임베딩 및 Chroma에 추가
    batch_size = 64
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch = chunks[i:i+batch_size]
        texts = [b.text for b in batch]
        vecs = embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
        ids = [str(uuid.uuid4()) for _ in batch]
        metadatas = [{"source_path": b.meta["source_path"], "chunk_index": b.meta["chunk_index"]} for b in batch]
        embeddings = [v.tolist() for v in vecs]
        # add to collection
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print("Done. Chunks indexed into Chroma.")

if __name__ == "__main__":
    main()
