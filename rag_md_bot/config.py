import os
from dotenv import load_dotenv

load_dotenv()

# Chroma settings
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "md_rag")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")

# Gemma3 LLM 이름
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/gemma-3-1b-it")

TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))     # 문자 기준
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DATA_DIR = os.getenv("DATA_DIR", "data/docs")
PUBLIC_LLM_MODEL_NAME = os.getenv("PUBLIC_LLM_MODEL_NAME", "gpt2")
