import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR

def main():
    print(f"Checking Chroma (persist={CHROMA_PERSIST_DIR})...")
    try:
        client = chromadb.Client()
        # list collections
        cols = client.list_collections()
        names = [c["name"] if isinstance(c, dict) else str(c) for c in cols]
        print("Connected to Chroma.")
        print("Collections:", names)
    except Exception as e:
        print("Failed to connect to Chroma:", e)
        raise SystemExit(1)

if __name__ == '__main__':
    main()
