# RAG MD Bot

마크다운 문서를 기반으로 질문에 답변하는 로컬 RAG 챗봇입니다.
임베딩은 `BAAI/bge-m3`, 벡터 DB는 `Chroma`, LLM은 `Ollama`를 사용합니다.
싱글턴으로 구성

---

## 요구사항

- Python 3.8+
- [Ollama](https://ollama.ai) 설치 (LLM 실행용)

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 환경설정

`.env` 파일을 열어 설정합니다:

```env
CHROMA_PERSIST_DIR=./chroma_db
OLLAMA_MODEL=qwen2.5:3b
```

| 항목 | 설명 | 기본값 |
|------|------|--------|
| `CHROMA_PERSIST_DIR` | 벡터 DB 저장 경로 | `./chroma_db` |
| `OLLAMA_MODEL` | 사용할 Ollama 모델명 | `qwen2.5:7b` |

---

## 실행 순서

### 1. Ollama 모델 다운로드 (최초 1회)

```bash
ollama pull qwen2.5:3b
```

> 한국어 지원 모델 추천: `qwen2.5:3b` (2GB), `qwen2.5:7b` (4.7GB)

### 2. 문서 인덱싱

`data/docs/` 폴더에 `.md` 파일을 넣은 후 실행:

```bash
python ingest.py
```

### 3. 챗봇 실행

```bash
python app.py
```

`exit` 입력 시 종료됩니다.

---

## 파일 구조

```
rag_md_bot/
├── app.py          # 챗봇 메인 (질문 → 검색 → 답변)
├── ingest.py       # 문서 인덱싱 (md 파일 → Chroma)
├── config.py       # 환경변수 설정
├── .env            # API 키 및 경로 설정
├── requirements.txt
├── data/
│   └── docs/       # md 문서 파일 위치
└── chroma_db/      # 벡터 DB 저장 폴더 (자동 생성)
```

---

## 모델 변경

`.env`에서 `OLLAMA_MODEL` 값을 바꾸면 됩니다:

```env
OLLAMA_MODEL=llama3.2:3b
```

변경 후 해당 모델을 먼저 pull 해야 합니다:

```bash
ollama pull llama3.2:3b
```

