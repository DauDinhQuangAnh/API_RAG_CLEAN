# RAG API

Tai lieu nay chi ghi cac thong tin can thiet de chay local. Khong commit file `.env`, API key, du lieu nguoi dung, file upload mau, log runtime, hoac noi dung collection len repo.

## Chay local

```bash
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
uvicorn API_RAG_NEW.main:app --host 0.0.0.0 --port 8000
```
ngrok http 8000

Hoac dung Python portable trong repo neu co:

```bash
.\python\python.exe -m uvicorn API_RAG_NEW.main:app --host 0.0.0.0 --port 8000
```

Base URL mac dinh: `http://127.0.0.1:8000`

## Cau hinh

Tao file `.env` tu `.env.example`, sau do dien gia tri phu hop cho moi truong local. File `.env` chua secret nen khong duoc commit.

Mot so cau hinh thuong dung:

- model LLM
- duong dan Chroma DB
- so luong chunk truy xuat
- che do chunking
- CORS origins cho local/dev

Endpoint `/runtime-config` chi tra ve cau hinh runtime khong chua secret.

## API chinh

- `GET /health`: kiem tra server.
- `GET /runtime-config`: xem cau hinh runtime an toan.
- `GET /collections`: lay danh sach collections.
- `POST /collections`: tao collection.
- `GET /collections/{collection_name}`: xem thong tin collection.
- `PATCH /collections/{collection_name}`: cap nhat thong tin collection.
- `DELETE /collections/{collection_name}`: xoa collection.
- `GET /collections/{collection_name}/records`: xem records trong collection.
- `POST /ingest`: ingest file tai lieu.
- `POST /collections/{collection_name}/query`: hoi dap RAG tren collection.

## File ho tro ingest

- PDF
- DOCX
- TXT/TEXT

Vi du ingest:

```bash
curl -X POST http://127.0.0.1:8000/ingest ^
  -F "file=@document.pdf" ^
  -F "collection_name=rag_collection_demo"
```

Vi du query:

```bash
curl -X POST http://127.0.0.1:8000/collections/rag_collection_demo/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Tom tat tai lieu\", \"number_docs_retrieval\": 3}"
```

