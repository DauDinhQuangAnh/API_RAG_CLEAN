# RAG API

Tai lieu nay chi ghi cac thong tin can thiet de chay va test local. Khong commit file `.env`, API key, du lieu nguoi dung, file upload mau, log runtime, hoac noi dung collection len repo.

## Chay local

```bash
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
uvicorn API_RAG_NEW.main:app --host 0.0.0.0 --port 8000
```

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
- `POST /ingest`: ingest file tai lieu hoac bang du lieu.
- `POST /collections/{collection_name}/query`: hoi dap RAG tren collection.

## Phase 6 search quality

He thong van chi co mot luong query chinh: `POST /collections/{collection_name}/query`.
Phase 6 khong them search mode cho user, khong them tham so request, khong BM25,
khong hybrid keyword search, va khong graph search.

Pipeline query van dua tren vector search, sau do mo rong context co kiem soat bang
metadata Phase 5/5.1 nhu `parent_id`, `section_path`, `table_index`, `page_number`,
va reranker nhan metadata gon hon de uu tien evidence dung nguon, section, va table
row khi cau hoi lien quan den bang, so lieu, chi phi, gia, doanh thu, hoac so sanh.

Distance guard la tuy chon va mac dinh tat:

- `RAG_ENABLE_DISTANCE_GUARD=false`
- `RAG_MAX_DISTANCE` chi nen khai bao khi muon bat nguong distance cu the cho moi truong.

## File ho tro ingest

- PDF
- DOCX
- TXT/TEXT
- CSV
- XLSX

CSV/XLSX can truyen `index_column` de chon cot dung cho chunking.

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

## UI local

Mo UI:

- `http://127.0.0.1:8000/ui`
- Hoac mo truc tiep file `api_test_ui.html` va giu Base URL la `http://127.0.0.1:8000`

UI ho tro health check, runtime config, quan ly collection, ingest file, xem records/chunks, va query RAG.

## Test

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

Hoac neu dung Python portable:

```bash
.\python\python.exe -m pytest -q
```
