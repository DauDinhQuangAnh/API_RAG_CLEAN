# RAG API

## Chay local

```bash
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
uvicorn API_RAG_NEW.main:app --host 0.0.0.0 --port 8000
```
.\python\python.exe -m uvicorn API_RAG_NEW.main:app --host 0.0.0.0 --port 8000

Base URL mac dinh: `http://127.0.0.1:8000`

## API chinh

- `GET /health`: kiem tra server.
- `GET /collections`: lay danh sach collections.
- `POST /collections`: tao collection.
- `GET /collections/{collection_name}`: xem thong tin collection.
- `PATCH /collections/{collection_name}`: sua ten hoac metadata collection.
- `DELETE /collections/{collection_name}`: xoa collection.
- `POST /ingest`: ingest file document hoac bang du lieu.
- `POST /collections/{collection_name}/query`: hoi dap RAG tren collection.

## Ingest document va bang du lieu

Flow document dung file `.docx`, `.pdf`, `.txt`, `.text`. API se doc noi dung file va chunk truc tiep toan bo text.

```bash
curl -X POST http://127.0.0.1:8000/ingest ^
  -F "file=@document.pdf" ^
  -F "collection_name=rag_collection_demo"
```

Metadata cho document chunk:

- `chunk`
- `doc_id`
- `source`
- `source_type`
- `chunk_index`

Flow bang du lieu dung file `.csv` hoac `.xlsx`. Khi upload CSV/XLSX, bat buoc truyen `index_column` de chon cot chunking:

```bash
curl -X POST http://127.0.0.1:8000/ingest ^
  -F "file=@data.csv" ^
  -F "index_column=content" ^
  -F "collection_name=rag_collection_demo"
```

```bash
curl -X POST http://127.0.0.1:8000/ingest ^
  -F "file=@data.xlsx" ^
  -F "index_column=description" ^
  -F "collection_name=rag_collection_demo"
```

Voi CSV/XLSX, API se chunk noi dung trong `index_column` va van luu cac cot con lai cua tung row vao metadata.

## Query RAG

Co the bo trong `columns_to_answer`; API se mac dinh dua `chunk`, `source`, `source_type`, `chunk_index` vao prompt neu co.

```bash
curl -X POST http://127.0.0.1:8000/collections/rag_collection_demo/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Noi dung tai lieu noi gi ve dieu kien giao hang?\", \"number_docs_retrieval\": 3}"
```

Neu muon chi dinh metadata dua vao prompt:

```bash
curl -X POST http://127.0.0.1:8000/collections/rag_collection_demo/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Tom tat tai lieu\", \"columns_to_answer\": [\"chunk\", \"source\"], \"number_docs_retrieval\": 3}"
```
