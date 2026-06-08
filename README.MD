# RAG API

## Chay local

```bash
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
uvicorn API_RAG_NEW.main:app --host 0.0.0.0 --port 8000
```

Hoac dung Python portable trong repo:

```bash
.\python\python.exe -m uvicorn API_RAG_NEW.main:app --host 0.0.0.0 --port 8000
```

Base URL mac dinh: `http://127.0.0.1:8000`

## API chinh

- `GET /health`: kiem tra server.
- `GET /collections`: lay danh sach collections.
- `POST /collections`: tao collection.
- `GET /collections/{collection_name}`: xem thong tin collection.
- `PATCH /collections/{collection_name}`: sua ten hoac metadata collection.
- `DELETE /collections/{collection_name}`: xoa collection.
- `GET /collections/{collection_name}/records`: debug records, tra ve `ids`, `metadatas`, va `documents`.
- `POST /ingest`: ingest file document hoac bang du lieu.
- `POST /collections/{collection_name}/query`: hoi dap RAG tren collection.

## Ingest Phase 1

Phase 1 ho tro cac loai file:

- PDF
- DOCX
- TXT/TEXT
- CSV
- XLSX

Khi ingest, chunk text duoc luu vao Chroma `documents`. Metadata van giu truong `chunk` de tuong thich nguoc voi query/prompt hien tai. Ghi lai cung file vao cung collection dung stable record IDs va Chroma `upsert`, nen record cu duoc cap nhat thay vi tao ban sao trung lap.

## Ingest PDF

PDF duoc doc theo tung trang bang `_extract_pdf_pages()`, sau do chunk theo trang. Metadata cua moi chunk PDF gom:

- `chunk`
- `doc_id`
- `source`
- `source_type`
- `chunk_index`: 1-based, tang toan cuc theo tat ca chunk cua file.
- `page_number`: 1-based.
- `page_chunk_index`: 1-based trong tung trang.

```bash
curl -X POST http://127.0.0.1:8000/ingest ^
  -F "file=@document.pdf" ^
  -F "collection_name=rag_collection_demo"
```

## Ingest DOCX/TXT

DOCX, TXT, va TEXT duoc doc thanh text, sau do chunk truc tiep. Metadata cua moi chunk gom:

- `chunk`
- `doc_id`
- `source`
- `source_type`
- `chunk_index`: 1-based theo file.

```bash
curl -X POST http://127.0.0.1:8000/ingest ^
  -F "file=@document.txt" ^
  -F "collection_name=rag_collection_demo"
```

## Ingest CSV/XLSX

CSV/XLSX bat buoc truyen `index_column` de chon cot chunking. Moi row co text hop le trong `index_column` duoc chunk rieng.

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

Metadata cua moi chunk CSV/XLSX gom:

- `chunk`
- `doc_id`: stable theo file hash va `row_index`.
- `source`
- `source_type`
- `chunk_index`: 1-based, tang toan cuc theo tat ca chunk tabular.
- `row_index`: 1-based theo thu tu row sau khi reset index.
- `row_chunk_index`: 1-based trong tung row.
- Cac field cua row sau khi sanitize, tru cac key noi bo nhu `id`, `_id`, `chunk`, `doc_id`, `source`, `source_type`, `chunk_index`, `row_index`, `row_chunk_index`, `page_number`, `page_chunk_index`.

## Debug records

Dung `/collections/{collection_name}/records` de xem metadata va document text da luu trong Chroma:

```bash
curl "http://127.0.0.1:8000/collections/rag_collection_demo/records?limit=10"
```

Response gom cac truong hien co va them `documents`:

- `ids`
- `metadatas`
- `documents`

## Query RAG - Phase 2 Retrieval

`/query`, `QueryRequest`, va `QueryResponse` van tuong thich nguoc. API khong them field request/response moi trong Phase 2. Co the bo trong `columns_to_answer`; API se mac dinh dua `chunk`, `source`, `source_type`, `chunk_index` vao prompt neu co.

Phase 2 retrieval hien dang dung:

- Initial vector retrieval voi `RAG_INITIAL_TOP_K` de lay candidate set lon hon.
- Optional neighbor chunk expansion voi `RAG_INCLUDE_NEIGHBORS`.
- Gemini LLM reranking khi `RAG_RERANKER_TYPE=llm`.
- Fallback ve vector order neu reranker loi, tra JSON sai format, ID khong hop le, hoac tra qua it chunk.
- Final selected chunks dua vao prompt duoc dieu khien boi:
  - `RAG_FINAL_TOP_N` neu request khong gui `number_docs_retrieval`.
  - `number_docs_retrieval` neu request gui explicit field nay.

Env mac dinh cho Phase 2:

```env
RAG_INITIAL_TOP_K=20
RAG_FINAL_TOP_N=6
RAG_INCLUDE_NEIGHBORS=true
RAG_RERANKER_TYPE=llm
```

Chua implement citations trong `/query`. Citations la Phase 3.

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

## Roadmap

Phase 2 retrieval/reranking da duoc implement. Phase 3 citations dang duoc len ke hoach, chua duoc implement trong code hien tai.
