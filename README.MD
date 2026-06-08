# RAG API

## Chay local

```bash
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
uvicorn API_RAG_NEW.main:app --host 0.0.0.0 --port 8000
```
cd "D:\AI RAG\API_RAG_NEW" ; .\.venv\Scripts\python.exe -m pytest tests
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

`/query`, `QueryRequest`, va cac field cu cua `QueryResponse` van tuong thich nguoc. Co the bo trong `columns_to_answer`; API se mac dinh dua chunk duoc truy xuat kem marker citation vao prompt.

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

## Query RAG - Phase 3 Citations

`/query` tra them field `citations`. Field nay duoc backend tao tu metadata cua cac chunk cuoi cung sau vector search, neighbor expansion, va reranking. LLM co the tra loi voi inline marker nhu `[1]`, `[2]`, nhung structured citations khong do LLM tao.

Response van giu cac field cu:

- `metadatas`
- `retrieved_data`
- `answer`
- `full_prompt`
- `citations`

`retrieved_data` hien thi cac chunk theo block co marker:

```text
[1] source=document.pdf | page=2 | chunk_index=5
chunk: Noi dung chunk...

[2] source=data.xlsx | row=4 | chunk_index=12
chunk: Noi dung chunk...
```

Moi citation gom:

- `id`
- `source`
- `source_type`
- `page_number`
- `chunk_index`
- `page_chunk_index`
- `row_index`
- `row_chunk_index`
- `doc_id`
- `snippet`

`Citation.snippet` chi la preview ngan da normalize whitespace va cat ngan, khong phai full chunk. Full chunk van co trong `retrieved_data` va `metadatas`.

Example response:

```json
{
  "answer": "Penalty Engine xu ly du lieu thieu bang cach ap du lieu mac dinh [1].",
  "citations": [
    {
      "id": 1,
      "source": "document.pdf",
      "source_type": "pdf",
      "page_number": 2,
      "chunk_index": 5,
      "page_chunk_index": 1,
      "row_index": null,
      "row_chunk_index": null,
      "doc_id": "doc_xxx",
      "snippet": "Xu ly khuyet du lieu..."
    }
  ]
}
```

## Phase 5 - Production-grade Hybrid Chunking

Phase 5 cai thien chunking cho tai lieu co cau truc nhu report, proposal, contract, standard, tai lieu ky thuat, tai lieu tai chinh, va PDF co bang. Semantic chunker hien co van duoc giu lam fallback va co the bat lai lam hanh vi chinh.

Mac dinh:

```env
RAG_CHUNKING_PROFILE=hybrid
```

Gia tri ho tro:

- `hybrid`: tach text thanh logical blocks, detect heading/section/table caption/bullet, chunk semantic ben trong section, tao PDF table row chunks khi `pdfplumber` extract duoc bang, va them parent-child metadata.
- `semantic`: dung duong ingest semantic cu gan nhu nguyen ban, khong them metadata hybrid nhu `chunk_type`, `section_path`, `table_row_index`, hay `parent_id`.

Metadata cu van duoc giu:

- `chunk`
- `doc_id`
- `source`
- `source_type`
- `chunk_index`
- PDF van giu `page_number` va `page_chunk_index`.
- CSV/XLSX van giu `row_index` va `row_chunk_index`.

Hybrid co the them metadata moi khi co:

- `chunk_type`: `section_child`, `table_row`, hoac `fallback_semantic`.
- `section_title`
- `section_path`
- `block_index`
- `parent_id`
- `table_index`
- `table_title`
- `table_row_index`

Stable IDs va Chroma `upsert` van duoc giu. Neu hybrid gap loi bat ngo, ingest fallback ve semantic va tra warning trong `IngestResponse.warnings`. Query, reranking, citations, `/ui`, va `/runtime-config` tiep tuc hoat dong voi API response cu.

Muon quay ve hanh vi chunking cu:

```env
RAG_CHUNKING_PROFILE=semantic
```

## Phase 4 - Local UI and API Polish

Phase 4 them local UI va cac endpoint debug an toan de demo/dev tren may local. Phase nay khong them production auth, OAuth, JWT, database users, Docker, hay cloud deployment.

Chay server local:

```bash
.\python\python.exe -m uvicorn API_RAG_NEW.main:app --host 0.0.0.0 --port 8000
```

Mo UI local:

- `http://127.0.0.1:8000/ui`
- Hoac mo truc tiep file `api_test_ui.html` trong browser va giu Base URL la `http://127.0.0.1:8000`.

UI ho tro health check, runtime config, list/create/delete collection, ingest file, xem records/chunks, query RAG, xem answer, citations, retrieved data, metadata, va full prompt.

Tao file `.env` tu `.env.example`:

```env
GEMINI_API_KEY=replace_with_your_key
GEMINI_MODEL=gemini-2.5-flash
CHROMA_DB_PATH=db
RAG_INITIAL_TOP_K=20
RAG_FINAL_TOP_N=6
RAG_CHUNKING_PROFILE=hybrid
RAG_INCLUDE_NEIGHBORS=true
RAG_RERANKER_TYPE=llm
RAG_CORS_ORIGINS=*
```

`GET /runtime-config` tra ve config runtime khong chua secret, gom top-k retrieval, final top-n, chunking profile, neighbor expansion, reranker type, Gemini model, embedding model dang dung, Chroma path, va CORS origins. Endpoint nay khong tra `GEMINI_API_KEY`.

## Roadmap

Phase 4 local UI/API polish da duoc implement. Production auth/deployment va UI day du hon se nam o phase rieng neu can.
