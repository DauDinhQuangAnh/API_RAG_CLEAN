# Tạo venv & chạy API
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
uvicorn API_RAG_NEW.main:app --host 0.0.0.0 --port 8000
ngrok http 8000 --authtoken 3ATrkoV53hkN7BiSFRulmhXogXx_4i5pFjfLNexGcA7qrD6mU
## Các API chính (base URL: http://127.0.0.1:8000)

- **Health**: `GET /health` → trả `{ "status": "ok" }`
- **List collections**: `GET /collections`
- **Tạo collection**: `POST /collections`
  - Body: `{ "name": "rag_collection_demo", "description": "Demo" }`
- **Xem 1 collection**: `GET /collections/{collection_name}`
- **Sửa collection**: `PATCH /collections/{collection_name}`
  - Body (ít nhất 1 field): `{ "new_name": "new_name", "metadata": { ... } }`
- **Xóa collection**: `DELETE /collections/{collection_name}`
- **Ingest CSV**: `POST /ingest`
  - Form-data: `file` (CSV), `index_column`, `collection_name` (optional)
- **Query RAG**: `POST /collections/{collection_name}/query`
  - Body:
    `{ "query": "...", "columns_to_answer": [...], "number_docs_retrieval": 3, "gemini_api_key": "..." (optional) }`

## Ví dụ nhanh (curl)

- **Health – GET /health**
  ```bash
  curl http://127.0.0.1:8000/health
  ```

- **List collections – GET /collections**
  ```bash
  curl http://127.0.0.1:8000/collections
  ```

- **Tạo collection – POST /collections**
  ```bash
  curl -X POST http://127.0.0.1:8000/collections ^
    -H "Content-Type: application/json" ^
    -d "{\"name\": \"rag_collection_demo\", \"description\": \"Demo\"}"
  ```

- **Xem 1 collection – GET /collections/{collection_name}**
  ```bash
  curl http://127.0.0.1:8000/collections/rag_collection_demo
  ```

- **Sửa collection – PATCH /collections/{collection_name}**
  ```bash
  curl -X PATCH http://127.0.0.1:8000/collections/rag_collection_demo ^
    -H "Content-Type: application/json" ^
    -d "{\"new_name\": \"rag_collection_new\", \"metadata\": {\"description\": \"Updated\"}}"
  ```

- **Xóa collection – DELETE /collections/{collection_name}**
  ```bash
  curl -X DELETE http://127.0.0.1:8000/collections/rag_collection_new
  ```

- **Ingest CSV – POST /ingest**
  ```bash
  curl -X POST http://127.0.0.1:8000/ingest ^
    -F "file=@data.csv" ^
    -F "index_column=content" ^
    -F "collection_name=rag_collection_demo"
  ```

- **Query RAG – POST /collections/{collection_name}/query**
  ```bash
  curl -X POST http://127.0.0.1:8000/collections/rag_collection_demo/query ^
    -H "Content-Type: application/json" ^
    -d "{\"query\": \"Hỏi gì đó...\", \"columns_to_answer\": [\"content\"], \"number_docs_retrieval\": 3}"
  ```
