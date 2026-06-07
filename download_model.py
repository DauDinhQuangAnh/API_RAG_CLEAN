from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

PRIMARY_MODEL_NAME = "keepitreal/vietnamese-sbert"
FALLBACK_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_STATE_FILE = Path(__file__).resolve().with_name(".embedding_model_state.json")
TEST_SENTENCES = ["Xin chao"]


def _load_saved_model_name() -> Optional[str]:
    if not MODEL_STATE_FILE.exists():
        return None

    try:
        payload = json.loads(MODEL_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    model_name = payload.get("model_name")
    return model_name if isinstance(model_name, str) and model_name else None


def _save_model_name(model_name: str) -> None:
    payload = {"model_name": model_name}
    MODEL_STATE_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _validate_model(model: SentenceTransformer) -> None:
    embeddings = model.encode(TEST_SENTENCES)
    if not len(embeddings) or not len(embeddings[0]):
        raise RuntimeError("Embedding model returned an empty vector.")


def _try_load_local_model(model_name: str) -> Optional[SentenceTransformer]:
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
        _validate_model(model)
        return model
    except TypeError:
        return None
    except Exception:
        return None


def ensure_embedding_model(
    preferred_model: str = PRIMARY_MODEL_NAME,
    fallback_model: str = FALLBACK_MODEL_NAME,
) -> tuple[SentenceTransformer, str, bool]:
    saved_model_name = _load_saved_model_name()
    candidates: list[str] = []

    for candidate in (saved_model_name, preferred_model, fallback_model):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        model = _try_load_local_model(candidate)
        if model is not None:
            print(f"Embedding model already available locally: {candidate}")
            _save_model_name(candidate)
            return model, candidate, False

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            print(f"Preparing embedding model: {candidate}")
            model = SentenceTransformer(candidate)
            _validate_model(model)
            _save_model_name(candidate)
            print(f"Embedding model ready: {candidate}")
            return model, candidate, True
        except Exception as exc:
            last_error = exc
            print(f"Failed to prepare embedding model '{candidate}': {exc}")

    raise RuntimeError("Unable to prepare any embedding model.") from last_error


def main() -> None:
    print(
        "SentenceTransformer cache:",
        os.path.expanduser("~/.cache/torch/sentence_transformers/"),
    )
    _, model_name, downloaded_now = ensure_embedding_model()

    if downloaded_now:
        print(f"Downloaded model successfully: {model_name}")
    else:
        print(f"Model already present, skipping download: {model_name}")


if __name__ == "__main__":
    main()
