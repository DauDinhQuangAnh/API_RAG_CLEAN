from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

PRIMARY_MODEL_NAME = "keepitreal/vietnamese-sbert"
FALLBACK_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_STATE_FILE = Path(__file__).resolve().with_name(".embedding_model_state.json")
TEST_SENTENCES = ["Xin chao"]
_SENTENCE_TRANSFORMER_CLASS: Any | None = None
_SENTENCE_TRANSFORMER_CHECKED = False


class HuggingFaceTransformerEmbeddingModel:
    """Minimal SentenceTransformer-compatible encoder backed by transformers."""

    def __init__(self, model_name: str, *, local_files_only: bool = False):
        _disable_transformers_sklearn_probe()
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.model_name = model_name
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.dimension = int(getattr(self.model.config, "hidden_size", 0) or 0)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: Any) -> list[list[float]]:
        text_list = [str(text) for text in texts]
        if not text_list:
            return []

        torch = self._torch
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"]
            input_mask = attention_mask.unsqueeze(-1)
            input_mask = input_mask.expand(token_embeddings.size())
            input_mask = input_mask.to(token_embeddings.dtype)
            summed = torch.sum(token_embeddings * input_mask, dim=1)
            counts = torch.clamp(input_mask.sum(dim=1), min=1e-9)
            embeddings = summed / counts
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()


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


def _validate_model(model: Any) -> None:
    embeddings = model.encode(TEST_SENTENCES)
    if not len(embeddings) or not len(embeddings[0]):
        raise RuntimeError("Embedding model returned an empty vector.")


def _load_sentence_transformer_class() -> Any | None:
    global _SENTENCE_TRANSFORMER_CHECKED, _SENTENCE_TRANSFORMER_CLASS
    if _SENTENCE_TRANSFORMER_CHECKED:
        return _SENTENCE_TRANSFORMER_CLASS

    _SENTENCE_TRANSFORMER_CHECKED = True
    try:
        from sentence_transformers import SentenceTransformer

        _SENTENCE_TRANSFORMER_CLASS = SentenceTransformer
    except Exception as exc:
        print(
            "sentence_transformers is unavailable; "
            f"using transformers fallback: {exc}"
        )
        _SENTENCE_TRANSFORMER_CLASS = None

    return _SENTENCE_TRANSFORMER_CLASS


def _disable_transformers_sklearn_probe() -> None:
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils

        if hasattr(import_utils.is_sklearn_available, "cache_clear"):
            import_utils.is_sklearn_available.cache_clear()
        import_utils.is_sklearn_available = lambda: False
        transformers_utils.is_sklearn_available = lambda: False
    except Exception:
        return


def _build_embedding_model(model_name: str, *, local_files_only: bool) -> Any:
    sentence_transformer = _load_sentence_transformer_class()
    if sentence_transformer is not None:
        try:
            return sentence_transformer(
                model_name,
                local_files_only=local_files_only,
            )
        except TypeError:
            if local_files_only:
                return sentence_transformer(model_name)
            raise

    return HuggingFaceTransformerEmbeddingModel(
        model_name,
        local_files_only=local_files_only,
    )


def _try_load_local_model(model_name: str) -> Optional[Any]:
    try:
        model = _build_embedding_model(model_name, local_files_only=True)
        _validate_model(model)
        return model
    except Exception:
        return None


def ensure_embedding_model(
    preferred_model: str = PRIMARY_MODEL_NAME,
    fallback_model: str = FALLBACK_MODEL_NAME,
) -> tuple[Any, str, bool]:
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
            model = _build_embedding_model(candidate, local_files_only=False)
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
