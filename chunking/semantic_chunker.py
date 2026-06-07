from __future__ import annotations

import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .base_chunker import BaseChunker


_TOKENIZERS_READY = False


def _ensure_tokenizers() -> None:
    global _TOKENIZERS_READY
    if _TOKENIZERS_READY:
        return

    for resource_path, package_name in (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ):
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)

    _TOKENIZERS_READY = True


class ProtonxSemanticChunker(BaseChunker):
    def __init__(self, threshold=0.3, model="keepitreal/vietnamese-sbert"):
        self.threshold = threshold
        self.model = (
            model if isinstance(model, SentenceTransformer) else SentenceTransformer(model)
        )
        _ensure_tokenizers()

    def embed_function(self, sentences):
        return self.model.encode(sentences)

    def split_text(self, text):
        sentences = nltk.sent_tokenize(text)
        sentences = [item for item in sentences if item and item.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        vectors = self.embed_function(sentences)
        similarities = cosine_similarity(vectors)
        chunks = [[sentences[0]]]

        for i in range(1, len(sentences)):
            sim_score = similarities[i - 1, i]

            if sim_score >= self.threshold:
                chunks[-1].append(sentences[i])
            else:
                chunks.append([sentences[i]])

        return [" ".join(chunk) for chunk in chunks]
