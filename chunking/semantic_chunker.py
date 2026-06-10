from __future__ import annotations

import nltk
import re
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
    def __init__(
        self,
        threshold=0.3,
        model="keepitreal/vietnamese-sbert",
        min_chunk_chars=120,
        max_chunk_chars=1400,
    ):
        self.threshold = threshold
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.model = SentenceTransformer(model) if isinstance(model, str) else model
        _ensure_tokenizers()

    def embed_function(self, sentences):
        return self.model.encode(sentences)

    def split_text(self, text):
        sentences = nltk.sent_tokenize(text)
        sentences = self._prepare_sentences(sentences)
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

        merged_chunks = self._merge_short_chunks([" ".join(chunk) for chunk in chunks])
        split_chunks = self._split_long_chunks(merged_chunks)
        return self._merge_short_chunks(split_chunks)

    def _prepare_sentences(self, sentences):
        prepared = []
        index = 0

        while index < len(sentences):
            sentence = self._normalize_sentence(sentences[index])
            index += 1

            if not sentence or self._is_noise_sentence(sentence):
                continue

            if self._is_heading_fragment(sentence) and index < len(sentences):
                next_sentence = self._normalize_sentence(sentences[index])
                index += 1
                if next_sentence and not self._is_noise_sentence(next_sentence):
                    sentence = f"{sentence} {next_sentence}"

            prepared.append(sentence)

        return prepared

    def _normalize_sentence(self, sentence):
        sentence = re.sub(r"[ \t]+", " ", sentence).strip()
        sentence = re.sub(r"\s*\n\s*", " ", sentence)
        return sentence.strip()

    def _is_noise_sentence(self, sentence):
        if re.fullmatch(r"Trang\s+\d+\s*/\s*\d+", sentence, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"[•●○▪\-–—\s]+", sentence):
            return True
        return False

    def _is_heading_fragment(self, sentence):
        if re.fullmatch(r"\d+\.", sentence):
            return True
        if len(sentence) <= 90 and re.search(r"(?:^|\s)\d+\.$", sentence):
            return True
        return False

    def _merge_short_chunks(self, chunks):
        merged = []
        index = 0

        while index < len(chunks):
            chunk = chunks[index].strip()
            index += 1
            if not chunk:
                continue

            if len(chunk) < self.min_chunk_chars:
                if self._is_heading_fragment(chunk) and index < len(chunks):
                    chunks[index] = f"{chunk} {chunks[index].strip()}"
                    continue
                if merged:
                    merged[-1] = f"{merged[-1]} {chunk}"
                    continue
                if index < len(chunks):
                    chunks[index] = f"{chunk} {chunks[index].strip()}"
                    continue

            merged.append(chunk)

        return merged

    def _split_long_chunks(self, chunks):
        split_chunks = []

        for chunk in chunks:
            if len(chunk) <= self.max_chunk_chars:
                split_chunks.append(chunk)
                continue

            current = ""
            for sentence in nltk.sent_tokenize(chunk):
                sentence = self._normalize_sentence(sentence)
                if not sentence:
                    continue

                if len(sentence) > self.max_chunk_chars:
                    if current:
                        split_chunks.append(current)
                        current = ""
                    split_chunks.extend(self._split_long_text(sentence))
                    continue

                if current and len(current) + len(sentence) + 1 > self.max_chunk_chars:
                    split_chunks.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()

            if current:
                split_chunks.append(current)

        return split_chunks

    def _split_long_text(self, text):
        parts = []
        current = ""
        for word in text.split():
            if current and len(current) + len(word) + 1 > self.max_chunk_chars:
                parts.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()

        if current:
            parts.append(current)
        return parts
