"""Compact, deterministic BM25 retrieval baseline."""

from collections import Counter
from collections.abc import Sequence
from math import log

from ..contracts import Evidence, KnowledgeCard
from .tokenizer import tokenize


class BM25Retriever:
    """BM25 over card title, content and tags."""

    def __init__(self, cards: Sequence[KnowledgeCard], k1: float = 1.5, b: float = 0.75):
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 >= 0 and b between 0 and 1")
        self.cards = tuple(sorted(cards, key=lambda card: card.id))
        self.k1 = k1
        self.b = b
        self._documents = [
            tokenize(" ".join((card.title, card.content, *card.tags)))
            for card in self.cards
        ]
        self._term_frequencies = [Counter(document) for document in self._documents]
        document_frequency: Counter[str] = Counter()
        for document in self._documents:
            document_frequency.update(set(document))
        self._document_frequency = document_frequency
        self._document_count = len(self.cards)
        self._average_document_length = (
            sum(len(document) for document in self._documents) / self._document_count
            if self._document_count
            else 0.0
        )

    def _idf(self, term: str) -> float:
        frequency = self._document_frequency.get(term, 0)
        return log(
            (self._document_count - frequency + 0.5) / (frequency + 0.5) + 1
        )

    def _score(self, query_terms: Sequence[str], index: int) -> float:
        if not query_terms or not self._average_document_length:
            return 0.0
        frequencies = self._term_frequencies[index]
        document_length = len(self._documents[index])
        score = 0.0
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                continue
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * document_length / self._average_document_length
            )
            score += self._idf(term) * (
                term_frequency * (self.k1 + 1) / denominator
            )
        return score

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if top_k <= 0:
            return []
        query_terms = tokenize(query)
        ranked = [
            (self._score(query_terms, index), card)
            for index, card in enumerate(self.cards)
        ]
        ranked = [item for item in ranked if item[0] > 0]
        ranked.sort(key=lambda item: (-item[0], item[1].id))
        return [
            Evidence(
                source_id=card.id,
                title=card.title,
                excerpt=card.content,
                source_url=card.source_url,
                score=score,
            )
            for score, card in ranked[:top_k]
        ]
