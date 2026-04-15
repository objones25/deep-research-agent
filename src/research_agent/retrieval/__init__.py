"""Retrieval package: hybrid dense+BM25 search, RRF fusion, and FlashRank reranking."""

from research_agent.retrieval.bm25 import BM25Encoder
from research_agent.retrieval.collection import ensure_collection
from research_agent.retrieval.embedder import HuggingFaceEmbedder
from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.retrieval.protocols import Embedder, Reranker, Retriever, SearchResult
from research_agent.retrieval.reranker import FlashRankReranker

__all__ = [
    "SearchResult",
    "Embedder",
    "Retriever",
    "Reranker",
    "BM25Encoder",
    "HuggingFaceEmbedder",
    "HybridRetriever",
    "FlashRankReranker",
    "ensure_collection",
]
