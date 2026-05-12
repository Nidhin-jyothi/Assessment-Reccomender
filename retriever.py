"""
retriever.py - Hybrid retrieval: ChromaDB (dense) + BM25 (sparse) + reranking.

"""

from __future__ import annotations

import os
import time
from typing import Optional

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types as gtypes

from catalog import Catalog, CatalogItem


#  Config 
CHROMA_PERSIST_DIR  = "./chroma_db"  # unused on Render (ephemeral FS)
COLLECTION_NAME     = "shl_catalog"
EMBEDDING_MODEL     = "gemini-embedding-001"   # correct model name
TOP_K_DENSE         = 20
TOP_K_BM25          = 20
TOP_K_FINAL         = 10
RRF_K               = 60   # Reciprocal Rank Fusion constant


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using Google's Gemini gemini-embedding-001.
    Uses task_type='RETRIEVAL_DOCUMENT' for indexing and 'RETRIEVAL_QUERY' for queries.
    """

    def __init__(self, task_type: str = "RETRIEVAL_DOCUMENT"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self._client = genai.Client(api_key=api_key)
        self._task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        BATCH = 50  # conservative batch size for rate limiting
        for i in range(0, len(input), BATCH):
            batch = input[i : i + BATCH]
            result = self._client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=gtypes.EmbedContentConfig(task_type=self._task_type),
            )
            embeddings.extend([e.values for e in result.embeddings])
            if i + BATCH < len(input):
                time.sleep(1.0)  # respect rate limits during bulk indexing
        return embeddings


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class HybridRetriever:
    """
    Hybrid retriever combining dense (ChromaDB) and sparse (BM25) search
    with Reciprocal Rank Fusion for merging.
    """

    def __init__(self, catalog: Catalog):
        self.catalog     = catalog
        self._items      = catalog.all_items()
        self._id_to_item = {item.entity_id: item for item in self._items}

        #  Dense index (ChromaDB in-memory + Gemini Embeddings) 
        self._ef = GeminiEmbeddingFunction(task_type="RETRIEVAL_DOCUMENT")
        self._ef_query = GeminiEmbeddingFunction(task_type="RETRIEVAL_QUERY")
        self._chroma_client = chromadb.EphemeralClient()  # in-memory; safe for stateless Render
        self._collection = self._build_collection()

        #  Sparse index (BM25) 
        self._bm25_corpus  = [_tokenize(item.corpus) for item in self._items]
        self._bm25         = BM25Okapi(self._bm25_corpus)

        print(f"[Retriever] Ready - {len(self._items)} items indexed.")

    def _build_collection(self) -> chromadb.Collection:
        """Build in-memory ChromaDB collection with Gemini embeddings."""
        print("[Retriever] Building ChromaDB collection via Gemini embeddings...")
        collection = self._chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        items = self._items
        # Pre-compute embeddings in batches (task_type=RETRIEVAL_DOCUMENT)
        CHROMA_BATCH = 100
        all_embeddings = self._ef([item.corpus for item in items])

        for i in range(0, len(items), CHROMA_BATCH):
            batch      = items[i : i + CHROMA_BATCH]
            batch_embs = all_embeddings[i : i + CHROMA_BATCH]
            collection.add(
                ids        = [item.entity_id for item in batch],
                embeddings = batch_embs,
                metadatas  = [
                    {
                        "name":       item.name,
                        "url":        item.url,
                        "keys":       ",".join(item.keys),
                        "job_levels": ",".join(item.job_levels),
                    }
                    for item in batch
                ],
            )
        print(f"[Retriever] ChromaDB built: {collection.count()} items.")
        return collection

    #  Dense retrieval 
    def _dense_search(self, query: str, k: int = TOP_K_DENSE) -> list[tuple[str, float]]:
        """Returns list of (entity_id, similarity_score)."""
        # Use query-specific embedding function for better retrieval quality
        query_embedding = self._ef_query([query])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, len(self._items)),
            include=["distances"],
        )
        ids       = results["ids"][0]
        distances = results["distances"][0]
        # ChromaDB cosine distance -> similarity: sim = 1 - dist
        return [(eid, 1.0 - dist) for eid, dist in zip(ids, distances)]

    #  Sparse retrieval 
    def _bm25_search(self, query: str, k: int = TOP_K_BM25) -> list[tuple[str, float]]:
        """Returns list of (entity_id, bm25_score)."""
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            (self._items[i].entity_id, float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    #  Reciprocal Rank Fusion 
    @staticmethod
    def _rrf_merge(
        dense_results: list[tuple[str, float]],
        bm25_results:  list[tuple[str, float]],
        k: int = RRF_K,
    ) -> list[tuple[str, float]]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.
        score(d) = sum over lists of 1 / (k + rank(d))
        """
        scores: dict[str, float] = {}
        for rank, (eid, _) in enumerate(dense_results, start=1):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank)
        for rank, (eid, _) in enumerate(bm25_results, start=1):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    #  Public search interface 
    def search(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        filter_job_levels: Optional[list[str]] = None,
        filter_languages:  Optional[list[str]] = None,
        filter_test_types: Optional[list[str]] = None,
        filter_remote:     Optional[bool]       = None,
    ) -> list[CatalogItem]:
        """
        Full hybrid search pipeline:
          1. Dense search (embeddings)
          2. BM25 sparse search
          3. RRF merge
          4. Post-hoc filtering (job level, language, test type)
          5. Return top_k items
        """
        if not query.strip():
            return []

        dense  = self._dense_search(query, k=TOP_K_DENSE)
        sparse = self._bm25_search(query,  k=TOP_K_BM25)
        merged = self._rrf_merge(dense, sparse)

        # Resolve entity_ids  CatalogItems
        results: list[CatalogItem] = []
        for eid, score in merged:
            item = self._id_to_item.get(eid)
            if item:
                results.append(item)

        # Post-hoc filters (applied after retrieval to avoid narrowing BM25 index)
        if filter_job_levels:
            fl = [l.lower() for l in filter_job_levels]
            results = [
                r for r in results
                if any(jl.lower() in fl for jl in r.job_levels)
            ] or results  # fall back to unfiltered if nothing matches

        if filter_languages:
            fl = [l.lower() for l in filter_languages]
            results_lang = [
                r for r in results
                if not r.languages or any(lang.lower() in fl for lang in r.languages)
            ]
            results = results_lang or results

        if filter_test_types:
            ft = [t.upper() for t in filter_test_types]
            results_type = [r for r in results if any(t in ft for t in r.test_types)]
            results = results_type or results

        if filter_remote is True:
            results_remote = [r for r in results if r.remote]
            results = results_remote or results

        return results[:top_k]

    def get_item(self, entity_id: str) -> Optional[CatalogItem]:
        return self._id_to_item.get(entity_id)
