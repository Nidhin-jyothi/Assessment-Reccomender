"""
retriever.py - Hybrid retrieval: ChromaDB (dense) + BM25 (sparse) + reranking.

"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from catalog import Catalog, CatalogItem


#  Config 
CHROMA_PERSIST_DIR  = "./chroma_db"
COLLECTION_NAME     = "shl_catalog"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"   # fast, free, good quality
TOP_K_DENSE         = 20
TOP_K_BM25          = 20
TOP_K_FINAL         = 10
RRF_K               = 60   # Reciprocal Rank Fusion constant


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class HybridRetriever:
    """
    Hybrid retriever combining dense (ChromaDB) and sparse (BM25) search
    with Reciprocal Rank Fusion for merging.
    """

    def __init__(self, catalog: Catalog, persist_dir: str = CHROMA_PERSIST_DIR):
        self.catalog     = catalog
        self._items      = catalog.all_items()
        self._id_to_item = {item.entity_id: item for item in self._items}

        #  Dense index (ChromaDB) 
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._chroma_client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._get_or_build_collection()

        #  Sparse index (BM25) 
        self._bm25_corpus  = [_tokenize(item.corpus) for item in self._items]
        self._bm25         = BM25Okapi(self._bm25_corpus)

        print(f"[Retriever] Ready - {len(self._items)} items indexed.")

    def _get_or_build_collection(self) -> chromadb.Collection:
        """Get existing collection or build+persist a new one."""
        existing = [c.name for c in self._chroma_client.list_collections()]

        if COLLECTION_NAME in existing:
            collection = self._chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self._ef,
            )
            # Validate size matches catalog
            if collection.count() == len(self._items):
                print(f"[Retriever] Loaded existing ChromaDB collection ({collection.count()} items).")
                return collection
            else:
                print(f"[Retriever] Collection size mismatch - rebuilding.")
                self._chroma_client.delete_collection(COLLECTION_NAME)

        return self._build_collection()

    def _build_collection(self) -> chromadb.Collection:
        """Build ChromaDB collection from catalog items."""
        print("[Retriever] Building ChromaDB collection (first run, ~1-2 min)...")
        collection = self._chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        # Batch upsert (ChromaDB handles batching internally)
        BATCH = 100
        items = self._items
        for i in range(0, len(items), BATCH):
            batch = items[i : i + BATCH]
            collection.add(
                ids        = [item.entity_id for item in batch],
                documents  = [item.corpus    for item in batch],
                metadatas  = [
                    {
                        "name":      item.name,
                        "url":       item.url,
                        "keys":      ",".join(item.keys),
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
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, len(self._items)),
            include=["distances"],
        )
        ids       = results["ids"][0]
        distances = results["distances"][0]
        # ChromaDB cosine distance  similarity: sim = 1 - dist
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
