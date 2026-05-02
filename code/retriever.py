"""
Corpus Retriever — Loads, embeds, and searches the support corpus.
Uses FAISS for fast vector search with numpy cosine fallback.
"""
import os
import sys
import json
import pickle
import re

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CORPUS_DIRS, EMBEDDINGS_CACHE, RETRIEVAL_QUERY_PROMPT, TOP_K_DOCS
from llm_client import call_llm, get_embeddings
from logger import log


class CorpusRetriever:
    """Manages corpus loading, embedding, and retrieval."""

    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.index = None
        self._loaded = False

    def load_corpus(self):
        """Load all documents from all corpus directories."""
        if self._loaded:
            return
        log("RETRIEVER", "Loading corpus...")
        for domain, path in CORPUS_DIRS.items():
            if path.exists():
                self._load_directory(str(path), domain)
        self._loaded = True
        log("RETRIEVER", f"Loaded {len(self.documents)} document chunks total")

    def _load_directory(self, directory: str, domain: str):
        """Recursively load all .md, .txt, .json, .html files."""
        supported = {'.md', '.txt', '.json', '.html', '.htm'}
        for root, dirs, files in os.walk(directory):
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in supported:
                    continue
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, directory)
                try:
                    content = self._read_file(filepath, ext)
                    if content and len(content.strip()) > 20:
                        chunks = self._chunk_document(content)
                        for i, chunk in enumerate(chunks):
                            doc_id = f"{domain}/{rel_path}" if len(chunks) == 1 else f"{domain}/{rel_path}#chunk{i+1}"
                            self.documents.append({
                                "id": doc_id,
                                "content": chunk,
                                "source": domain,
                                "filename": filename,
                            })
                except Exception as e:
                    log("RETRIEVER", f"Error loading {filepath}: {e}")

    def _read_file(self, filepath: str, ext: str) -> str:
        """Read file content."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            with open(filepath, "r", encoding="latin-1") as f:
                content = f.read()

        if ext in ('.html', '.htm'):
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()
        return content

    def _chunk_document(self, text: str, max_size: int = 1500) -> list:
        """Split document into chunks at paragraph boundaries."""
        if len(text) <= max_size:
            return [text]
        chunks, current = [], ""
        for para in text.split('\n\n'):
            if len(current) + len(para) + 2 <= max_size:
                current += ("\n\n" if current else "") + para
            else:
                if current:
                    chunks.append(current)
                if len(para) > max_size:
                    # Split long paragraphs by sentence
                    for i in range(0, len(para), max_size):
                        chunks.append(para[i:i + max_size])
                    current = ""
                else:
                    current = para
        if current:
            chunks.append(current)
        return chunks or [text[:max_size]]

    def build_embeddings(self, force_rebuild: bool = False):
        """Build or load cached embeddings."""
        if not self.documents:
            return

        if not force_rebuild and EMBEDDINGS_CACHE.exists():
            try:
                with open(EMBEDDINGS_CACHE, "rb") as f:
                    cache = pickle.load(f)
                if cache.get("doc_count") == len(self.documents):
                    self.embeddings = np.array(cache["embeddings"], dtype=np.float32)
                    log("RETRIEVER", f"Loaded cached embeddings ({len(self.documents)} docs)")
                    self._build_index()
                    return
            except Exception as e:
                log("RETRIEVER", f"Cache load failed: {e}")

        log("RETRIEVER", f"Building embeddings for {len(self.documents)} documents...")
        texts = [doc["content"] for doc in self.documents]
        embeddings_list = get_embeddings(texts, input_type="passage")
        self.embeddings = np.array(embeddings_list, dtype=np.float32)

        try:
            EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(EMBEDDINGS_CACHE, "wb") as f:
                pickle.dump({"doc_count": len(self.documents), "embeddings": embeddings_list}, f)
            log("RETRIEVER", "Embeddings cached")
        except Exception as e:
            log("RETRIEVER", f"Cache save failed: {e}")
        self._build_index()

    def _build_index(self):
        """Build FAISS index for fast search."""
        try:
            import faiss
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = (self.embeddings / norms).astype(np.float32)
            self.index.add(normalized)
            log("RETRIEVER", f"FAISS index built ({self.index.ntotal} vectors)")
        except ImportError:
            log("RETRIEVER", "FAISS not available, using numpy cosine similarity")
            self.index = None

    def retrieve(self, issue: str, subject: str = "", company: str = "", top_k: int = None, issue_id: str = None) -> list:
        """Retrieve most relevant documents for a query."""
        if not self.documents:
            return []
        if top_k is None:
            top_k = TOP_K_DOCS

        # Rewrite query for better retrieval
        try:
            prompt = RETRIEVAL_QUERY_PROMPT.format(issue=issue, subject=subject, company=company)
            search_query = call_llm(prompt, temperature=0.0, max_tokens=200).strip().strip('"\'')
            log("RETRIEVER", f"Search query: {search_query}", issue_id)
        except Exception:
            search_query = f"{issue} {subject} {company}".strip()

        query_embedding = np.array(get_embeddings([search_query], input_type="query")[0], dtype=np.float32)

        if self.index is not None:
            q_norm = np.linalg.norm(query_embedding)
            if q_norm > 0:
                query_normalized = (query_embedding / q_norm).reshape(1, -1)
            else:
                query_normalized = query_embedding.reshape(1, -1)
            scores, indices = self.index.search(query_normalized, min(top_k, len(self.documents)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                doc = self.documents[idx]
                results.append({"id": doc["id"], "content": doc["content"], "source": doc["source"], "score": float(score)})
        else:
            results = self._numpy_search(query_embedding, top_k)

        # Boost docs matching the company domain — stronger preference for same-domain results
        company_lower = (company or "").lower().strip()
        if company_lower and company_lower != "none":
            for r in results:
                if r["source"] == company_lower:
                    r["score"] += 0.2  # Boost same-domain docs
            results.sort(key=lambda x: x["score"], reverse=True)

        top_scores = [f"{r['score']:.3f}" for r in results[:3]]
        log("RETRIEVER", f"Retrieved {len(results)} docs (top scores: {top_scores})", issue_id)
        return results

    def _numpy_search(self, query_embedding: np.ndarray, top_k: int) -> list:
        """Fallback cosine similarity search."""
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            q_normalized = query_embedding / q_norm
        else:
            q_normalized = query_embedding
        norms = np.linalg.norm(self.embeddings, axis=1)
        norms[norms == 0] = 1
        doc_normalized = self.embeddings / norms[:, np.newaxis]
        similarities = doc_normalized @ q_normalized
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [
            {"id": self.documents[i]["id"], "content": self.documents[i]["content"],
             "source": self.documents[i]["source"], "score": float(similarities[i])}
            for i in top_indices
        ]


# Singleton
_retriever = None

def get_retriever() -> CorpusRetriever:
    """Get or create singleton retriever."""
    global _retriever
    if _retriever is None:
        _retriever = CorpusRetriever()
        _retriever.load_corpus()
        _retriever.build_embeddings()
    return _retriever

def retrieve_docs(issue: str, subject: str = "", company: str = "", top_k: int = None, issue_id: str = None) -> list:
    """Convenience function."""
    return get_retriever().retrieve(issue, subject, company, top_k, issue_id)
