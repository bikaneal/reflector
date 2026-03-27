import os
import asyncio
import logging
import time
from typing import List, Optional
import pdfplumber
import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Minimal document ingestion for PDFs into LangChain Document objects."""

    @staticmethod
    async def process_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        loop = asyncio.get_event_loop()

        def _sync_work():
            docs: List[Document] = []
            try:
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text()
                        if not text:
                            continue
                        cleaned = ' '.join(text.split())
                        chunks = splitter.split_text(cleaned)
                        for chunk_idx, chunk in enumerate(chunks):
                            docs.append(Document(
                                page_content=chunk, 
                                metadata={
                                    "source": file_path, 
                                    "page": i,
                                    "chunk_index": len(docs)  # Global chunk sequence
                                }
                            ))
            except Exception as e:
                print(f"Error reading PDF {file_path}: {e}")
            return docs

        docs = await loop.run_in_executor(None, _sync_work)
        return docs

    @staticmethod
    async def process_documents(directory: str) -> List[Document]:
        documents: List[Document] = []
        if not os.path.exists(directory):
            print(f"Documents directory does not exist: {directory}")
            return documents

        tasks = []
        for fn in os.listdir(directory):
            if fn.lower().endswith('.pdf'):
                tasks.append(DocumentProcessor.process_pdf(os.path.join(directory, fn)))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    print(f"Error processing file: {res}")
                else:
                    documents.extend(res)

        print(f"Loaded {len(documents)} document chunks from {directory}")
        return documents


class SingleVectorStore:
    """Simple single Chroma store wrapper (one collection)."""

    def __init__(self, persist_directory: str = None, embeddings = None):
        # Default to project-root absolute path to avoid creating data under app/
        if persist_directory is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(project_root, "personality_data", "db")
        self.persist_directory = persist_directory
        # Use provided embeddings or create new one
        if embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "mps"}  # Use Metal Performance Shaders on Mac
            )
        else:
            self.embeddings = embeddings
        self.client = None
        self.store: Optional[Chroma] = None

    async def init_store(self, documents_path: Optional[str] = None, collection_name: str = "default"):
        """Load existing Chroma DB or create from documents_path if provided."""
        loop = asyncio.get_event_loop()

        def _create_or_load():
            # Modern API: use PersistentClient
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            return Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )

        self.store = await loop.run_in_executor(None, _create_or_load)
        logger.info(f"Vector store initialized at: {self.persist_directory}, collection: {collection_name}")

        # If collection is empty and documents_path provided, ingest
        if documents_path and os.path.exists(documents_path):
            existing = await loop.run_in_executor(None, lambda: self.store._collection.count())
            if existing == 0:
                logger.info(f"Collection '{collection_name}' is empty, ingesting from {documents_path}")
                docs = await DocumentProcessor.process_documents(documents_path)
                if docs:
                    await loop.run_in_executor(None, self.store.add_documents, docs)
                    logger.info(f"Ingested {len(docs)} chunks into collection '{collection_name}'")
            else:
                logger.info(f"Loaded existing collection '{collection_name}' with {existing} documents")

    async def similarity_search(self, query: str, k: int = 3) -> str:
        """Search within a single store and return formatted text context."""
        if not self.store:
            raise RuntimeError("Vector store is not initialized")
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self.store.similarity_search_with_relevance_scores,
            query,
            k,
        )
        context_parts: List[str] = []
        for i, (doc, score) in enumerate(results, start=1):
            normalized_score = (score + 1) / 2
            snippet = doc.page_content
            src = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"Fragment {i} (relevance score={normalized_score:.4f})\n"
                f"Source: {os.path.basename(src)}\n{snippet}\n"
            )

        return "\n\n".join(context_parts)

    async def fetch_full_document_by_source(self, source: str) -> str:
        """Reconstruct the full document by fetching all chunks with the same 'source'.

        Uses insertion order from Chroma to preserve document sequence.
        Prefers 'chunk_index' if available for explicit ordering.
        """
        if not self.store:
            raise RuntimeError("Vector store is not initialized")

        loop = asyncio.get_event_loop()

        def _fetch_all() -> str:
            t0 = time.time()
            # Access underlying Chroma collection directly for flexible filters
            col = self.store._collection
            res = col.get(where={"source": source}, include=["documents", "metadatas"], limit=100000, offset=0)
            t1 = time.time()
            logger.info(f"[fetch_full_document] Chroma query took {t1-t0:.3f}s")
            
            docs = res.get("documents", []) or []
            metas = res.get("metadatas", []) or []

            items = []
            for idx, (meta, doc) in enumerate(zip(metas, docs)):
                sort_key = idx  # Default: preserve Chroma insertion order
                
                # Prefer chunk_index if present (for explicit ordering)
                if "chunk_index" in meta:
                    try:
                        sort_key = int(meta.get("chunk_index", idx))
                    except Exception:
                        sort_key = idx
                
                items.append((sort_key, doc or ""))
            
            items.sort(key=lambda x: x[0])
            t2 = time.time()
            logger.info(f"[fetch_full_document] Sorting {len(items)} chunks took {t2-t1:.3f}s")
            return "\n".join([d for _, d in items])

        full_text = await loop.run_in_executor(None, _fetch_all)
        return full_text


class MultiVectorStore:
    """Wrapper to query several SingleVectorStore instances in parallel.

    Each underlying store is assumed to be already initialized via ``init_store``.
    ``similarity_search`` aggregates results from all stores into a single text
    context string, tagging which DB each fragment came from.
    """

    def __init__(self, stores: dict[str, SingleVectorStore], strategies: Optional[dict[str, str]] = None):
        # key -> human readable database name, value -> initialized SingleVectorStore
        self.stores = stores
        # strategies: store_name -> 'chunks' | 'full_case'
        self.strategies: dict[str, str] = strategies or {name: "chunks" for name in stores.keys()}

    async def _search_in_store(
        self,
        store: SingleVectorStore,
        store_name: str,
        query: str,
        k: int,
    ) -> List[str]:
        if not store.store:
            raise RuntimeError(f"Vector store '{store_name}' is not initialized")

        strategy = self.strategies.get(store_name, "chunks")
        loop = asyncio.get_event_loop()

        if strategy == "full_case":
            t_start = time.time()
            # For full_case strategy, always retrieve top 1 chunk, then reconstruct entire case
            t0 = time.time()
            base_results = await loop.run_in_executor(
                None,
                store.store.similarity_search_with_relevance_scores,
                query,
                1,  # Always 1 for full_case mode
            )
            t1 = time.time()
            logger.info(f"[{store_name}] Similarity search took {t1-t0:.3f}s")
            
            chunks: List[str] = []
            seen_sources: set[str] = set()
            for doc, score in base_results:
                src = doc.metadata.get("source")
                if not src or src in seen_sources:
                    continue
                seen_sources.add(src)
                t2 = time.time()
                full_text = await store.fetch_full_document_by_source(src)
                t3 = time.time()
                logger.info(f"[{store_name}] Full document reconstruction took {t3-t2:.3f}s")
                
                normalized_score = (score + 1) / 2
                chunks.append(
                    f"[DB: {store_name}] Full case (relevance score={normalized_score:.4f})\n"
                    f"Source: {os.path.basename(src)}\n{full_text}\n"
                )
                if len(chunks) >= k:
                    break
            t_end = time.time()
            logger.info(f"[{store_name}] Full-case search completed in {t_end-t_start:.3f}s total")
            return chunks
        else:
            # Default chunk-level behavior
            results = await loop.run_in_executor(
                None,
                store.store.similarity_search_with_relevance_scores,
                query,
                k,
            )

            chunks: List[str] = []
            for i, (doc, score) in enumerate(results, start=1):
                normalized_score = (score + 1) / 2
                snippet = doc.page_content
                src = doc.metadata.get("source", "unknown")
                chunks.append(
                    f"[DB: {store_name}] Fragment {i} (relevance score={normalized_score:.4f})\n"
                    f"Source: {os.path.basename(src)}\n{snippet}\n"
                )

            return chunks

    async def similarity_search(self, query: str, k: int = 3) -> str:
        """Run similarity search in all configured stores in parallel."""
        if not self.stores:
            raise RuntimeError("No vector stores configured for MultiVectorStore")

        tasks = []
        for name, store in self.stores.items():
            tasks.append(self._search_in_store(store, name, query, k))

        all_chunks: List[str] = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in parallel search: {result}")
                else:
                    all_chunks.extend(result)

        return "\n\n".join(all_chunks)

    async def similarity_search_separate(self, query: str, k: int = 3) -> dict[str, str]:
        """Run similarity search and keep results per-store as separate strings.

        Returns a dict {store_name: context_string}.
        """
        if not self.stores:
            raise RuntimeError("No vector stores configured for MultiVectorStore")

        # Keep names in a list to preserve ordering with gathered results
        names: List[str] = []
        tasks = []
        for name, store in self.stores.items():
            names.append(name)
            tasks.append(self._search_in_store(store, name, query, k))

        per_store: dict[str, str] = {}
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(names, results):
                if isinstance(result, Exception):
                    logger.error(f"Error in parallel search for store '{name}': {result}")
                    per_store[name] = ""
                else:
                    per_store[name] = "\n\n".join(result)

        return per_store

