import logging

from loaders.loader_factory import LoaderFactory
from processing.chunker import TextChunker

from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_store import VectorStore

from retrieval.bm25_retriever import BM25Retriever

from services.document_summary import DocumentSummarizer

logger = logging.getLogger(__name__)


class NoExtractableContentError(Exception):
    """Raised when no text could be extracted from any uploaded file."""
    pass


class IndexingService:

    def __init__(self):

        self.chunker = TextChunker(
            chunk_size=500,
            overlap=100
        )

        self.embedding_model = EmbeddingModel()

        self.summarizer = DocumentSummarizer()

    def build_index(self, file_paths):

        all_chunks = []
        failed_files = []

        # ----------------------------------
        # Load and Chunk Documents
        # ----------------------------------

        for file_path in file_paths:

            try:

                loader = LoaderFactory.get_loader(
                    file_path
                )

                pages = loader.load()

                chunks = self.chunker.chunk(
                    pages
                )

                all_chunks.extend(
                    chunks
                )

            except Exception as exc:

                # Don't let one bad file (bad encoding, corrupt PDF,
                # unsupported extension, etc.) kill the whole batch.
                logger.warning(
                    "Failed to process %s: %s", file_path, exc
                )

                failed_files.append(
                    {"file": file_path, "error": str(exc)}
                )

        if not all_chunks:

            raise NoExtractableContentError(
                "No text could be extracted from the uploaded documents."
            )

        # ----------------------------------
        # Embeddings
        # ----------------------------------

        texts = [

            chunk["text"]

            for chunk in all_chunks

        ]

        embeddings = self.embedding_model.encode(
            texts
        )

        # ----------------------------------
        # FAISS
        # ----------------------------------

        vector_store = VectorStore(
            embeddings.shape[1]
        )

        vector_store.add(
            embeddings,
            all_chunks
        )

        # ----------------------------------
        # BM25
        # ----------------------------------

        bm25_store = BM25Retriever()

        bm25_store.add(
            all_chunks
        )

        # ----------------------------------
        # Return
        # ----------------------------------

        return {

            "vector_store": vector_store,

            "bm25_store": bm25_store,

            "documents": len(file_paths) - len(failed_files),

            "chunks": len(all_chunks),

            "failed_files": failed_files

        }

    def summarize_documents(
        self,
        documents
    ):

        return self.summarizer.summarize(
            documents
        )

    def generate_questions(
        self,
        documents
    ):

        return self.summarizer.generate_questions(
            documents
        )
