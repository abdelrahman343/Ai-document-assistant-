import logging

logger = logging.getLogger(__name__)


class HybridRetriever:

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        rrf_k=60,
        min_score=0.0
    ):

        self.faiss = faiss_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

        # A chunk only ranked by one retriever (dense OR sparse, not
        # both) at a low rank scores close to 1/rrf_k. Below this, a
        # result is more "best of a bad lot" than actually relevant.
        self.min_score = min_score

    def _rrf_score(self, rank):

        return 1 / (self.rrf_k + rank)

    def retrieve(
        self,
        question,
        question_embedding,
        k=5,
        fetch_k=10
    ):

        dense_results = self.faiss.retrieve(
            question_embedding,
            k=fetch_k
        )

        sparse_results = self.bm25.retrieve(
            question,
            k=fetch_k
        )

        fused = {}

        # ------------------------
        # Dense Ranking
        # ------------------------

        for rank, result in enumerate(dense_results, start=1):

            doc = result["document"]

            key = (
                doc["metadata"]["source"],
                doc["metadata"]["page"],
                doc["text"]
            )

            if key not in fused:

                fused[key] = {

                    "document": doc,

                    "score": 0.0

                }

            fused[key]["score"] += self._rrf_score(rank)

        # ------------------------
        # Sparse Ranking
        # ------------------------

        for rank, result in enumerate(sparse_results, start=1):

            doc = result["document"]

            key = (
                doc["metadata"]["source"],
                doc["metadata"]["page"],
                doc["text"]
            )

            if key not in fused:

                fused[key] = {

                    "document": doc,

                    "score": 0.0

                }

            fused[key]["score"] += self._rrf_score(rank)

        results = sorted(
            fused.values(),
            key=lambda x: x["score"],
            reverse=True)

        results = [
            r for r in results
            if r["score"] >= self.min_score
        ]

        logger.debug("Question: %s", question)
        logger.debug("Dense: %d", len(dense_results))
        logger.debug("Sparse: %d", len(sparse_results))

        if logger.isEnabledFor(logging.DEBUG):
            for r in results[:5]:
                logger.debug(
                    "%s %s %s",
                    round(r["score"], 4),
                    r["document"]["metadata"]["source"],
                    r["document"]["metadata"]["page"]
                )

        return results[:k]