from rank_bm25 import BM25Okapi
import re


class BM25Retriever:

    def __init__(self):

        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None

    def tokenize(self, text):

        text = text.lower()

        return re.findall(r"\w+", text)

    def add(self, documents):

        self.documents = documents

        self.tokenized_docs = [

            self.tokenize(
                doc["text"]
            )

            for doc in documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )

    def retrieve(self, query, k=5):

        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked = sorted(

            zip(scores, self.documents),

            key=lambda x: x[0],

            reverse=True

        )

        results = []

        for score, doc in ranked[:k]:

            results.append({

                "score": float(score),

                "document": doc

            })

        return results