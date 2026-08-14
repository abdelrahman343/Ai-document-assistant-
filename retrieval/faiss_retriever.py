class FaissRetriever:

    def __init__(self, vector_store):

        self.vector_store = vector_store

    def retrieve(self, query_embedding, k=5):

        return self.vector_store.search(
            query_embedding,
            k=k
        )