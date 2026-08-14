import faiss
import numpy as np
import pickle
import os


class VectorStore:

    def __init__(self, dimension):

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.documents = []


    def add(self, embeddings, documents):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        faiss.normalize_L2(
            embeddings
        )

        self.index.add(
            embeddings
        )

        self.documents.extend(
            documents
        )


    def search(self, embedding, k=5):

        embedding = np.asarray(
            [embedding],
            dtype="float32"
        )

        faiss.normalize_L2(
            embedding
        )

        scores, indices = self.index.search(
            embedding,
            min(k, len(self.documents))
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            results.append({

                "score": float(score),

                "document": self.documents[idx]

            })

        return results


    def save(
        self,
        index_path,
        metadata_path
    ):

        os.makedirs(
            os.path.dirname(index_path),
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            index_path
        )

        with open(
            metadata_path,
            "wb"
        ) as f:

            pickle.dump(
                self.documents,
                f
            )


    @classmethod
    def load(
        cls,
        index_path,
        metadata_path
    ):

        if (
            not os.path.exists(index_path)
            or
            not os.path.exists(metadata_path)
        ):

            return None


        index = faiss.read_index(
            index_path
        )


        with open(
            metadata_path,
            "rb"
        ) as f:

            documents = pickle.load(
                f
            )


        store = cls(
            index.d
        )

        store.index = index

        store.documents = documents

        return store


    def append(
        self,
        embeddings,
        documents
    ):

        self.add(
            embeddings,
            documents
        )


    def __len__(self):

        return len(
            self.documents
        )