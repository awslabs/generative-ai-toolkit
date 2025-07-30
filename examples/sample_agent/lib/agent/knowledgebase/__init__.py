from dataclasses import dataclass

import faiss

_DEFAULT_MODEL = None


def load_default_model():
    global _DEFAULT_MODEL  # noqa: PLW0603
    if _DEFAULT_MODEL is None:
        print("Importing SentenceTransformer ...")
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        _DEFAULT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _DEFAULT_MODEL


@dataclass
class Document:
    text: str
    metadata: dict


class FaissKnowledgeBase:
    def __init__(self, model=None, dimension=None):
        if model:
            self.model = model
        else:
            self.model = load_default_model()
        dimension = dimension or self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: list[Document] = []

    # def add_documents(self, documents: list[Document]):
    #     vec = self.model.encode([d.text for d in documents])
    #     self.index.add(vec)  # type: ignore
    #     self.documents.extend(documents)

    def add_documents(self, documents: list[Document]):
        if not documents:
            return

        vec = self.model.encode([d.text for d in documents])

        # Check if index needs to be recreated with the correct dimension
        if not hasattr(self, "index") or self.index.d != vec.shape[1]:
            print(f"Recreating index with dimension {vec.shape[1]}")
            self.index = faiss.IndexFlatL2(vec.shape[1])

            # If we already have documents, we need to re-add them
            if self.documents:
                old_vecs = self.model.encode([d.text for d in self.documents])
                self.index.add(old_vecs)

        self.index.add(vec)
        self.documents.extend(documents)

    def search(self, query: str, k=5) -> list[tuple[Document, float]]:
        if not self.documents:
            return []
        vec = self.model.encode([query])
        distances, indices = self.index.search(vec, k)  # type: ignore
        similar_documents = [
            (self.documents[doc_index], float(distances[0][distance_index]))
            for distance_index, doc_index in enumerate(indices[0])
        ]
        return similar_documents
