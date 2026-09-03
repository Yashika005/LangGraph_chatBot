from sentence_transformers import SentenceTransformer

class EmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, input): 

        return self.model.encode(input).tolist()

    def name(self):
        return "sentence-transformers-all-MiniLM-L6-v2"