from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    # Loading a SentenceTransformer model is expensive (seconds, not
    # milliseconds). Previously a new one was instantiated every time
    # EmbeddingModel() was called - once per uploaded file, and again
    # every time QAService or a summary/questions button was used.
    # Caching it at the class level means the model loads exactly
    # once per process, no matter how many EmbeddingModel() instances
    # get created.
    _model = None
    _model_name = None

    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):

        if (
            EmbeddingModel._model is None
            or EmbeddingModel._model_name != model_name
        ):

            EmbeddingModel._model = SentenceTransformer(model_name)
            EmbeddingModel._model_name = model_name

        self.model = EmbeddingModel._model

    def encode(self, texts):

        return self.model.encode(
            texts,
            convert_to_numpy=True
        )