import numpy as np
from sentence_transformers import SentenceTransformer


def embed_text(text: list[str], model_name="all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(text)
    # return np.random.rand(len(text), 4)
