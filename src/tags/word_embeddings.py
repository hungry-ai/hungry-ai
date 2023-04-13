import numpy as np
import torchtext  # type: ignore[import]


class WordEmbedding(dict[str, np.ndarray]):
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    def __setitem__(self, word: str, vector: np.ndarray) -> None:
        if len(vector) != self.dimension:
            raise ValueError("Vector length does not match word embedding dimension")

        super().__setitem__(word, vector)

    @property
    def dimension(self) -> int:
        return self._dimension

    def distance(self, word_1: str, word_2: str) -> float:
        return np.mean((self[word_1] - self[word_2]) ** 2)


class PytorchWordEmbedding(WordEmbedding):
    def __init__(self, words: list[str], dimension: int) -> None:
        super().__init__(dimension)

        glove = torchtext.vocab.GloVe(name="6B", dim=dimension)

        for word in set(words):
            word = word.lower()

            if word in glove.stoi:
                vector = glove.vectors[glove.stoi[word]].numpy()
                self[word] = vector

        print(f"Number of words kept: {len(self)}")
