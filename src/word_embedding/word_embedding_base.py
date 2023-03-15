from abc import ABC, abstractmethod

#TODO(gnehzza): Add more comments to these functions.
class WordEmbeddingBase(ABC):

    # Returns the dimension of the word embedding.
    def dimension(self):
        pass
    
    # Returns number of words in the embedding.
    @abstractmethod
    def number_of_words(self):
        pass
    
    # Returns the vector corresponding to input word.
    @abstractmethod
    def get_word_vector(self, word: str):
        pass

    # Returns i-th vector of the embedding.
    # This is used to through all words of an embedding.
    @abstractmethod
    def get_word_vector(self, i: int):
        pass

    # Adds a word and vector pair to the word embedding.
    # The argument v should be an object that satisfies needs. 
    @abstractmethod
    def add_word_vector(self, v):
        pass