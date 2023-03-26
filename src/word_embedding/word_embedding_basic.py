from word_embedding_base import WordEmbeddingBase

#TODO(gnehzza): Add more comments to these functions.
class WordEmbeddingBasic(WordEmbeddingBase):
    def __init__(self, dimension = 5):
        self.word_dict = dict()
        self.words = []
        self.dimension = dimension

    # Returns the dimension of the word embedding.
    def get_dimension(self):
        return self.dimension
    
    # Returns number of words in the embedding.
    def number_of_words(self):
        return len(self.word_dict)
    
    # Returns the vector corresponding to input word.
    def get_word_vector(self, word: str):
        if word in self.word_dict:
            return (word, self.word_dict[word])
        else: return None

    # Returns i-th vector of the embedding.
    # This is used to loop through all words of an embedding.
    def get_ith_word_vector(self, i: int):
        if i < len(self.words) and i >= 0:
            return self.words[i]
        else: return None

    # Adds a word and vector pair to the word embedding.
    # The argument v should be an object that satisfies needs. 
    # Returns 1 if successfully added and returns 0 otherwise.
    def add_word_vector(self, v):
        (word, vec) = v 
        if self.dimension != len(vec): return 0
        self.word_dict[word] = vec
        self.words.append(v)
        return 1