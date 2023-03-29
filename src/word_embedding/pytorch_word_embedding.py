import torch
import torchtext

# Load the pre-trained word embeddings
glove = torchtext.vocab.GloVe(name='6B', dim=100)

# Get the word embedding for the word "dog"
word = "dog"
word_embedding = glove.vectors[glove.stoi[word]]

# Verify that the word embedding is a PyTorch tensor
print(type(word_embedding)) # Output: <class 'torch.Tensor'>

# Convert the word embedding to a numpy array
word_embedding_np = word_embedding.numpy()
print(word_embedding_np) # Output: <class 'numpy.ndarray'>