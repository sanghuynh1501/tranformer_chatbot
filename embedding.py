import numpy as np
import torch
from torch import nn


class Embedder(nn.Module):

    def __init__(self, weights_matrix, non_trainable=True):
        super().__init__()
        self.non_trainable = non_trainable
        self.embedding, num_embeddings, embedding_dim = self.create_emb_layer(weights_matrix)

    @staticmethod
    def initial_word_vector(path):
        idx = 0
        words = []
        vectors = []
        word2idx = {}
        with open(path, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
        return {w: vectors[word2idx[w]] for w in words}

    @staticmethod
    def initial_weights_matrix(path, target_vocab, vector_length):
        word_vector = Embedder.initial_word_vector(path)
        matrix_len = target_vocab.n_words
        weights_matrix = torch.zeros((matrix_len, vector_length))
        for i in range(target_vocab.n_words):
            word = target_vocab.index2word[i]
            try:
                if word == "PAD":
                    weights_matrix[i] = torch.zeros((vector_length, ))
                elif word == "SOS":
                    weights_matrix[i] = torch.ones((vector_length, )) * 0.1
                elif word == "EOS":
                    weights_matrix[i] = torch.ones((vector_length, )) * 0.2
                else:
                    weights_matrix[i] = torch.tensor(word_vector[word])
            except KeyError:
                weights_matrix[i] = torch.from_numpy(np.random.normal(scale=0.6, size=(vector_length, )))

        return weights_matrix

    def create_emb_layer(self, weights_matrix):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if self.non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim

    def forward(self, inp):
        return self.embedding(inp)