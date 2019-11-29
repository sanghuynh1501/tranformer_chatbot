import time
import math

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from embedding import Embedder
from helpers import blue_score_batch
from load_data import data_generator
from transformer import Transformer

clip = 1
lang1 = "eng"
lang2 = "vie"
SOS_token = 0
EOS_token = 1
PAD_token = 2
batch_size = 32
criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(type, lang, sentence, max_length):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    if len(indexes) == max_length:
        return torch.tensor(indexes, dtype=torch.long, requires_grad=False).unsqueeze(0)
    while len(indexes) < max_length:
        indexes.append(PAD_token)
    if type == "de":
        indexes = [SOS_token] + indexes
    return torch.tensor(indexes, dtype=torch.long, requires_grad=False).unsqueeze(0)


def tensors_from_pair(input_lang, output_lang, pair, max_length_en, max_length_de):
    input_tensor = tensor_from_sentence("en", input_lang, pair[0], max_length_en)
    target_tensor = tensor_from_sentence("de", output_lang, pair[1], max_length_de)
    return input_tensor, target_tensor


language, total_data = data_generator(batch_size, 20, device)
train_data, test_data, y_train, y_test = train_test_split(total_data, np.zeros(len(total_data,)), test_size=0.1, random_state=42)

d_model = 128
heads = 8
N = 6
src_vocab = language.n_words
trg_vocab = language.n_words
en_weight_matrix = Embedder.initial_weights_matrix("word_vector/glove.6B.300d.txt", language, 300)
src_vocab = language.n_words
trg_vocab = language.n_words

model = Transformer(src_vocab, trg_vocab, d_model, N, heads, device, en_weight_matrix, en_weight_matrix)
try:
    model.load_state_dict(torch.load("model/transformer.pt", map_location=device))
    model.eval()
except:
    print("no weights exist")
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


def create_masks(src, trg):
    input_seq = src
    input_pad = PAD_token
    input_msk = (input_seq != input_pad).unsqueeze(1)

    target_seq = trg
    target_pad = PAD_token
    target_msk = (target_seq != target_pad).unsqueeze(1)
    size = target_seq.size(1)  # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), 1).astype('uint8')
    nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(device)
    target_msk = target_msk & nopeak_mask

    return input_msk.to(device), target_msk.to(device)


def train():
    model.train()
    train_total_loss = 0

    with tqdm(total=len(train_data)) as pbar:
        for i, batch in enumerate(train_data):
            src = batch["src"]
            trg = batch["trg"]
            # the French sentence we input has all words except
            # the last, as it is using each word to predict the next

            trg_input = trg[:, :-1]

            # the words we are trying to predict

            targets = trg[:, 1:].contiguous().view(-1)

            # create function to make masks using mask code above

            src_mask, trg_mask = create_masks(src, trg_input)

            preds = model(src, trg_input, src_mask, trg_mask)

            optim.zero_grad()

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                                   targets, ignore_index=PAD_token)
            loss.backward()
            optim.step()

            train_total_loss += loss.item()

            pbar.update(1)

    return train_total_loss / len(train_data)


def evaluate():
    model.eval()
    test_total_loss = 0
    test_total_score_1 = 0
    test_total_score_2 = 0
    test_total_score_3 = 0
    test_total_score_4 = 0

    with torch.no_grad():
        with tqdm(total=len(test_data)) as pbar:
            for i, batch in enumerate(test_data):
                src = batch["src"]
                trg = batch["trg"]
                # the French sentence we input has all words except
                # the last, as it is using each word to predict the next

                trg_input = trg[:, :-1]

                # the words we are trying to predict

                targets = trg[:, 1:].contiguous().view(-1)

                # create function to make masks using mask code above

                src_mask, trg_mask = create_masks(src, trg_input)

                preds = model(src, trg_input, src_mask, trg_mask)

                score_1, score_2, score_3, score_4 = blue_score_batch(preds, trg, output_lang, True)
                test_total_score_1 += score_1
                test_total_score_2 += score_2
                test_total_score_3 += score_3
                test_total_score_4 += score_4

                loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                                       targets, ignore_index=PAD_token)

                test_total_loss += loss.item()

                pbar.update(1)

    return test_total_loss / len(test_data), test_total_score_1 / len(test_data), test_total_score_2 / len(test_data), test_total_score_3 / len(test_data), test_total_score_4 / len(test_data)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_model(epochs):

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), "model/transformer.pt")

        valid_loss, score_1, score_2, score_3, score_4 = evaluate()

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print(f'\t BLEU 1: {score_1:.3f} | BLEU 2: {score_2:.3f} | BLEU 3: {score_3:.3f} | BLEU 4: {score_4:.3f}')


train_model(150)
