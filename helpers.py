import time
import math
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
from nltk.translate.bleu_score import sentence_bleu

plt.switch_backend('agg')


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def mask_3d(inputs, seq_len, mask_value=0.):
    batches = inputs.size()[0]
    assert batches == len(seq_len)
    max_idx = max(seq_len)
    for n, idx in enumerate(seq_len):
        if idx < max_idx.item():
            if len(inputs.size()) == 3:
                inputs[n, idx.int():, :] = mask_value
            else:
                assert len(inputs.size()) == 2, "The size of inputs must be 2 or 3, received {}".format(inputs.size())
                inputs[n, idx.int():] = mask_value
    return inputs


def skip_add_pyramid(x, seq_len, skip_add="add"):
    if len(x.size()) == 2:
        x = x.unsqueeze(0)
    x_len = x.size()[1] // 2
    even = x[:, torch.arange(0, x_len*2-1, 2).long(), :]
    odd = x[:, torch.arange(1, x_len*2, 2).long(), :]
    if skip_add == "add":
        return (even+odd) / 2, (seq_len / 2).int()
    else:
        return even, (seq_len / 2).int()


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def bleu_score(reference, candidate):
    blue_1 = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))
    blue_2 = sentence_bleu([reference], candidate, weights=(0.5, 0.5, 0, 0))
    blue_3 = sentence_bleu([reference], candidate, weights=(0.33, 0.33, 0.33, 0))
    blue_4 = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
    return blue_1, blue_2, blue_3, blue_4


def blue_score_batch(preds, targets, vocab, batch_first=True):
    preds = tensors2words(preds, vocab, batch_first)
    targets = tensors2words(targets, vocab, batch_first)
    score_1 = 0
    score_2 = 0
    score_3 = 0
    score_4 = 0
    for pred, target in zip(preds, targets):
        blue_1, blue_2, blue_3, blue_4 = bleu_score(pred.split(" "), target.split(" "))
        score_1 += blue_1
        score_2 += blue_2
        score_3 += blue_3
        score_4 += blue_4
    return score_1 / len(preds), score_2 / len(preds), score_3 / len(preds), score_4 / len(preds)


def tensors2words(vectors, vocab, batch_first=True):
    sentences = []
    with torch.no_grad():
        if batch_first:
            if len(vectors.shape) == 3:
                values, indexes = torch.max(vectors, -1)
                for i in range(len(indexes)):
                    sentence = " "
                    for j in range(len(indexes[i])):
                        if values[i][j].item() == 0:
                            word = vocab.index2word[0]
                        else:
                            word = vocab.index2word[indexes[i][j].item()]
                        if word != "SOS" and word != "EOS" and word != "PAD":
                            sentence += (word + " ")
                    sentences.append(sentence)
            else:
                for i in range(len(vectors)):
                    sentence = " "
                    for j in range(len(vectors[i])):
                        word = vocab.index2word[vectors[i][j].item()]
                        if word != "SOS" and word != "EOS" and word != "PAD":
                            sentence += (word + " ")
                    sentences.append(sentence)
    return sentences
