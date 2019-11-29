import torch
import numpy as np
from embedding import Embedder
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from load_data import prepare_data
from transformer import Transformer

clip = 1
lang1 = "eng"
lang2 = "vie"
SOS_token = 0
EOS_token = 1
PAD_token = 2
batch_size = 128
criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
device = torch.device("cpu")


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


input_lang, output_lang, _ = prepare_data(lang1, lang2, 40)

d_model = 128
heads = 8
N = 6
src_vocab = input_lang.n_words
trg_vocab = output_lang.n_words
en_weight_matrix = Embedder.initial_weights_matrix("word_vector/glove.6B.300d.txt", input_lang, 300)
de_weight_matrix = Embedder.initial_weights_matrix("word_vector/vn_word2vec_300d.txt", input_lang, 300)
src_vocab = input_lang.n_words
trg_vocab = output_lang.n_words

model = Transformer(src_vocab, trg_vocab, d_model, N, heads, device, en_weight_matrix, de_weight_matrix)
model.load_state_dict(torch.load("model/transformer.pt", map_location=device))


def translate(model, sentence, lang_input, lang_output, max_len=80):
    model.eval()

    src = tensor_from_sentence("en", lang_input, sentence, len(sentence))

    src_mask = (src != PAD_token).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([SOS_token])

    for i in range(1, max_len):

        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0)

        out = model.out(model.decoder(outputs[:i].unsqueeze(0),
                                      e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == EOS_token:
            break

    return ' '.join(
        [lang_output.index2word[ix.item()] for ix in outputs[:i]]
    )


while True:
    try:
        input_sentence = input("\nPlease enter a string: ")
        output_sentence = translate(model, input_sentence.lower(), input_lang, output_lang)
        print("\nTranslated string: ", output_sentence)
    except:
        print("error translate")
