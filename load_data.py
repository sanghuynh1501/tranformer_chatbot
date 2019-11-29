"""
A corpus parser for preparing data for a tensorflow chatbot
"""
import re
import torch
import unicodedata
from ast import literal_eval


DELIM = ' +++$+++ '
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2


class CornellMovieCorpusProcessor:

    def __init__(self, lines='movie_lines.txt', conversations='movie_conversations.txt'):
        self.movie_lines_filepath = lines
        self.movie_conversations = conversations

    def get_id2line(self):
        id2line = {}
        id_index = 0
        text_index = 4
        with open(self.movie_lines_filepath, 'r', encoding='iso-8859-1') as f:
            for line in f:
                items = line.split(DELIM)
                if len(items) == 5:
                    line_id = items[id_index]
                    dialog_text = items[text_index]
                    id2line[line_id] = dialog_text
        return id2line

    def get_conversations(self):
        conversation_ids_index = -1
        conversations = []
        with open(self.movie_conversations, 'r', encoding='iso-8859-1') as f:
            for line in f:
                items = line.split(DELIM)
                conversation_ids_field = items[conversation_ids_index]
                conversation_ids = literal_eval(conversation_ids_field)  # evaluate as a python list
                conversations.append(conversation_ids)
        return conversations

    def get_question_answer_set(self, id2line, conversations):

        questions = []
        answers = []

        # This uses a simple method in an attempt to gather question/answers
        for conversation in conversations:
            if len(conversation) % 2 != 0:
                conversation = conversation[:-1]  # remove last item

            for idx, line_id in enumerate(conversation):
                if idx % 2 == 0:
                    questions.append(id2line[line_id])
                else:
                    answers.append(id2line[line_id])

        return questions, answers


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<sos>": 0, "<eos>": 1, "<pad>": 2}
        self.word2count = {}
        self.index2word = {0: "<sos>", 1: "<eos>", 2: "<pad>"}
        self.n_words = 3  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = s.replace(" .", "")
    s = s.replace(".", "")
    return s


def filter_pair(p, max_length):
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length


def filter_pairs(pairs, max_length):
    return [pair for pair in pairs if filter_pair(pair, max_length)]


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(type, lang, sentence, max_length):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    if len(indexes) == max_length:
        return torch.tensor(indexes, dtype=torch.long, requires_grad=False).unsqueeze(0)
    while len(indexes) < max_length:
        indexes.append(PAD_TOKEN)
    if type == "de":
        indexes = [SOS_TOKEN] + indexes
    return torch.tensor(indexes, dtype=torch.long, requires_grad=False).unsqueeze(0)


def tensors_from_pair(input_lang, output_lang, pair, max_length_en, max_length_de):
    input_tensor = tensor_from_sentence("en", input_lang, pair[0], max_length_en)
    target_tensor = tensor_from_sentence("de", output_lang, pair[1], max_length_de)
    return input_tensor, target_tensor


def load_data(max_length):
    pairs = []
    language = Lang("english")

    processor = CornellMovieCorpusProcessor("conversation/movie_lines.txt", "conversation/movie_conversations.txt")
    print('Collection line-ids...')
    id2lines = processor.get_id2line()
    print('Collection conversations...')
    conversations = processor.get_conversations()
    print('Preparing question/answer sets...')
    questions, answers = processor.get_question_answer_set(id2lines, conversations)

    for question, answer in zip(questions, answers):
        pairs.append([question, answer])

    pairs = filter_pairs(pairs, max_length)
    for pair in pairs:
        language.add_sentence(pair[0])
        language.add_sentence(pair[1])

    print("len(pairs) ", len(pairs))
    print("Counted words:")
    print(language.n_words)

    return language, pairs


def data_generator(batch_size, max_length, device):
    total_data = []
    language, pairs = load_data(max_length)
    for i in range(0, len(pairs), batch_size):
        if i + batch_size < len(pairs):
            batch = pairs[i: i + batch_size]
        else:
            batch = pairs[len(pairs) - batch_size: len(pairs)]

        max_length_en = 0
        max_length_de = 0
        for idx, pair in enumerate(batch):
            if len(pair[0].split(" ")) > max_length_en:
                max_length_en = len(pair[0].split(" "))
            if len(pair[1].split(" ")) > max_length_de:
                max_length_de = len(pair[1].split(" "))
        en_tensor, de_tensor = tensors_from_pair(language, language, batch[0], max_length_en + 1, max_length_de + 2)
        for idx, pair in enumerate(batch[1:]):
            en, de = tensors_from_pair(language, language, pair, max_length_en + 1, max_length_de + 2)
            en_tensor = torch.cat((en_tensor, en), 0)
            de_tensor = torch.cat((de_tensor, de), 0)
        total_data.append({
            "src": en_tensor.to(device),
            "trg": de_tensor.to(device)
        })
    return language, total_data

