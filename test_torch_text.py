import torchtext
from ast import literal_eval
from torchtext.data import Field, BucketIterator, TabularDataset, Dataset
from torchtext import datasets

# set up fields
TEXT = Field(lower=True, include_lengths=True, batch_first=True)
LABEL = Field(sequential=False)

# make splits for data
train, test = datasets.IMDB.splits(TEXT, LABEL)

print("train ", train)

# # build the vocabulary
# TEXT.build_vocab(train)
# LABEL.build_vocab(train)
#
# # make iterator for splits
# train_iter, test_iter = data.BucketIterator.splits(
#     (train, test), batch_size=3, device=0)