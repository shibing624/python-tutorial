# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

"""
sort-of minimal end-to-end example of handling input sequences (sentences) of variable length in pytorch
the sequences are considered to be sentences of words, meaning we then want to use embeddings and an RNN
using pytorch stuff for basically everything in the pipeline of:
dataset -> data_loader -> padding -> embedding -> packing -> lstm -> unpacking (~padding)
based mostly on: https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial
pytorch version 1.4.0
gist url: https://gist.github.com/MikulasZelinka/9fce4ed47ae74fca454e88a39f8d911a
"""

import torch
from torch.nn import Embedding, LSTM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# dataset is a list of sequences/sentences
# the elements of the sentences could be anything, as long as it can be contained in a torch tensor
# usually, these will be indices of words based on some vocabulary
# 0 is commonly reserved for the padding token, here it appears once explicitly and on purpose,
#  to check that it functions properly (= in the same way as the automatically added padding tokens)
DATA = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9],
    [4, 6, 2, 9, 0]
]
# need torch tensors for torch's pad_sequence(); this could be a part of e.g. dataset's __getitem__ instead
DATA = list(map(lambda x: torch.tensor(x), DATA))
# vocab size (for embedding); including 0 (the padding token)
NUM_WORDS = 10

SEED = 0
# for consistent results between runs
torch.manual_seed(SEED)

BATCH_SIZE = 3
EMB_DIM = 2
LSTM_DIM = 5


class MinimalDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


dataset = MinimalDataset(DATA)
# len(data) is not divisible by batch_size on purpose to verify consistency across batch sizes
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
# collate_fn is crucial for handling data points of varying length (as is the case here)
print(next(iter(data_loader)))
# I would think that we should always obtain:
# [ [1, 2, 3], [4, 5], [6, 7, 8, 9] ]
# but, without collate_fn set to identity as above, you would get:
# RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 3 and 2 in dimension 1 ...
# ¯\_(ツ)_/¯

# iterate through the dataset:
for i, batch in enumerate(data_loader):
    print(f'{i}, {batch}')

# playing around with padding (= unpacking) and packing (= unpadding)
print('padding and [un]packing')
# this always gets you the first batch of the dataset:
batch = next(iter(data_loader))

print(f'batch: \n{batch}\n')
# need to store the sequence lengths explicitly if we want to later pack the sequence:
lens = list(map(len, batch))

padded = pad_sequence(batch, batch_first=True)
print(f' [0] padded: \n{padded}\n')

# pytorch <1.1.0 does not support enforce_sorted=False and you would have to sort the sequences manually
packed = pack_padded_sequence(padded, lens, batch_first=True, enforce_sorted=False)
print(f' [1] packed: \n{packed}\n')
padded2 = pad_packed_sequence(packed, batch_first=True)
print(f' [2] padded: \n{padded2}\n')
# pad(pack(pad(x))) == pad(x) as pad() and pack() are inverse to each other
assert torch.all(torch.eq(padded2[0], padded))
##################################################


##################################################
# putting everything together: dataset - data_loader - padding - embedding - packing - lstm - unpacking (padding)
print('embedding')
batch = next(iter(data_loader))
# or:
# for batch in data_loader:

print(f'------------------------\nbatch: \n{batch}\n')
lens = list(map(len, batch))

embedding = Embedding(NUM_WORDS, EMB_DIM)
lstm = LSTM(input_size=EMB_DIM, hidden_size=LSTM_DIM, batch_first=True)

# we first have to pad, making all sequences in the batch equally long
padded = pad_sequence(batch, batch_first=True)
print(f'> pad: \n{padded}\n')

# now add the embedding dimension:
pad_embed = embedding(padded)
print(f'> pad_embed: \n{pad_embed}\n')

# pack it up to one sequence (where each element is EMB_DIM long)
pad_embed_pack = pack_padded_sequence(pad_embed, lens, batch_first=True, enforce_sorted=False)
print(f'> pad_embed_pack: \n{pad_embed_pack}\n')

# run that through the lstm
pad_embed_pack_lstm = lstm(pad_embed_pack)
print(f'> pad_embed_pack_lstm: \n{pad_embed_pack_lstm}\n')

# unpack the results (we can do that because it remembers how we packed the sentences)
# the [0] just takes the first element ("out") of the LSTM output (hidden states after each timestep)
pad_embed_pack_lstm_pad = pad_packed_sequence(pad_embed_pack_lstm[0], batch_first=True)
print(f'> pad_embed_pack_lstm_pad: \n{pad_embed_pack_lstm_pad}\n')

# however, usually, we would just be interested in the last hidden state of the lstm for each sequence,
# i.e., the [last] lstm state after it has processed the sentence
# for this, the last unpacking/padding is not necessary, as we can obtain this already by:
seq, (ht, ct) = pad_embed_pack_lstm
print(f'lstm last state without unpacking:\n{ht[-1]}')
# which is the same as
outs, lens = pad_embed_pack_lstm_pad
print(f'lstm last state after unpacking:\n'
      f'{torch.cat([outs[i, len - 1] for i, len in enumerate(lens)]).view((BATCH_SIZE, -1))}')
# i.e. the last non-masked/padded/null state
# so, you probably shouldn't unpack the sequence if you don't need to