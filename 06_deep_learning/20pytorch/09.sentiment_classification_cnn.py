# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: sentiment classification
"""

# 本notebook参考了https://github.com/bentrevett/pytorch-sentiment-analysis
#
# 在这份notebook中，我们会用PyTorch模型和TorchText再来做情感分析(检测一段文字的情感是正面的还是负面的)。我们会使用[IMDb 数据集](http://ai.stanford.edu/~amaas/data/sentiment/)，即电影评论。
#
# 模型从简单到复杂，我们会依次构建：
# - Word Averaging模型
# - RNN/LSTM模型
# - CNN模型(now)
import random

import torch
import torchtext
from torchtext import datasets

SEED = 1
torch.manual_seed(SEED)
random_state = random.seed(SEED)

TEXT = torchtext.data.Field()
LABEL = torchtext.data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

print(vars(train_data.examples[0]))

train_data, valid_data = train_data.split(random_state=random_state)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

TEXT.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)

print(f'Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}')
print(f'Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}')

print('most common vocab: ', TEXT.vocab.freqs.most_common(10))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)

BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    shuffle=True
)

# Word Average model
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_fileters, filter_sizes,
                 output_dim, pad_idx, dropout=0.5):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_fileters,
                      kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_fileters, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(1, 0)  # batch_size, sent_len
        x = self.embedding(x)  # batch_size, sent_len, emb_dim
        x = x.unsqueeze(1)  # batch_size, 1, sent_len, emb_dim
        # conv
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # max pool
        x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]  # batch_size, n_filters
        x = self.drop(torch.cat(x, dim=1))  # batch_size, n_filters * len(filter_sizes)
        x = self.fc(x)
        return x


VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 50
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model = CNN(VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, PAD_IDX, DROPOUT).to(device)

print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'Model has {count_parameters(model):,} trainable parameters')

# pretrained_embeddings = TEXT.vocab.vectors
# model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCEWithLogitsLoss().to(device)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train_one_batch(model, data, optimizer, loss_fn):
    epoch_loss = 0.
    epoch_acc = 0.
    model.train()
    for batch in data:
        text, label = batch.text.to(device), batch.label.to(device)
        optimizer.zero_grad()
        preds = model(text).squeeze(1)
        loss = loss_fn(preds, label)
        acc = binary_accuracy(preds, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(data), epoch_acc / len(data)


def evaluate(model, data, loss_fn):
    epoch_loss = 0.
    epoch_acc = 0.
    model.eval()
    with torch.no_grad():
        for batch in data:
            text = batch.text.to(device)
            label = batch.label.to(device)
            preds = model(text).squeeze(1)
            loss = loss_fn(preds, label)
            acc = binary_accuracy(preds, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(data), epoch_acc / len(data)


import time


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


NUM_EPOCHS = 5
MODEL_PATH = 'cnn_model.pth'


def train():
    best_val_loss = 1e3
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_one_batch(model, train_iter, optimizer, loss_fn)
        valid_loss, valid_acc = evaluate(model, valid_iter, loss_fn)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), MODEL_PATH)
        print(f"Epoch: {epoch + 1:02}/{NUM_EPOCHS} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.4f}%")
        print(f"\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.4f}%")


# train()

model.load_state_dict(torch.load(MODEL_PATH))
# test_loss, test_acc = evaluate(model, test_iter, loss_fn)
# print(f"\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.4f}%")


def predict_sentiment(sentence):
    tokens = [token for token in sentence.split()]
    indexed = [TEXT.vocab.stoi[t] for t in tokens]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    pred = torch.sigmoid(model(tensor))
    return pred.item()


print(predict_sentiment('This film is terrible'))

print(predict_sentiment('This film is good'))
