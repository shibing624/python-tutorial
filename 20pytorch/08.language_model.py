#!/usr/bin/env python
# coding: utf-8

# #语言模型
# 
# - build language model
# - torchtext vocab
# - torch.nn 
# 	- lstm
# 	- rnn
# 	- gru
# 	- linear
# - gradient clipping
# - save and read model

# In[2]:


import random

import numpy as np
import torch
import torchtext

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

MAX_VOCAB_SIZE = 10000
BATCH_SIZE = 32
EMBEDDING_SIZE = 20

# In[3]:


TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path='..',
                                                                     train="./data/nietzsche.txt",
                                                                     validation="./data/nietzsche.txt",
                                                                     test="./data/nietzsche.txt",
                                                                     text_field=TEXT)

# In[4]:


TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
print(f"vocab size {len(TEXT.vocab)}")
VOCAB_SIZE = len(TEXT.vocab)

# In[22]:


print(TEXT.vocab.itos[:10])

# In[23]:


print(TEXT.vocab.stoi["mother"])

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test),
                                                                     batch_size=BATCH_SIZE,
                                                                     device=device,
                                                                     bptt_len=50,
                                                                     repeat=False,
                                                                     shuffle=True)

# In[28]:


batch = next(iter(train_iter))

print(" ".join(TEXT.vocab.itos[i] for i in batch.text[:, 0].data.cpu()))
print()
print(" ".join(TEXT.vocab.itos[i] for i in batch.target[:, 0].data.cpu()))

# ## 定义模型


import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, nlayers=1, dropout=0.5):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_size, hidden_size, nlayers, dropout=dropout)
        else:
            raise ValueError("model type:['LSTM', 'GRU]")
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(dropout)

        self.init_weights()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.nlayers = nlayers

    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, x, hidden):
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.nlayers, batch_size, self.hidden_size), requires_grad=requires_grad),
                    weight.new_zeros((self.nlayers, batch_size, self.hidden_size), requires_grad=requires_grad))
        else:
            return weight.new_zeros((self.nlayers, batch_size, self.hidden_size), requires_grad=requires_grad)


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001


def evaluate(model, data):
    model.eval()
    total_loss, total_count = 0., 0.
    it = iter(data)
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text.to(device), batch.target.to(device)
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss += loss.item() * np.multiply(*data.size())
    loss = total_loss / total_count
    model.train()
    return loss


GRAD_CLIP = 1.
NUM_EPOCHS = 2
MODEL_PATH = 'lm_best.pth'
model = RNNModel('LSTM', VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 2, dropout=0.5).to(device)
print(model)


def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    val_losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        it = iter(train_iter)
        hidden = model.init_hidden(BATCH_SIZE)
        for i, batch in enumerate(it):
            data, target = batch.text.to(device), batch.target.to(device)
            hidden = repackage_hidden(hidden)
            model.zero_grad()
            output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            if i % 1000 == 0:
                print('epoch', epoch, 'iter', i, 'loss', loss.item())

            if i % 10000 == 0:
                val_loss = evaluate(model, val_iter)
                if len(val_losses) == 0 or val_loss < min(val_losses):
                    print('best model, val loss:', val_loss)
                    torch.save(model.state_dict(), MODEL_PATH)
                else:
                    scheduler.step()
                val_losses.append(val_loss)


# train()

model.load_state_dict(torch.load(MODEL_PATH))

val_loss = evaluate(model, val_iter)
print('val perplexity:', np.exp(val_loss))

test_loss = evaluate(model, test_iter)
print('test perplexity:', np.exp(val_loss))


def generate_text():
    hidden = model.init_hidden(1)
    x = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
    words = []
    for i in range(100):
        output, hidden = model(x, hidden)
        word_weights = output.squeeze().exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        x.fill_(word_idx)
        word = TEXT.vocab.itos[word_idx]
        words.append(word)
    result = ' '.join(words)
    return result


text = generate_text()
print(text)
