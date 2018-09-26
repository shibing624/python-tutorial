# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#define-the-network

# define network
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


network = Network()
print(network)

params = list(network.parameters())
print(len(params))
print(params[0].size())

input = Variable(torch.randn(1, 1, 32, 32))
out = network(input)
print(out)

network.zero_grad()
out.backward(torch.randn(1, 10))

# loss function
output = network(input)
target = Variable(torch.randn(1, 10))
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print('loss:', loss)

print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

network.zero_grad()
print(network.conv1.bias.grad)

loss.backward()
print(network.conv1.bias.grad)

# update weights
learning_rate = 0.1
for i in network.parameters():
    i.data.sub_(i.grad.data * learning_rate)

print(i.data)

import torch.optim as optim

optimizer = optim.SGD(network.parameters(), lr=0.01)
optimizer.zero_grad()
output = network(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # does the update

print(loss)