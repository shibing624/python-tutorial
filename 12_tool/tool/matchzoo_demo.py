# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""


import torch
import matchzoo as mz

ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.MeanAveragePrecision()
]

train_pack = mz.datasets.wiki_qa.load_data('train', task=ranking_task)
valid_pack = mz.datasets.wiki_qa.load_data('dev', task=ranking_task)

preprocessor = mz.models.ArcI.get_default_preprocessor()
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)

trainset = mz.dataloader.Dataset(
    data_pack=train_processed,
    mode='pair',
    num_dup=1,
    num_neg=4
)
validset = mz.dataloader.Dataset(
    data_pack=valid_processed,
    mode='point'
)

padding_callback = mz.models.ArcI.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    batch_size=32,
    stage='train',
    callback=padding_callback
)
validloader = mz.dataloader.DataLoader(
    dataset=validset,
    batch_size=32,
    stage='dev',
    callback=padding_callback
)

model = mz.models.ArcI()
model.params['task'] = ranking_task
model.guess_and_fill_missing_params()
model.build()
optimizer = torch.optim.Adam(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    epochs=1
)

trainer.run()

