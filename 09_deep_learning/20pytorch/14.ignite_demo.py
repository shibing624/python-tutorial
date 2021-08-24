# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 演示pytorch高级库ignite的使用

pip install pytorch-ignite
"""

import os

import ignite.distributed as idist
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
from torchvision.transforms import Resize,Compose, Normalize, Pad, RandomCrop, RandomHorizontalFlip, ToTensor

train_transform = Compose([
    # Pad(4),
    # RandomCrop(32),
Resize(224),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.23, 0.225)),
])

test_transform = Compose([
Resize(224),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.23, 0.225)),
])


def get_datasets(path):
    path = os.path.expanduser(path)
    train_ds = datasets.CIFAR10(root=path, train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root=path, train=False, download=False, transform=test_transform)

    return train_ds, test_ds


def get_model(name):
    if name in models.__dict__:
        fn = models.__dict__[name]
    else:
        raise RuntimeError("Unknown model name {}".format(name))

    return fn(num_classes=10)


def get_dataflow(config):
    # Get train/test datasets
    if idist.get_rank() > 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()
    train_dataset, test_dataset = get_datasets(config.get("data_path", "."))

    if idist.get_rank() == 0:
        idist.barrier()

    train_loader = idist.auto_dataloader(
        train_dataset,
        batch_size=config.get("batch_size", 512),
        num_workers=config.get("num_workers", 8),
        shuffle=True,
        drop_last=True,
    )
    config["num_iters_per_epoch"] = len(train_loader)
    test_loader = idist.auto_dataloader(
        test_dataset,
        batch_size=2 * config.get("batch_size", 512),
        num_workers=config.get("num_workers", 8),
        shuffle=False,
    )

    return train_loader, test_loader


def initialize(config):
    model = get_model(config["model"])
    # Adapt model for distributed settings if configured
    model = idist.auto_model(model)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("learning_rate", 0.1),
        momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 1e-5),
        nesterov=True,
    )
    optimizer = idist.auto_optim(optimizer)

    loss_fn = nn.CrossEntropyLoss().to(idist.device())
    le = config["num_iters_per_epoch"]
    lr_scheduler = StepLR(optimizer, step_size=le, gamma=0.9)

    return model, optimizer, loss_fn, lr_scheduler


def create_trainer(model, optimizer, loss_fn, lr_scheduler, config):
    # Define any training logic for iteration update
    def train_step(engine, batch):
        x = batch[0].to(idist.device())
        y = batch[1].to(idist.device())

        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        return loss.item()

    # Define trainer engine
    trainer = Engine(train_step)

    if idist.get_rank() == 0:
        # Add any custom handlers
        @trainer.on(Events.EPOCH_COMPLETED(every=1))
        def save_checkpoint():
            model_path = os.path.join((config.get("output_path", "output")), "checkpoint.pt")
            torch.save(model.state_dict(), model_path)

        # Add progress bar showing batch loss value
        ProgressBar().attach(trainer, output_transform=lambda x: {"batch loss": x})

    return trainer


def training(local_rank, config):
    train_loader, val_loader = get_dataflow(config)
    model, optimizer, loss_fn, lr_scheduler = initialize(config)

    # model trainer and evaluator
    trainer = create_trainer(model, optimizer, loss_fn, lr_scheduler, config)
    evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy()}, device=idist.device())

    # model evaluation every 3 epochs and show results
    @trainer.on(Events.EPOCH_COMPLETED(every=3))
    def evaluate_model():
        state = evaluator.run(val_loader)
        if idist.get_rank() == 0:
            print(state.metrics)

    # tensorboard experiment tracking
    if idist.get_rank() == 0:
        tb_logger = common.setup_tb_logging(
            config.get("output_path", "output"), trainer, optimizer, evaluators={"validation": evaluator},
        )

    trainer.run(train_loader, max_epochs=config.get("max_epochs", 3))
    if idist.get_rank() == 0:
        tb_logger.close()


if __name__ == '__main__':
    backend = None  # or "nccl", "gloo", "xla-tpu" ...
    nproc_per_node = None  # or N to spawn N processes
    config = {
        "model": "alexnet",
        # "model": "resnet18",
        "dataset": "cifar10",
        "data_path": "~/.pytorch/datasets/CIFAR10",
    }

    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as p:
        p.run(training, config)
