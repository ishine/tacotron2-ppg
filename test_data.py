"""Make sure data load correctly and have the correct types.
"""

import os
#import time
#import argparse
#import math
#from numpy import finfo
import torch
#from torch.utils.data import DataLoader

#from model import Tacotron2
from data_utils import PPGMelLoader, PPGMelCollate
from train import prepare_dataloaders, load_model
from hparams import create_hparams
print("All imported.")

print("==================================")
hparams = create_hparams()
base_loader = PPGMelLoader(hparams)
pair = base_loader.__getitem__(5)
print("PPG shape:", pair[0].shape)
print("Mel shape:", pair[1].shape)
print("==================================")

model = load_model(hparams)
train_loader, valset, collate_fn = prepare_dataloaders(hparams)

print("==================================")
batch = next(iter(train_loader))
print("batch[0].shape:", batch[0].shape)
print("batch[1].shape:", batch[1].shape)
print("==================================")
model.zero_grad()
x, y = model.parse_batch(batch)
print("x shape:", x.shape)
print("y shape:", y.shape)
y_pred = model(x)
print(y_pred.shape)