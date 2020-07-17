import torch.utils.data as data
import torch
import torch.nn as nn
from models.effnet import EfficientNet
from models.res2net import res2net50
from utils.data import GestureData
from utils.solver import Solver


def train():
    path = "/home/dhaval/I6_Gestures"

    train_data = GestureData(path, mode="train")
    val_data = GestureData(path, mode="val")
    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4)

    model = res2net50(num_classes=35)
    model = model.cuda()

    optim_args = {"lr": 1e-3, "weight_decay": 1e-3, "nesterov": True, "momentum": 0.9}
    solver = Solver(optim=torch.optim.SGD, optim_args=optim_args)
    solver.train(model, train_loader, val_loader, num_epochs=5, log_nth=5)


if __name__ == "__main__":
    train()
