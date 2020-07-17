import torch.utils.data as data
import torch
import torch.nn as nn
import os
from models.effnet import EfficientNet
from models.res2net import res2net50
from utils.data import GestureData
from utils.solver import Solver


def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
    path = "/home/dshah/I6_Gestures"

    train_data = GestureData(path, mode="train")
    val_data = GestureData(path, mode="val")
    train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4)

    model = nn.DataParallel(res2net50(num_classes=35))
    model = model.cuda()

    optim_args = {"lr": 1e-4, "weight_decay": 1e-3}
    solver = Solver(optim=torch.optim.Adam, optim_args=optim_args)
    solver.train(model, train_loader, val_loader, num_epochs=10, log_nth=50)

    name = "res2net50.pth"
    torch.save(model.state_dict(), name)

    
if __name__ == "__main__":
    train()
