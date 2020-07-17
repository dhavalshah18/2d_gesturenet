import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import os

from utils.data import GestureData
from models.res2net import res2net50


def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    path = "/home/dshah/I6_Gestures"
    saved_model_path = "/home/dshah/2dgesturenet/res2net50.pth"

    test_data = GestureData(path, mode="test")
    test_loader = data.DataLoader(test_data, batch_size=16, shuffle=True, num_workers=4)

    model = nn.DataParallel(res2net50(num_classes=35))
    model = model.cuda()
    model.load_state_dict(torch.load(saved_model_path))

    model.eval()
    test_acc = []
    log_nth = 10
    for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda().to(torch.float), targets.cuda().to(torch.long)
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        acc = np.mean((targets == preds).detach().cpu().numpy())
        test_acc.append(acc)

        if i % log_nth == 0:
            avg_acc = sum(test_acc)/len(test_acc)
            print("Average accuracy: %.3f" % avg_acc)


    mean_test_acc = sum(test_acc)/len(test_acc)
    print("Mean accuracy over full test set: %.3f" % mean_test_acc)


if __name__ == "__main__":
    test()
