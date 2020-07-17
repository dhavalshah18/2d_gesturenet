import torch
import torch.utils.data as data
import numpy as np
import pathlib
import PIL.Image as Image


class GestureData(data.Dataset):
    def __init__(self, root_path, mode="train"):
        super().__init__()
        self.root_path = pathlib.Path(root_path)
        self.mode = mode

        split_filename = self.root_path.joinpath(mode + ".txt")

        with open(split_filename, "r") as f:
            self.split_file = f.readlines()

    def __len__(self):
        return len(self.split_file)

    def __getitem__(self, index):
        line = self.split_file[index]

        img_path = line.split(" ")[0]
        img_label = line.split(" ")[-1]

        img = Image.open(img_path)
        img = np.asarray(img)
        img = img.transpose((2, 0, 1))

        img_label = int(img_label) - 1

        return img, img_label
