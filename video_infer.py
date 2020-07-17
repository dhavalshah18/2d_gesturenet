import cv2 as cv
import torch
import torch.nn as nn
import numpy as np
import pathlib
import os

from models.res2net import res2net50


def vid_2_imgs(vid_path):
    vid_in = cv.VideoCapture(vid_path)
    success, img = vid_in.read()
    imgs = []

    while success:
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        rgb_img = rgb_img.transpose((2, 0, 1))

        imgs.append(rgb_img)

        success, image = vid_in.read()

    vid_in.release()

    return imgs


def create_vid(gestures, vid_out_name):
    pass


def vid_classification(model, vid_path):
    frames = vid_2_imgs(vid_path)

    model.eval()
    gestures = []
    for i in frames:
        i = torch.from_numpy(i).cuda().to(torch.float)
        i = i.unsqueeze(0)

        outputs = model(i)

        _, pred = torch.max(outputs, 1)
        gestures.append(pred)

    return gestures


def vid_infer():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

    vid_in_path = input("Full video path: ")
    vid_out_name = input("Output video name: ")

    saved_model_path = input("Full saved model path: ")
    model = res2net50(num_classes=35)
    model.cuda()
    model.load_state_dict(torch.load(saved_model_path))

    gestures = vid_classification(model, vid_in_path)

    create_vid(gestures, vid_out_name)


if __name__ == "__main__":
    vid_infer()
