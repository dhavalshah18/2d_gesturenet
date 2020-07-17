import cv2 as cv
import torch
import torch.nn as nn
import numpy as np
import pathlib
import os

from models.res2net import res2net50

gesture_ids = {}


def create_gesture_ids():
    with open("gesture_ids.txt", "r") as f:
        gestures = f.readlines()
    global gesture_ids

    for i in range(0, 35):
        line = gestures[i]
        gesture_num = line.split(" = ")[0]
        gesture_label = line.split(" = ")[-1]

        gesture_ids[gesture_num] = gesture_label


def vid_2_imgs(vid_path):
    vid_in = cv.VideoCapture(vid_path)
    fps = vid_in.get(cv.CAP_PROP_FPS)
    success, img = vid_in.read()
    imgs = []

    while success:
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        rgb_img = rgb_img.transpose((2, 0, 1))

        imgs.append(rgb_img)

        success, image = vid_in.read()

    vid_in.release()
    img_shape = tuple(img[0][0].shape)
    return imgs, fps, img_shape


def create_vid(gestures, vid_out_name, fps, size):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    vid_out = cv.VideoWriter(vid_out_name, fourcc, fps, size)

    font = cv.FONT_HERSHEY_SIMPLEX
    x = size[0] - 40
    y = size[1] - 40
    pt = (x, y)
    scale = 50
    global gesture_ids

    for i in gestures:
        curr_gesture = gesture_ids[(str(i + 1))]

        img = np.zeros(size)
        img = cv.putText(img, curr_gesture, pt, font, scale, (255, 255, 255))
        vid_out.write(img)

    vid_out.release()


def vid_classification(model, vid_path):
    frames, fps, img_shape = vid_2_imgs(vid_path)

    model.eval()
    gestures = []
    for i in frames:
        i = torch.from_numpy(i).cuda().to(torch.float)
        i = i.unsqueeze(0)

        outputs = model(i)

        _, pred = torch.max(outputs, 1)
        gestures.append(pred)

    return gestures, fps, img_shape


def vid_infer():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

    vid_in_path = input("Full video path: ")
    vid_out_name = input("Output video name: ")

    saved_model_path = input("Full saved model path: ")
    model = res2net50(num_classes=35)
    model.cuda()
    model.load_state_dict(torch.load(saved_model_path))

    gestures, fps, img_shape = vid_classification(model, vid_in_path)

    create_vid(gestures, vid_out_name, fps, img_shape)


if __name__ == "__main__":
    vid_infer()
