import cv2 as cv
import torch
import torch.nn as nn
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt

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
    width = int(vid_in.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid_in.get(cv.CAP_PROP_FRAME_HEIGHT))
    img_shape = (width, height)
    success, img = vid_in.read()
    imgs = []

    while success:
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        rgb_img = rgb_img.transpose((2, 0, 1))

        imgs.append(rgb_img)

        success, image = vid_in.read()

    vid_in.release()
    return imgs, fps, img_shape


def create_vid(gestures, vid_out_name, fps, size):
    fourcc = cv.VideoWriter_fourcc(*'mjpg')
    vid_out = cv.VideoWriter(vid_out_name, fourcc, 1, size)


    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 5
    thickness = 20

    global gesture_ids

    for i in gestures:
        img = np.zeros(size, dtype=np.uint8)
        
        curr_gesture = gesture_ids[str(i + 1)].strip()
        textsize = cv.getTextSize(curr_gesture, font, scale, thickness)[0]
        textX = (img.shape[1] - textsize[0]) // 2
        textY = (img.shape[0] + textsize[1]) // 2

        # add text centered on image
        img = cv.putText(img, curr_gesture, (textX, textY), font, scale, (255, 255, 255), thickness).astype(np.uint8)
#         plt.imshow(img)
#         plt.show()
        
        vid_out.write(img)

#     vid_out.release()


def vid_classification(model, vid_path):
    frames, fps, img_shape = vid_2_imgs(vid_path)

    model.eval()
    gestures = []
    for i in frames:
        i = torch.from_numpy(i).cuda().to(torch.float)
        i = i.unsqueeze(0)

        outputs = model(i)

        _, pred = torch.max(outputs, 1)
        gestures.append(int(pred.detach().cpu().numpy()))

    return gestures, fps, img_shape


def vid_infer():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

    vid_in_path = "/home/dshah/2dgesturenet/vid_in.mpg"
    vid_out_name = "/home/dshah/2dgesturenet/vid_out.mpg"

    saved_model_path = "/home/dshah/2dgesturenet/res2net50.pth"
    model = nn.DataParallel(res2net50(num_classes=35))
    model.cuda()
    model.load_state_dict(torch.load(saved_model_path))

    gestures, fps, img_shape = vid_classification(model, vid_in_path)

    create_vid(gestures, vid_out_name, fps, img_shape)


if __name__ == "__main__":
    vid_infer()
