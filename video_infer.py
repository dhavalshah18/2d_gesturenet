import cv2 as cv
import torch
import torch.nn as nn
import numpy as np
import pathlib
from subprocess import Popen, PIPE

from models.res2net import res2net50
from models.effnet import EfficientNet


class VidInfer:
    def __init__(self, vid_in_path, vid_out_path, model_path, font_scale=2, font_thickness=2):
        self.vid_in_path = vid_in_path
        self.vid_out_path = vid_out_path

        if "resnet" in model_path:
            self.model = nn.DataParallel(res2net50(num_classes=35))
        elif "efficient" in model_path or "eff" in model_path:
            self.model = nn.DataParallel(EfficientNet.from_name("efficientnet-b0", num_classes=35))

        self.model.load_state_dict(torch.load(model_path))
        self.gesture_ids = {}
        self.create_gesture_ids()

        self.frames = []
        self.gestures = []
        self.fps = 0
        self.size = ()
        self.scale = font_scale
        self.thickness = font_thickness

    def do_vid_infer(self):
        self.vid_2_imgs()
        self.vid_classification()
        self.create_vid()

    def create_gesture_ids(self):
        """
        Create dict of gestures from gesture_ids.txt
        """
        with open("gesture_ids.txt", "r") as f:
            gestures = f.readlines()

        for i in range(0, 35):
            line = gestures[i]
            gesture_num = line.split(" = ")[0].strip()
            gesture_label = line.split(" = ")[-1].strip()

            self.gesture_ids[gesture_num] = gesture_label

    def vid_2_imgs(self):
        """
        Extract each frame of input video and store in self.frames of object
        """
        vid_in = cv.VideoCapture(self.vid_in_path)
        self.fps = vid_in.get(cv.CAP_PROP_FPS)
        width = int(vid_in.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(vid_in.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.size = (height, width)

        success, img = vid_in.read()

        while success:
            # rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            rgb_img = img
            rgb_img = rgb_img.transpose((2, 0, 1))

            self.frames.append(rgb_img)

            success, img = vid_in.read()

        vid_in.release()

    def vid_classification(self):
        """
        Use model to classify each image in self.frames and save predicted gestures in self.gestures
        """
        self.model.eval()

        for i in self.frames:
            i = torch.from_numpy(i).to(torch.float)
            i = i.unsqueeze(0)

            outputs = self.model(i)

            val, pred = torch.max(outputs, 1)
            self.gestures.append([int(val.detach().cpu().numpy()), int(pred.detach().cpu().numpy())])

    def create_vid(self):
        """
        Find label of predicted gesture, create image with label text, and create video
        """
        font = cv.FONT_HERSHEY_SIMPLEX
        num = 0

        print("CREATING VIDEO")
        # Creating images for each frame
        for j, (val, i) in enumerate(self.gestures):
            img = np.zeros(self.size)
            curr_gesture = self.gesture_ids[str(i + 1)].strip()
            curr_gesture = str(val) + " " + curr_gesture

            textsize = cv.getTextSize(curr_gesture, font, self.scale, self.thickness)[0]
            text_x = (img.shape[1] - textsize[0]) // 2
            text_y = (img.shape[0] + textsize[1]) // 2

            # add text centered on image
            img = cv.putText(img, curr_gesture, (text_x, text_y), font, self.scale, (255, 255, 255), self.thickness).astype(np.uint8)

            cv.imwrite("./tmp/img_%03d.png" % num, img)
            num += 1

        # Using created images to make video with ffmpeg
        # frame rate 25
        process = Popen(['ffmpeg', '-y', '-i', './tmp/img_%03d.png', self.vid_out_path], stdout=PIPE, stderr=PIPE)
        process.communicate()

        path = pathlib.Path("./tmp")
        img_paths = path.glob("img*.png")
        for file in img_paths:
            file.unlink()


if __name__ == "__main__":
    vid_in = "/home/dhaval/2d_gesture_classification/vid_in_1.mpg"
    vid_out = "vid_out_1.mp4"
    saved_model_path = "/home/dhaval/2d_gesture_classification/efficientnet-b0.pth"

    vid_infer = VidInfer(vid_in, vid_out, saved_model_path)
    vid_infer.do_vid_infer()


