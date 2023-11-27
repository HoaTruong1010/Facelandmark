import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch.utils import data

device = "cuda" if torch.cuda.is_available() else "cpu"


# plot 1 file .txt để xem chi tiết
def plot():
    annotations = pd.read_csv("./annotation/1.txt", sep=",",
                    header=None, skiprows=1, names=['x','y'])
    file_name = pd.read_csv("./annotation/1.txt", nrows=1, header=None)[0][0]

    im = Image.open(f"./images/{file_name}.jpg")
    draw = ImageDraw.Draw(im=im)
    for idx, r in annotations.iterrows():
        x_i = r['x']
        y_i = r['y']

        draw.regular_polygon((x_i, y_i, 2), n_sides=4, width=3)
    im.show()


# chọn ra khung chứa khuôn mặt đã được label
def check_face_and_landmarks(boxes, landmarks):
    if boxes is not None:
        for box in boxes:
            i = 0
            for idx in range(0, len(landmarks), 2):
                x = float(landmarks[idx])
                y = float(landmarks[idx+1])
                if (x > box[0]) and (x < box[2])\
                and (y > box[1]) and (y < box[3]):
                    i += 1
            if i >= (len(landmarks) / 2):
                return box
        return boxes[0]
    return None


# Khai báo dataset
class HelenDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None):
        self.line = None
        self.path = None
        self.landmarks = None
        self.filenames = None
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):        
        #get txt path
        self.line = self.lines[index].strip().split()[0]
        
        #read image name and landmarks
        with open(self.line, 'r') as f:
            file = f.readlines()

        #read image
        img = Image.open(f"./face/{file[0].strip()}.jpg").convert("RGB")

        if self.transforms:
            self.img = self.transforms(img)

        #read landmarks
        landmarks = []
        for i in file[1:]:
            (xi, yi) = i.strip().split(", ")
            landmarks.extend([float(xi), float(yi)])

        self.landmarks = np.asarray(landmarks, dtype=np.float32)
        
        return (self.img, self.landmarks)

    def __len__(self):
        return len(self.lines)