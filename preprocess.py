from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np


def check_face_and_landmarks(boxes, landmarks):
    if boxes is not None:
        min = len(landmarks) / 3
        for box in boxes:
            i = 0
            for idx in range(0, len(landmarks), 2):
                x = float(landmarks[idx])
                y = float(landmarks[idx+1])
                if (x > box[0]) and (x < box[2])\
                and (y > box[1]) and (y < box[3]):
                    i += 1
            if i >= min:
                return box
            
    return None


def detect_face(file_path, mtcnn, landmarks):
    img = Image.open(file_path).convert("RGB")

    boxes, _ = mtcnn.detect(img) 
    if boxes is not None:   
        if boxes.shape[0] > 1:
            box = check_face_and_landmarks(boxes, landmarks)
        else:
            box = boxes[0]  
    else:
        box = [0, 0, img.size[0], img.size[1]]
    padding = 25
    face_box = [box[0]-padding, box[1]-padding, box[2]+padding, box[3]+padding]
    face_cropped = img.crop(face_box)

    return face_cropped, face_box
    

def convert_landmarks(landmarks, size, face_cropped, box):
    (width, height) = face_cropped.size
    lm = []
    for idx in range(0, len(landmarks), 2):
        lx = np.multiply(float(landmarks[idx]-box[0]), np.divide(size, width))
        ly = np.multiply(float(landmarks[idx+1]-box[1]), np.divide(size, height))  
        lm.append(f'{lx}, {ly}')     
    
    return lm


if __name__ == "__main__":
    mtcnn = MTCNN()
    file_path = ['./list.txt', './list_test.txt', './list_val.txt']
    idx = 0
    for path in file_path:
        with open(path, 'r') as f:
            file = f.readlines()
        for f in file:
            idx += 1
            with open(f.strip(), 'r') as f:
                image_and_landmarks = f.readlines()
            
            img_name = image_and_landmarks[0].strip()
            img_path = f"./images/{img_name}.jpg"
            landmarks = []
            for i in image_and_landmarks[1:]:
                (xi, yi) = i.strip().split(" , ")
                landmarks.extend([float(xi), float(yi)])
            img, box = detect_face(img_path, mtcnn, landmarks)
            size = 112
            new_landmarks = convert_landmarks(landmarks, size, img, box)
            
            if path.__contains__("test"):
                with open(f'./new_annotation_test/{str(idx)}.txt', 'w') as f:
                    f.write(img_name)
                    f.write('\n')
                    f.write('\n'.join(new_landmarks))
            elif path.__contains__("val"):
                with open(f'./new_annotation_val/{str(idx)}.txt', 'w') as f:
                    f.write(img_name)
                    f.write('\n')
                    f.write('\n'.join(new_landmarks))
            else:
                with open(f'./new_annotation_train/{str(idx)}.txt', 'w') as f:
                    f.write(img_name)
                    f.write('\n')
                    f.write('\n'.join(new_landmarks))
            img.save(f"./face/{img_name}.jpg")