import torch
from torchvision import transforms
from model import PFLDInference
from PIL import Image
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load checkpoint vào model
checkpoint = torch.load('./checkpoint/checkpoint_epoch_48.pth', map_location=device)
pfld_backbone = PFLDInference().to(device)
pfld_backbone.load_state_dict(checkpoint)

#khai báo model và thực hiện nhận diện khuôn mặt
mtcnn = MTCNN()
path = './demo.jpg'
img = Image.open(path)

boxes, _ = mtcnn.detect(img)

if boxes is None:  
    boxes = [0, 0, img.size[0], img.size[1]]

#cắt các khuôn mặt đã nhận diện được với padding = 25 và transform image
padding = 25
face_cropped = []
face_img = []
face_boxes = []
transform = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor()
])

for box in boxes:    
    face_box = [box[0]-padding, box[1]-padding, box[2]+padding, box[3]+padding]
    face_img.append(img.crop(face_box))
    face_cropped.append(transform(img.crop(face_box)))
    face_boxes.append(face_box)

# tìm landmark và plot lên ảnh demo.jpg
pfld_backbone.eval()
with torch.no_grad():    
    pfld_backbone = pfld_backbone.to(device)

    all_landmarks = []
    for face in face_cropped:
        face = face.unsqueeze(0).to(device)
        _, landmarks = pfld_backbone(face)
        all_landmarks.append(landmarks)

for idx, landmarks in enumerate(all_landmarks):
    all_landmarks[idx] = landmarks.cpu().numpy().reshape((-1, 2))  
    
for idx, face in enumerate(face_cropped):        
    plt.imshow(img)
    
    (width, height) = face_img[idx].size
    for i in range(all_landmarks[idx].shape[0]):
        x_i = np.divide(float(all_landmarks[idx][i][0]), np.divide(112, width)) + face_boxes[idx][0]
        y_i = np.divide(float(all_landmarks[idx][i][1]), np.divide(112, height)) + face_boxes[idx][1]
        plt.scatter(x_i, y_i, c='r', s=1) 

plt.show()
    

        