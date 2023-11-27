import torch
from torchvision import transforms
from model import PFLDInference
from dataset import HelenDatasets
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('./checkpoint/checkpoint_epoch_48.pth', map_location=device)
pfld_backbone = PFLDInference().to(device)
pfld_backbone.load_state_dict(checkpoint)

transform = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor()
])
dataset = HelenDatasets('./new_list_test.txt', transform)
dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0)

pfld_backbone.eval()

nme_list = []
cost_time = []
with torch.no_grad():
    for img, landmark_gt in dataloader:
        img = img.to(device)
        landmark_gt = landmark_gt.to(device)
        pfld_backbone = pfld_backbone.to(device)

        start_time = time.time()
        _, landmarks = pfld_backbone(img)
        cost_time.append(time.time() - start_time)

        landmarks = landmarks.cpu().numpy()
        landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)  
        plt.imshow(np.transpose(img[0].cpu().detach().numpy(), (1, 2, 0)))
        for idx in range(landmarks[0].shape[0]):
            x_i = landmarks[0][idx][0]
            y_i = landmarks[0][idx][1]
            plt.scatter(x_i, y_i, c='r', s=1)
        plt.show()
