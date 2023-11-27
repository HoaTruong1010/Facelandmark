import os
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import model
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import HelenDatasets
from torch.optim import Adam
import torch.nn as nn
from loss import PFLDLoss
from utils import AverageMeter
import matplotlib.pyplot as plt
import time


def train(dataloader, pfld_backbone, auxiliarynet, optimizer):
    losses = AverageMeter()
    # Define your execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert model parameters and buffers to CPU or Cuda
    pfld_backbone.to(device)
    auxiliarynet.to(device)

    for images, landmarks in dataloader:
        
        # get the inputs
        images = images.to(device)
        landmarks = landmarks.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # predict classes using images from the training set
        (feature, r_landmarks) = pfld_backbone(images)
        angle = auxiliarynet(feature)

        # print("labels: ", landmarks)
        # print("predict: ", r_landmarks)
        # break
        
        # compute the loss based on model output and real labels
        loss = loss_fn(landmarks, r_landmarks)
        # backpropagate the loss
        loss.backward()
        # adjust parameters based on the calculated gradients
        optimizer.step()
        # print(loss.item())
        losses.update(loss.item())

    return loss


def validate(val_dataloader, pfld_backbone, auxiliarynet):
    pfld_backbone.eval()
    auxiliarynet.eval()
    losses = []
    with torch.no_grad():
        for img, landmark_gt in val_dataloader:
            img = img.to(device)
            pfld_backbone = pfld_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            _, landmark = pfld_backbone(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            losses.append(loss.cpu().numpy())
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} '.format(np.mean(losses)))
    return np.mean(losses)


if __name__ == '__main__':   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pfld_backbone = model.PFLDInference()    
    auxiliarynet = model.AuxiliaryNet()
    transform = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor()
    ])
    dataset = HelenDatasets("./new_list_train.txt", transform)
    dataloader = DataLoader(dataset,
                            batch_size=10,
                            shuffle=True,
                            num_workers=4,
                            drop_last=False)
    dataset_val = HelenDatasets("./new_list_val.txt", transform)
    dataloader_val = DataLoader(dataset_val,
                            batch_size=10,
                            shuffle=True,
                            num_workers=4,
                            drop_last=False)
    print("======Load data finish======")
    print("======Begin train model======")
    loss_fn = PFLDLoss()
    optimizer = Adam(pfld_backbone.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=40, verbose=True)
    losses = []
    val_losses = []
    for epoch in range(0, 50):
        loss = train(dataloader, pfld_backbone, auxiliarynet, optimizer)

        torch.save(pfld_backbone.state_dict(), f"./checkpoint/checkpoint_epoch_{str(epoch)}.pth")
        print(f"Loss epoch {str(epoch)} is: {loss}")
        losses.append(loss.item())
        val_loss = validate(dataloader_val, pfld_backbone, auxiliarynet)
        scheduler.step(val_loss)
        val_losses.append(val_loss.item())
    print("================Train model finish==============")
    
    plt.plot(losses, color="red", label="training loss")
    plt.plot(val_losses, color="green", label="validate loss")
    plt.show()
