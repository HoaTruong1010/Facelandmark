# ------------------------------------------------------------------------------
# Copyright (c) Zhichao Zhao
# Licensed under the MIT License.
# Created by Zhichao zhao(zhaozhichao4515@gmail.com)
# ------------------------------------------------------------------------------
import argparse
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset import HelenDatasets

from model import PFLDInference

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        interocular = np.linalg.norm(pts_gt[136, ] - pts_gt[116, ])
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt,
                                        axis=1)) / (interocular * L)

    return rmse


def compute_auc(errors, failureThreshold, step=0.0001, showCurve=True):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    return AUC, failureRate


def validate(test_dataloader, pfld_backbone):
    pfld_backbone.eval()

    nme_list = []
    cost_time = []
    with torch.no_grad():
        for img, landmark_gt in test_dataloader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)

            start_time = time.time()
            _, landmarks = pfld_backbone(img)
            cost_time.append(time.time() - start_time)

            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1,
                                          2)  # landmark
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1,
                                              2).cpu().numpy()  # landmark_gt

            nme_temp = compute_nme(landmarks, landmark_gt)
            for item in nme_temp:
                nme_list.append(item)

        # nme
        print('nme: {:.4f}'.format(np.mean(nme_list)))
        # auc and failure rate
        failureThreshold = 0.1
        auc, failure_rate = compute_auc(nme_list, failureThreshold)
        print('auc @ {:.1f} failureThreshold: {:.4f}'.format(
            failureThreshold, auc))
        print('failure_rate: {:}'.format(failure_rate))
        # inference time
        print("inference_cost_time: {0:4f}".format(np.mean(cost_time)))


def main(args):
    checkpoint = torch.load(args.model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint)

    transform = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor()
    ])
    dataset = HelenDatasets(args.test_dataset, transform)
    dataloader = DataLoader(dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0)

    validate(dataloader, pfld_backbone)


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path',
                        default="./checkpoint/checkpoint_epoch_48.pth",
                        type=str)
    parser.add_argument('--test_dataset',
                        default='./new_list_test.txt',
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)