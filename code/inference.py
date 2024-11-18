from dataset import XRayInferenceDataset
import config as cf
import argparse
import ttach as tta

# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt

# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)

def set_seed():
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

def test(model_name):
    set_seed()

    model = torch.load(os.path.join(args.saved_dir, args.model_name))

    transforms = tta.Compose(
        [
            tta.HorizontalFlip()
        ]
    )

    tta_model = tta.SegmentationTTAWrapper(model, transforms)

    tta_model = model.cuda()
    tta_model.eval()

    tf = A.Resize(512, 512)

    test_dataset = XRayInferenceDataset(transforms=tf)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    thr=0.5
    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(args.classes)

        for step, (images, image_names) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.cuda()
            outputs = tta_model(images)

            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{args.ind2class[c]}_{image_name}")

    # to csv
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    output_csv_path = os.path.join(args.outputs_dir, args.outputs_name)
    
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default=cf.MODEL_NAME)
    parser.add_argument('--saved_dir', type=str, default=cf.SAVED_DIR)
    parser.add_argument('--random_seed', type=str, default=cf.RANDOM_SEED)
    parser.add_argument('--classes', type=str, default=cf.CLASSES)
    parser.add_argument('--ind2class', type=str, default=cf.IND2CLASS)
    parser.add_argument('--outputs_dir', type=str, default=cf.OUTPUTS_DIR)
    parser.add_argument('--outputs_name', type=str, default=cf.OUTPUTS_NAME)

    args = parser.parse_args()
    test(args)