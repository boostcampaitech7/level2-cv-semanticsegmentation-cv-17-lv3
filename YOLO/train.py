import os
import cv2
import yaml
import wandb
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
import pandas as pd
import albumentations as A

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


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


def train(data_config_path: str):   
    with open(data_config_path, 'r') as file:
        data_config = yaml.safe_load(file)
        
    wandb_option = data_config["wandb_option"]
    wandb.init(
        entity='cv-17_segmentation',
        project=wandb_option["project"],
        name=wandb_option["name"],
    )
    
    train_option = data_config["train_option"]
    # custom_augment = A.Compose([
    #     A.CLAHE(p=0.5),
    #     A.Resize(train_option["imgsz"], train_option["imgsz"]),
    #     A.Rotate(limit=30, p=0.2),
    # ])
    model = YOLO("yolov8x-seg.pt")
    
    # want to customize, See under page
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
    model.train(
        data=data_config_path,
        epochs=train_option["epochs"],
        imgsz=train_option["imgsz"],
        device=train_option["device"],
        batch=train_option["batch"],
        workers=train_option["workers"],
        cos_lr=train_option["cos_lr"],
        optimizer=train_option["optimizer"],
        hsv_s=0.2,
        hsv_v=0.3
        # mosaic=1.0,
        # fliplr=0.0,
        # erasing=0.0,
        # scale=0.0,
        # translate=0.0,
    )
    
if __name__ == '__main__':
    train('config/yolo_config.yaml')