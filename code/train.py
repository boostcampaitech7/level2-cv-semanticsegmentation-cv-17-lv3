from dataset import XRayDataset
from config import NUM_EPOCHS, BATCH_SIZE, LR, CLASSES, SAVED_DIR, RANDOM_SEED, VAL_EVERY, THRESHOLD, NUM_CKPT, RESUME
import argparse

import segmentation_models_pytorch as smp

import time

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
from torch.utils.data import DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

# save_model 수정
'''
SAVED_DIR을 수정하면 원하는 DIR로 저장할 수 있습니다!!
file_name도 수정하면 checkpoint를 원하는 이름으로 저장 가능!
'''
class ModelCheckpoint:
    def __init__(self):
        self.best_models = []
        self.high_dice = 0.0
        
        if not os.path.exists(SAVED_DIR):
            os.makedirs(SAVED_DIR)

    def save_ckpt(self, model, path):
        torch.save(model.state_dict(), path)

    def delete_ckpt(self):
        dice_remove, epoch_remove, path_remove = self.best_models.pop(-1)
        if os.path.exists(path_remove):
            os.remove(path_remove)
            print(f"Delete model for epoch {epoch_remove+1} with dice = {dice_remove:.4f}")

    def save_model(self, model, epoch, loss, dice):
        model_file_name = f'epoch_{epoch+1}_dice_{dice:.4f}_loss_{loss:.4f}.pt'
        current_path = os.path.join(SAVED_DIR, model_file_name)

        if len(self.best_models) < NUM_CKPT:
            # num_ckpt 개수보다 적으면 모델 저장
            self.save_ckpt(model, current_path)
            self.best_models.append((dice, epoch, current_path))
            print(f"Save model for epoch {epoch+1} with dice = {dice:.4f}")

        elif dice > self.best_models[-1][0]:
            # dice가 더 높다면 가장 낮은 모델 제거 후 새 모델 저장
            self.delete_ckpt()
            self.save_ckpt(model, current_path)
            self.best_models.append((dice, epoch, current_path))
            print(f"Save model for epoch {epoch+1} with dice = {dice:.4f}")

        if dice > self.high_dice:
            # 최고 dice 갱신
            print(f"Best performance at epoch: {epoch + 1}, {self.high_dice:.4f} -> {dice:.4f}")
            self.high_dice = dice
        self.best_models.sort(reverse=True)

            # 최고 정확도 갱신
            if acc > self.high_acc:
                print(f"Best performance at epoch: {epoch + 1}, {self.high_acc:.4f} -> {acc:.4f}")
                self.high_acc = acc

    def save_best_model(self, model):
        # 가장 높은 dice 모델 저장
        best_model_path = os.path.join(SAVED_DIR, 'best_model.pt')
        best_dice, best_epoch, _ = self.best_models[0]
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved for epoch {best_epoch+1} with highest dice = {best_dice:.4f}")

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def train(args):
    if RESUME is not None:
        checkpoint = torch.load(RESUME)
        model.load_state_dict(checkpoint) 
           
    print(f'Start training..')

    # model 불러오기
    '''
    encoder모델과 weights를 변경하면 다른 모델을 사용할 수 있습니다!

    pip install git+https://github.com/qubvel/segmentation_models.pytorch
    '''
    model = smp.Unet(
        encoder_name="efficientnet-b0", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=29,                     # model output channels (number of classes in your dataset)
    )

    # Resize 변경하고 싶으면 변경
    '''
    1024도 좋아보입니다!
    '''
    tf = A.Resize(512, 512)

    train_dataset = XRayDataset(is_train=True, transforms = tf, n_splits=args.n_splits, n_fold=args.n_fold)
    valid_dataset = XRayDataset(is_train=False, transforms = tf, n_splits=args.n_splits, n_fold=args.n_fold)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    # Loss function을 정의합니다.
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer, Scheduler를 정의합니다.
    '''
    하이퍼 파라미터 수정 시 아래 함수 실행부분에서 수정하시면 됩니다!
    '''
    if args.optimizer == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=LR)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=LR)

    # 시드를 설정합니다.
    set_seed()

    checkpoint = ModelCheckpoint()

    for epoch in range(NUM_EPOCHS):
        model.train()

        total_train_loss = 0
        train_epoch_start = time.time()

        for step, (images, masks) in enumerate(train_loader):
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            # outputs = model(images)['out']
            outputs = model(images)

            # loss를 계산합니다.
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 20 == 0:
                print(
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )

        mean_train_loss = total_train_loss/len(train_loader)
        print(f"Train Mean Loss : {round(mean_train_loss, 4)}")
        
        elapsed_train_time = datetime.timedelta(seconds=round(time.time() - train_epoch_start))
        print(f"Elapsed Training Time: {elapsed_train_time}")
        print(f"ETA : {elapsed_train_time * (NUM_EPOCHS - epoch+1)}")

        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % VAL_EVERY == 0:
            print(f'Start validation #{epoch+1:2d}')

            set_seed()

            # init metrics
            total_valid_loss = 0
            dices = []
            
            model.eval()
            with torch.inference_mode():
                for step, (images, masks) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    images, masks = images.cuda(), masks.cuda()
                    model = model.cuda()
                    
                    outputs = model(images)
                    
                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)
                    
                    # restore original size
                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                    
                    loss = criterion(outputs, masks)
                    total_valid_loss += loss.item()
                    
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > THRESHOLD).detach().cpu()
                    masks = masks.detach().cpu()
                    
                    dice = dice_coef(outputs, masks)
                    dices.append(dice)
                
                # mean validation loss
                mean_valid_loss = total_valid_loss/len(valid_loader)
                
                dices = torch.cat(dices, 0)
                dices_per_class = torch.mean(dices, 0)
                dice_str = [
                    f"{c:<12}: {d.item():.4f}"
                    for c, d in zip(CLASSES, dices_per_class)
                ]
                dice_str = "\n".join(dice_str)
                print(dice_str)
                print(f"Valid Mean Loss : {round(mean_valid_loss, 4)}")
                
                # mean dice coefficient
                avg_dice = torch.mean(dices_per_class).item()
                
                checkpoint.save_model(model, epoch, mean_valid_loss, avg_dice)

    checkpoint.save_best_model(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #사용하고 싶은 파라미터 적용
    '''
    스케줄러 및 다른 optimizer도 추가 가능
    '''
    # optimizer
    parser.add_argument('--optimizer', type=str, default="adam")

    # data
    parser.add_argument('--n_splits', type=int, default="5")
    parser.add_argument('--n_fold', type=int, default="0")

    # num_ckpt
    parser.add_argument('--n_ckpt', type=int, default=3)

    # resume
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    train(args)