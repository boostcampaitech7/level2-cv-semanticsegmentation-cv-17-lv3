from dataset import XRayDataset
import config as cf

import argparse

import segmentation_models_pytorch as smp

import time
import wandb

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
# pip install adamp
from adamp import AdamP
# pip install lion-pytorch
from lion_pytorch import Lion

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt

from loss import iou_loss, dice_loss, bce_dice_loss, bce_iou_loss

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def validate(model, valid_loader, criterion, args):
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
            outputs = (outputs > args.threshold)
            
            dice = dice_coef(outputs, masks)
            dices.append(dice.detach().cpu())
        
        # mean validation loss
        mean_valid_loss = total_valid_loss/len(valid_loader)
        
        dices = torch.cat(dices, 0)
        dices_per_class = torch.mean(dices, 0)
        
        print(f"Valid Mean Loss : {round(mean_valid_loss, 4)}")
        
        # mean dice coefficient
        avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice, dices_per_class, mean_valid_loss

# save_model 수정
'''
SAVED_DIR을 수정하면 원하는 DIR로 저장할 수 있습니다!!
file_name도 수정하면 checkpoint를 원하는 이름으로 저장 가능!
'''

#Early Stopping
class EarlyStopping:
    def __init__(self, patience, mode, delta):
        """
        초기화 메서드:
        - patience: 성능이 개선되지 않는 epoch 수 제한.
        - mode: 성능을 개선으로 판단하는 기준 ("max"는 값이 클수록 좋고, "min"은 작을수록 좋음).
        - delta: 성능 개선을 최소로 판단하는 값. delta 이상 변화가 있어야 개선으로 간주.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None  # 현재까지 가장 좋은 점수
        self.counter = 0        # 개선되지 않은 epoch 횟수
        self.early_stop = False # 조기 종료 여부

    def __call__(self, current_score):
        """
        현재 점수에 따라 상태를 업데이트:
        - best_score가 없거나 개선되었으면 점수를 갱신하고 counter 초기화.
        - 개선되지 않았다면 counter 증가, patience 초과 시 early_stop 활성화.
        """
        if self.best_score is None or self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience

    def _is_improvement(self, current_score):
        """
        현재 점수가 개선되었는지 확인:
        - mode="max"이면 current_score > best_score + delta
        - mode="min"이면 current_score < best_score - delta
        """
        return (current_score > self.best_score + self.delta) if self.mode == "max" else (current_score < self.best_score - self.delta)



class ModelCheckpoint:
    def __init__(self):
        self.best_models = []
        self.high_dice = 0.0
        
        if not os.path.exists(args.saved_dir):
            os.makedirs(args.saved_dir)

    def save_ckpt(self, model, path):
        torch.save(model, path)

    def delete_ckpt(self):
        dice_remove, epoch_remove, path_remove = self.best_models.pop(-1)
        if os.path.exists(path_remove):
            os.remove(path_remove)
            print(f"Delete model for epoch {epoch_remove+1} with dice = {dice_remove:.4f}")

    def save_model(self, model, epoch, loss, dice):
        model_file_name = f'epoch_{epoch+1}_dice_{dice:.4f}_loss_{loss:.4f}.pt'
        current_path = os.path.join(args.saved_dir, model_file_name)

        if len(self.best_models) < args.num_ckpt:
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

    def save_best_model(self, model):
        # 가장 높은 dice 모델 저장
        best_model_path = os.path.join(args.saved_dir, 'best_model.pt')
        best_dice, best_epoch, _ = self.best_models[0]
        torch.save(model, best_model_path)
        print(f"Best model saved for epoch {best_epoch+1} with highest dice = {best_dice:.4f}")

def set_seed():
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

def train(args):
    wandb_config = {
        "entity" : 'cv-17_segmentation',
        "project": args.project,
        "config": {
            "optimizer": args.optimizer,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.num_epochs,
            "num_classes": len(args.classes),
        },
    }

    if args.exp_name is not None:
        wandb_config["name"] = args.exp_name

    wandb.init(**wandb_config)

    print(f'Start training..')

    # model 불러오기
    '''
    encoder모델과 weights를 변경하면 다른 모델을 사용할 수 있습니다!

    pip install git+https://github.com/qubvel/segmentation_models.pytorch
    '''
    model = smp.Unet(
        encoder_name=args.encoder_model, # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=args.encoder_model_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=29,                     # model output channels (number of classes in your dataset)
    )

    if args.resume is not None:
        model = torch.load(args.resume)

    # transforms
    train_tf, val_tf = cf.transforms['train'], cf.transforms['val']

    train_dataset = XRayDataset(is_train=True, transforms = train_tf, n_splits=args.n_splits, n_fold=args.n_fold)
    valid_dataset = XRayDataset(is_train=False, transforms = val_tf, n_splits=args.n_splits, n_fold=args.n_fold)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )

    # Loss function을 정의합니다.
    if args.loss_fn == 'bce_loss':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_fn == 'iou_loss':
        criterion = iou_loss
    elif args.loss_fn == 'dice_loss':
        criterion = dice_loss
    elif args.loss_fn == 'bce_dice_loss':
        criterion = bce_dice_loss
    elif args.loss_fn == 'bce_iou_loss':
        criterion = bce_iou_loss


    # Optimizer, Scheduler를 정의합니다.
    '''
    하이퍼 파라미터 수정 시 아래 함수 실행부분에서 수정하시면 됩니다!
    '''
    if args.optimizer == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)
    elif args.optimizer == "adamp":
        optimizer = AdamP(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    elif args.optimizer == "radam":
        optimizer = optim.RAdam(params=model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif args.scheduler == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1)
    

    # 시드를 설정합니다.
    set_seed()

    checkpoint = ModelCheckpoint()

    # EarlyStopping 초기화
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, mode="max")

    for epoch in range(args.num_epochs):
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

            # 현재 학습률 가져오기
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"train_loss": loss.item(), "learning_rate": current_lr})

            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 20 == 0:
                print(
                    f'Epoch [{epoch+1}/{args.num_epochs}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )

        mean_train_loss = total_train_loss/len(train_loader)
        print(f"Train Mean Loss : {round(mean_train_loss, 4)}")
        
        elapsed_train_time = datetime.timedelta(seconds=round(time.time() - train_epoch_start))
        print(f"Elapsed Training Time: {elapsed_train_time}")
        print(f"ETA : {elapsed_train_time * (args.num_epochs - epoch+1)}")

        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % args.val_every == 0:
            print(f'Start validation #{epoch+1:2d}')

            set_seed()

            # validation
            avg_dice, dices_per_class, mean_valid_loss = validate(model, valid_loader, criterion, args)
            dice_str = [
                f"{c:<12}: {d.item():.4f}"
                for c, d in zip(args.classes, dices_per_class)
            ]
            dice_str = "\n".join(dice_str)
            print(dice_str)
            print(f"Valid Mean Loss : {round(mean_valid_loss, 4)}")
            
            # mean dice coefficient
            print(f"Avg dice : {round(avg_dice, 4)}")
            wandb.log({
                "epoch": epoch + 1,
                "valid_loss": mean_valid_loss,
                "avg_dice": avg_dice,
                **{f"dice_{cls}": score for cls, score in zip(args.classes, dices_per_class)}
            })

            # EarlyStopping 체크
            early_stopping(avg_dice)  
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break 

            checkpoint.save_model(model, epoch, mean_valid_loss, avg_dice)
            
        if args.scheduler == 'CosineAnnealingLR' or args.scheduler == 'MultiStepLR':
            scheduler.step()
        
    checkpoint.save_best_model(model)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #사용하고 싶은 파라미터 적용
    '''
    스케줄러 및 다른 optimizer도 추가 가능
    '''
    # model
    parser.add_argument('--encoder_model', type=str, default=cf.ENCODER_MODEL)
    parser.add_argument('--encoder_model_weights', type=str, default=cf.ENCODER_MODEL_WEIGHTS)

    # classes
    parser.add_argument('--classes', type=str, default=cf.CLASSES)
    parser.add_argument('--class2ind', type=str, default=cf.CLASS2IND)
    parser.add_argument('--ind2class', type=str, default=cf.IND2CLASS)

    # parameters
    parser.add_argument('--batch_size', type=int, default=cf.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=cf.LR)
    parser.add_argument('--random_seed', type=int, default=cf.RANDOM_SEED)
    parser.add_argument('--threshold', type=float, default=cf.THRESHOLD)
    parser.add_argument('--num_epochs', type=int, default=cf.NUM_EPOCHS)
    parser.add_argument('--val_every', type=int, default=cf.VAL_EVERY)
    parser.add_argument('--num_ckpt', type=int, default=cf.NUM_CKPT)

    # dir
    parser.add_argument('--resume', type=str, default=cf.RESUME)
    parser.add_argument('--saved_dir', type=str, default=cf.SAVED_DIR)
    parser.add_argument('--outputs_dir', type=str, default=cf.OUTPUTS_DIR)

    # optimizer
    parser.add_argument('--optimizer', type=str, default=cf.OPTIMIZER)
    parser.add_argument('--scheduler', type=str, default=cf.SCHEDULER)

    # data
    parser.add_argument('--n_splits', type=int, default=cf.SPLITS)
    parser.add_argument('--n_fold', type=int, default=cf.FOLD)

    # Wandb
    parser.add_argument('--project', type=str, default=cf.PROJECT_NAME)
    parser.add_argument('--exp_name', default=cf.EXP_NAME)

    # early stopping
    parser.add_argument('--patience', type=int, default=cf.PATIENCE, 
                        help="Number of epochs with no improvement before stopping.")
    parser.add_argument('--delta', type=float, default=cf.DELTA, 
                        help="Minimum change in the monitored metric to qualify as an improvement.")   

    # loss
    parser.add_argument('--loss_fn', type=str, default=cf.LOSS)

    args = parser.parse_args()

    train(args)