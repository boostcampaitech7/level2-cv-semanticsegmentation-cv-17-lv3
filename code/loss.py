import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth = 1.):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def iou_loss(inputs, targets, smooth=1) : 
    inputs = torch.sigmoid(inputs)      
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)
    return 1 - IoU

def bce_dice_loss(pred, target, bce_weight = 0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)

    bce_weight = 0.5
    dice_weight = 0.5

    loss = bce*bce_weight + dice*dice_weight 
    return loss

def bce_iou_loss(pred, target, bce_weight = 0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    iou = iou_loss(pred, target)

    bce_weight = 0.5
    iou_weight = 0.5

    loss = bce*bce_weight + iou*iou_weight
    return loss