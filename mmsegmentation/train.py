import os
import json
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm.auto import tqdm
import wandb

import torch
from torch import nn
import torch.nn.functional as F

from mmseg.registry import DATASETS, TRANSFORMS, MODELS, METRICS, HOOKS
from mmseg.datasets import BaseSegDataset
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.decode_heads import ASPPHead, FCNHead, SegformerHead
from mmseg.models.utils.wrappers import resize


from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner, load_checkpoint
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.structures import PixelData
from mmengine.hooks import Hook

from mmcv.transforms import BaseTransform

# WandB setting
PROJECT = 'project_name'
NAME = 'run_name'
ENTITY = 'entity'
# Work_dir
WOKR_DIR = 'work_dir'

IMAGE_ROOT = "../data/train/DCM/"
LABEL_ROOT = "../data/train/outputs_json/"

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

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

pngs = sorted(pngs)
jsons = sorted(jsons)

# wandb Hook 정의
@HOOKS.register_module()
class WandbHook(Hook):
    def __init__(self, project_name, run_name=None, entity=None, config=None):
        self.project_name = project_name
        self.run_name = run_name
        self.entity = entity
        self.config = config

    def before_run(self, runner):
        """Initialize wandb before the training starts."""
        wandb.init(
            project=self.project_name,
            name=self.run_name,
            entity=self.entity,
            config=self.config or runner.cfg
        )

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Log training metrics to wandb."""
        # Get current learning rate
        lr = runner.optim_wrapper.get_lr()

        # If lr is a list, take the first value or compute an average
        if isinstance(lr, list):
            lr = sum(lr) / len(lr)  # 평균값 계산
        elif isinstance(lr, torch.Tensor):
            lr = lr.item()  # Tensor인 경우 스칼라로 변환

        # Log train loss and learning rate
        if outputs and 'loss' in outputs:
            wandb.log({
                'train_loss': outputs['loss'].item(),
                'learning_rate': lr['lr'][0],  # 스칼라 형태의 학습률
                'step': runner.iter
            })

    def after_val_epoch(self, runner, metrics):
        """Log validation metrics after each validation epoch."""
        log_data = {
        'epoch': int(runner.iter / 640)  # 현재 epoch 계산
        }

        print(metrics)
        # mDice 로그 기록
        if metrics and 'mDice' in metrics:
            log_data['val_mDice'] = metrics['mDice']

        # 클래스별 Dice 점수 기록
        if metrics and 'Dice' in metrics:  # 클래스별 Dice 값이 'Dice'에 저장되었다고 가정
            dice_values = metrics['Dice']  # 클래스별 Dice 값 배열
            for class_idx, dice_value in enumerate(dice_values):
                log_data[f'class_{IND2CLASS[class_idx]}_dice'] = dice_value

        # wandb에 로그 기록
        wandb.log(log_data)

    def after_run(self, runner):
        """Close wandb after the run finishes."""
        wandb.finish()

class PostProcessResultMixin:
    def postprocess_result(self,
                           seg_logits,
                           data_samples):
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            i_seg_logits = i_seg_logits.sigmoid()
            i_seg_pred = (i_seg_logits > 0.5).to(i_seg_logits)

            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples
    
@MODELS.register_module()
class EncoderDecoderWithoutArgmax(PostProcessResultMixin, EncoderDecoder):
    pass

class LossByFeatMixIn:
    def loss_by_feat(self, seg_logits, batch_data_samples) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        return loss
    
@MODELS.register_module()
class ASPPHeadWithoutAccuracy(LossByFeatMixIn, ASPPHead):
    pass

@MODELS.register_module()
class FCNHeadWithoutAccuracy(LossByFeatMixIn, FCNHead):
    pass

@MODELS.register_module()
class SegformerHeadWithoutAccuracy(LossByFeatMixIn, SegformerHead):
    pass

@DATASETS.register_module()
class XRayDataset(BaseSegDataset):
    def __init__(self, is_train, **kwargs):
        self.is_train = is_train

        super().__init__(**kwargs)

    def load_data_list(self):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)

        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]

        # dummy label
        ys = [0 for fname in _filenames]

        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 3:
                    continue

                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])

            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])

                # skip i > 0
                break

        data_list = []
        for i, (img_path, ann_path) in enumerate(zip(filenames, labelnames)):
            data_info = dict(
                img_path=os.path.join(IMAGE_ROOT, img_path),
                seg_map_path=os.path.join(LABEL_ROOT, ann_path),
            )
            data_list.append(data_info)

        return data_list
    
@TRANSFORMS.register_module()
class LoadXRayAnnotations(BaseTransform):
    def transform(self, result):
        label_path = result["seg_map_path"]

        image_size = (2048, 2048)

        # process a label of shape (H, W, NC)
        label_shape = image_size + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(image_size, dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        result["gt_seg_map"] = label

        return result
    
@TRANSFORMS.register_module()
class TransposeAnnotations(BaseTransform):
    def transform(self, result):
        result["gt_seg_map"] = np.transpose(result["gt_seg_map"], (2, 0, 1))
        return result
    
@METRICS.register_module()
class DiceMetric(BaseMetric):
    def __init__(self,
                 collect_device='cpu',
                 prefix=None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)

    @staticmethod
    def dice_coef(y_true, y_pred):
        y_true_f = y_true.flatten(-2)
        y_pred_f = y_pred.flatten(-2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)

        eps = 0.0001
        return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

    def process(self, data_batch, data_samples):
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data']

            label = data_sample['gt_sem_seg']['data'].to(pred_label)
            self.results.append(
                self.dice_coef(label, pred_label)
            )

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        results = torch.stack(self.results, 0)
        dices_per_class = torch.mean(results, 0)
        avg_dice = torch.mean(dices_per_class)

        ret_metrics = {
            "Dice": dices_per_class.detach().cpu().numpy(),
        }
        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        metrics = {
            "mDice": torch.mean(dices_per_class).item()
        }

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': CLASSES})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics


# load config
cfg = Config.fromfile("./custom/configs/mask2former.py")
cfg.launcher = "none"
cfg.work_dir = WOKR_DIR

# resume training
cfg.resume = False

cfg.custom_hooks = [
    dict(
        type='WandbHook',
        project_name=PROJECT,
        run_name=NAME,
        entity=ENTITY,
    )
]

runner = Runner.from_cfg(cfg)

# start training
runner.train()