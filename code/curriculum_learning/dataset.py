import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from config import TEST_IMAGE_ROOT, CLASSES, CLASS2IND
from sklearn.model_selection import GroupKFold

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root, is_train=True, transforms=None):
        """
        Args:
            image_root (str): 이미지가 저장된 루트 디렉토리
            label_root (str): JSON 레이블이 저장된 루트 디렉토리
            is_train (bool): 학습용 데이터셋 여부
            transforms (callable): 이미지와 레이블에 적용할 변환
        """
        # 이미지 및 레이블 경로 수집
        self.image_paths = sorted([
            os.path.join(root, fname)
            for root, _, files in os.walk(image_root)
            for fname in files
            if fname.lower().endswith(".png")
        ])
        
        self.label_paths = sorted([
            os.path.join(root, fname)
            for root, _, files in os.walk(label_root)
            for fname in files
            if fname.lower().endswith(".json")
        ])
        
        assert len(self.image_paths) == len(self.label_paths), "이미지와 레이블의 개수가 다릅니다."
        
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        # 이미지 로드
        image_path = self.image_paths[item]
        image = cv2.imread(image_path)
        image = image / 255.0

        # 레이블 로드
        label_path = self.label_paths[item]
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]

        # 레이블 생성
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"], dtype=np.int32)

            # 다각형 포맷을 채워진 마스크로 변환
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        # 변환 적용
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # Tensor로 변환
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=TEST_IMAGE_ROOT)
            for root, _dirs, files in os.walk(TEST_IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }

        _filenames = pngs
        _filenames = np.array(sorted(_filenames))

        self.filenames = _filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(TEST_IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()

        return image, image_name