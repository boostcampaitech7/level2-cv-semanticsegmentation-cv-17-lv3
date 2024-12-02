import os
import albumentations as A

# Paths for dataset
TRAIN_IMAGE_ROOT = "/data/ephemeral/home/euna/data/train_split/DCM"
VAL_IMAGE_ROOT = "/data/ephemeral/home/euna/data/val_split/DCM"
TEST_IMAGE_ROOT = "/data/ephemeral/home/euna/data/test/DCM"
TRAIN_LABEL_ROOT = "/data/ephemeral/home/euna/data/train_split/outputs_json"
VAL_LABEL_ROOT = "/data/ephemeral/home/euna/data/val_split/outputs_json"

# curriculum dataset
TRAIN_IMAGE_ROOT1 = "/data/ephemeral/home/euna/data/curriculum/train_data_1/DCM"
TRAIN_LABEL_ROOT1 = "/data/ephemeral/home/euna/data/curriculum/train_data_1/outputs_json"
TRAIN_IMAGE_ROOT2 = "/data/ephemeral/home/euna/data/curriculum/train_data_2/DCM"
TRAIN_LABEL_ROOT2 = "/data/ephemeral/home/euna/data/curriculum/train_data_2/outputs_json"
VAL_IMAGE_ROOT_C = "/data/ephemeral/home/euna/data/curriculum/val_data/DCM"
VAL_LABEL_ROOT_C = "/data/ephemeral/home/euna/data/curriculum/val_data/outputs_json"

# Class definitions
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

# Hyperparameters
BATCH_SIZE = 4
LR = 1e-3
RANDOM_SEED = 21
THRESHOLD = 0.5

NUM_EPOCHS = 45
VAL_EVERY = 1
NUM_CKPT = 5

RESUME = None
SAVED_DIR = "./checkpoints/45"
OUTPUTS_DIR = "./outputs"

# Optimizer and Scheduler
OPTIMIZER = "adamp" # Options: adam, adamw, adamp, radam, lion
SCHEDULER = "CosineAnnealingLR" # Options: CosineAnnealingLR, MultiStepLR

# Loss functions
LOSS = 'bce_dice_loss' # Options: bce_loss, iou_loss, dice_loss, bce_dice_loss, bce_iou_loss

# Data
SPLITS = 5
FOLD = 0

# Transforms
transforms = {
    'train': A.Compose([
        A.Resize(1024, 1024),
        A.CLAHE(p=0.5)
    ]),
    'val': A.Compose([
        A.Resize(1024, 1024),
        # A.CLAHE(p=0.5)
    ])
}

# Model
MODEL = 'UPerNet' # Options: UnetPlusPlus, Unet, FPN, Linknet, MAnet, PAN, PSPNet, DeepLabV3, DeepLabV3Plus, UPerNet
ENCODER_MODEL = "resnet152"
ENCODER_MODEL_WEIGHTS = "imagenet"
MODEL_NAME = "post35_epoch_40_dice_0.9641_loss_0.0194.pt"
OUTPUTS_NAME = "95_uper_curriculum_post30_40.csv"

# WandB settings
PROJECT_NAME = 'EUNA'
EXP_NAME = '95_uper_curriculum_clahe5_1e-3_1024_35_45'

# Early stopping
PATIENCE = 5
DELTA = 0.001

if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR)

if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)