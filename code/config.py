import os

# Paths for dataset
TRAIN_IMAGE_ROOT = "/data/ephemeral/home/code/train/DCM"
TEST_IMAGE_ROOT = "/data/ephemeral/home/code/test/DCM"
LABEL_ROOT = "/data/ephemeral/home/code/train/outputs_json"

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
BATCH_SIZE = 8
LR = 1e-4
RANDOM_SEED = 21
THRESHOLD = 0.5

NUM_EPOCHS = 5
VAL_EVERY = 1
NUM_CKPT = 3

SAVED_DIR = "checkpoints"
OUTPUTS_DIR = "outputs"

if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR)

if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)