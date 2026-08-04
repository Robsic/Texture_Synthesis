# --- Dataset ---
BATCH_SIZE = 4
IMG_HEIGHT = 256
IMG_WIDTH = 256
VAL_RATIO = 0.10
SEED = 42

PAIRS = [
    ("./testing_files/sample_images/Hm", "./testing_files/sample_images/Tx"),
]

# --- Model ---
OUTPUT_CHANNELS = 3
LAMBDA = 100

CHECKPOINT_DIR = './tensorboard/checkpoints'
LOG_DIR = './tensorboard/logs'

MODEL_SAVE_PATH = './testing_files/pre_trained_model'

# --- Training ---
STEPS = 1000
