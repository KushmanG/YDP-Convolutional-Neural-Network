
from a_imports import *

HERE = os.path.dirname(os.path.abspath(__file__))

# Data storage and access directory name

# You can change the name of the directory frm which you want the data frm here
DATA_DIR = os.path.join(HERE, "datasets", "data_27pts") 
MANIFEST = os.path.join(DATA_DIR, "manifest.csv")

# Rasterization 0arameters
RESOLUTION = 128
BLOB_SIGMA = 1.0

# Split percentages
TRAIN_FRAC = 0.67
VAL_FRAC   = 0.33

# Training vars
BATCH_SIZE   = 16
LR           = 1e-3
WEIGHT_DECAY = 1e-4      
EPOCHS       = 300       

# Output Directories
CKPT_DIR  = os.path.join(HERE, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "best.pt")
OUT_DIR   = os.path.join(HERE, "Diagnostics", "outputs")

# --- device (Apple GPU if present, else CUDA, else CPU) ---------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
