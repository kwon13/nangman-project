from pathlib import Path
from os import path

# the directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = path.join(PROJECT_ROOT, "data")
SAVED_DIR = path.join(DATA_DIR, "saved")

MODEL_NAME = "klue/bert-base"

# the files.
TRAIN_CSV = path.join(DATA_DIR, "train.csv")
VAL_CSV = path.join(DATA_DIR, "val.csv")

IDIOM_VOCAB = ['함흥차사', '마부작침', '독서망양', '군계일학', '대기만성', '수어지교', '개과천선', '조령모개', '다다익선', '백전백승']

# the models
MONO_EN_CKPT = path.join(DATA_DIR, "lightning_logs/version_1/checkpoints/RD_epoch=00_train_loss=0.00.tmp_end.ckpt")