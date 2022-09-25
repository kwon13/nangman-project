from pathlib import Path
from os import path

# the directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = path.join(PROJECT_ROOT, "data")
SAVED_DIR = path.join(DATA_DIR, "saved")

MODEL_NAME = "klue/roberta-base"

# the files.
TRAIN_CSV = path.join(DATA_DIR, "train_p_95.csv")
VAL_CSV = path.join(DATA_DIR, "val.csv")

# the idiom list
IDIOM_VOCAB = ['함흥차사', '마부작침', '독서망양', '군계일학', '대기만성', '수어지교', '개과천선', '조령모개', '다다익선', '백전백승', '종무소식', '수적천석', '망양지탄', '철중쟁쟁', '환골탈태', '조변석개', '마호체승', '백발백중', '감지덕지', '감언이설', '노심초사', '도탄지고', '불철주야', '삼인성호', '이구동성', '청빈낙도', '호각지세', '적반하장', '가담항설', '유언비어', '갑론을박', '과유불급', '구제불능', '권선징악', '기고만장', '노발대발', '논리정연', '다재다능', '동상이몽', '동병상련', '마이동풍', '무아지경', '새옹지마', '설상가상', '일촉즉발', '임기응변', '전화위복', '폭풍전야', '학수고대']

# the models
CKPT = path.join(DATA_DIR, "lightning_logs/version_4/checkpoints/idiomify_epoch=04_train_loss=0.02.tmp_end.ckpt")