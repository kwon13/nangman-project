from typing import List, Tuple
from idiomify.config import TRAIN_CSV, VAL_CSV
import pandas as pd

def load_train_dataset()-> List[List[str]]:
    
    data={}
    frame = pd.read_csv(TRAIN_CSV)
    
    data_keys = frame.idiom.unique()
    for k in data_keys:
        data[k] = []
        
    for idx in range(len(frame)):
        data[frame.idiom[idx]].append(frame.contents[idx])

    for k in data:
        data[k].insert(0, k)


    return list(data.values())

def load_val_dataset()-> List[List[str]]:
    
    data={}
    frame = pd.read_csv(VAL_CSV)
    
    data_keys = frame.idiom.unique()
    for k in data_keys:
        data[k] = []
        
    for idx in range(len(frame)):
        data[frame.idiom[idx]].append(frame.contents[idx])

    for k in data:
        data[k].insert(0, k)


    return list(data.values())
    