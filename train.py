from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from idiomify.datasets import Idiom2Def
from idiomify.loaders import load_train_dataset, load_val_dataset
from idiomify.models import ReverseDict
from idiomify.config import DATA_DIR, MODEL_NAME, IDIOM_VOCAB
import pytorch_lightning as pl
import torch
import argparse

def main():
    # --- arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int,
                        default=5)
    parser.add_argument("--lr", type=float,
                        default=0.0001)
    parser.add_argument("--max_epochs", type=int,
                        default=20)
    parser.add_argument("--batch_size", type=int,
                        default=32)
    parser.add_argument("--gpus", type=int,
                        default=2)

    args = parser.parse_args()
    k: int = args.k
    lr: float = args.lr
    max_epochs: int = args.max_epochs
    batch_size: int = args.batch_size
    gpus: int = args.gpus

    # --- instantiate the models & the data --- #
    train_dataset = load_train_dataset()
    val_dataset = load_val_dataset()


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert_mlm = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(['[IDIOM]'], special_tokens=True) #Add [IDIOM] token
    bert_mlm.resize_token_embeddings(len(tokenizer))
    
    model_name = "idiomify_{epoch:02d}_{train_loss:.2f}"
    train = Idiom2Def(train_dataset, tokenizer, k)
    val = Idiom2Def(val_dataset, tokenizer, k)
    
    model = ReverseDict(bert_mlm,tokenizer, train, val, IDIOM_VOCAB, k, lr, batch_size) 

    # --- init callbacks --- #
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=model_name
    )

    # --- instantiate the trainer --- #
    trainer = pl.Trainer(gpus=gpus,
                         max_epochs=max_epochs,
                         callbacks=[checkpoint_callback],
                         default_root_dir=DATA_DIR)
    # --- start training --- #
    trainer.fit(model)


if __name__ == '__main__':
    main()