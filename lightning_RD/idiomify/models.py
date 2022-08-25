"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from argparse import Namespace
from typing import Tuple, List
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch.optim import Optimizer
from transformers import AutoTokenizer, AutoModelForMaskedLM

from torch.nn import functional as F
from idiomify.datasets import Idiom2DefDataset

class ReverseDict(pl.LightningModule):
    """
    A reverse-dictionary.
    """
    def __init__(self, bert_mlm: AutoModelForMaskedLM, tokenizer: AutoTokenizer, train_data, val_data, IDIOM_VOCAB: list, k: int, lr: float, batch_size:int):
        super().__init__()
        self.bert_mlm = bert_mlm
        self.tokenizer = tokenizer
        self.k = k
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        # -- to be used to compute S_word -- #
        self.IDIOM_VOCAB = IDIOM_VOCAB
        
        # -- hyper params --- #
        # should be saved to self.hparams
        self.save_hyperparameters(Namespace(k=k, lr=lr))

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (N, 3, L) (num samples, 0=input_ids/1=token_type_ids/2=attention_mask, the maximum length)
        :return: (N, K, |V|); (num samples, k, the size of the vocabulary of subwords)
        """
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        H_all = self.bert_mlm.bert.forward(input_ids, attention_mask, token_type_ids)[0]  # (N, 3, L) -> (N, L, 768)
        H_k = H_all[:, 1: self.hparams['k'] + 1]  # (N, L, 768) -> (N, K, 768)
        S_subword = self.bert_mlm.cls(H_k)  # (N, K, 768) ->  (N, K, |S|)
        return S_subword

    def build_word2subs(self, tokenizer: AutoTokenizer, k: int) -> Tensor:
        vocab = self.IDIOM_VOCAB
        mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
        pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
        encoded = tokenizer(text=vocab,
                            add_special_tokens=False,
                            padding='max_length',
                            max_length=k,  # set to k
                            return_tensors="pt")
        input_ids = encoded['input_ids']
        input_ids[input_ids == pad_id] = mask_id  # replace them with masks
        return input_ids
    
    def S_word(self,word2subs, S_subword: Tensor) -> Tensor:
        # pineapple -> pine, ###apple, mask, mask, mask, mask, mask
        # [ ...,
        #   ...,
        #   ...
        #   [98, 122, 103, 103]]
        word2subs = word2subs.to('cuda')
        word2subs = word2subs.T.repeat(S_subword.shape[0], 1, 1)  # (|V|, K) -> (N, K, |V|)
        S_word = S_subword.gather(dim=-1, index=word2subs)  # (N, K, |S|) -> (N, K, |V|)
        S_word = S_word.sum(dim=1)  # (N, K, |V|) -> (N, |V|)
        return S_word

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        :param batch: A tuple of X, y, subword_ids; ((N, 3, L), (N,),
        :param batch_idx: the index of the batch
        :return: (1,); the loss for this batch
        """
        X, y = batch

        S_subword = self.forward(X)  # (N, 3, L) -> (N, K, |S|)
        word2subs = self.build_word2subs(self.tokenizer, self.k)
        S_word = self.S_word(word2subs, S_subword)  # (N, K, |S|) -> (N, |V|)
        loss = F.cross_entropy(S_word, y)  # (N, |V|) -> (N,)
        loss = loss.sum()  # (N,) -> scalar
        self.log("train_loss",loss, prog_bar=True, logger = True)

        return {"loss" : loss}
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        X, y = batch

        S_subword = self.forward(X) 
        word2subs = self.build_word2subs(self.tokenizer, self.k)
        S_word = self.S_word(word2subs, S_subword)  # (N, K, |S|) -> (N, |V|)
        loss = F.cross_entropy(S_word, y)
        loss = loss.sum()  
        self.log("val_loss", loss, prog_bar=True, logger = True)
        
        return {"val_loss": loss}
    

    def train_dataloader(self):
        return DataLoader(self.train_data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=2
                            )
    
    def val_dataloader(self):
        return DataLoader(self.val_data,
                        batch_size=self.batch_size,
                        shuffle=True,
                            )
    
    def configure_optimizers(self) -> Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])