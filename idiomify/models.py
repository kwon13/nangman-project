from argparse import Namespace
from typing import Tuple, List
import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch import Tensor
from torch.optim import Optimizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from flash.core.optimizers import LAMB
from torch.nn import functional as F


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
        self.IDIOM_VOCAB = IDIOM_VOCAB

        # -- hyper params --- #
        self.save_hyperparameters(Namespace(k=k, lr=lr))

    def forward(self, X: Tensor) -> Tensor:
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        H_all = self.bert_mlm.roberta.forward(input_ids, attention_mask, token_type_ids)[0]  # (N, 3, L) -> (N, L, H)
        H_k = H_all[:, 1: self.hparams['k'] + 1]  # (N, L, H) -> (N, K, H)
        # only roberta has lm_head (bert: cls)
        S_subword = self.bert_mlm.lm_head(H_k)  # (N, K, H) ->  (N, K, |S|)
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
        # 독서망양 -> 독서, ##망, ##양, mask
        # [ ...,
        #   ...,
        #   ...
        #   [98, 122, 103, 103]]
        word2subs = word2subs
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
        # loss = F.cross_entropy(S_word, y)  # (N, |V|) -> (N,)
        loss = F.binary_cross_entropy(torch.sigmoid(S_word), y)

        loss = loss.sum()  # (N,) -> scalar
        self.log("train_loss",loss, prog_bar=True, logger = True)

        return {"loss" : loss}
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        X, y = batch

        S_subword = self.forward(X) 
        word2subs = self.build_word2subs(self.tokenizer, self.k)
        S_word = torch.sigmoid(self.S_word(word2subs, S_subword))  # (N, K, |S|) -> (N, |V|)
        loss = F.binary_cross_entropy(S_word, y)
        loss = loss.sum()  

        f1=f1_score(torch.round(S_word).cpu().tolist(), y.cpu().tolist(), average='micro')
        
        # self.log("val_loss", loss, prog_bar=True, logger = True)
        self.log_dict({"val_loss": loss, 'f1_score':f1}, prog_bar=True, logger = True)
        
        return {"val_loss": loss, 'f1_score':f1_score}
    

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
   
        
    def configure_optimizers(self):
        return LAMB(self.parameters(), lr=self.hparams['lr'], amsgrad=True)