import argparse
from typing import Tuple, List
import torch
from torch import Tensor
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
from idiomify.models import ReverseDict
from idiomify.config import CKPT, MODEL_NAME, IDIOM_VOCAB
from torch.nn import functional as F
from idiomify.datasets import Idiom2DefDataset


class Idiomifier:
    def __init__(self, rd: ReverseDict, tokenizer: AutoTokenizer, IDIOM_VOCAB: List[str], k:int, gpu:bool):
        self.rd = rd  # a trained reverse dictionary
        self.tokenizer = tokenizer
        self.k = k
        self.IDIOM_VOCAB = IDIOM_VOCAB
            
    
    def build_word2subs(self, tokenizer: AutoTokenizer, k: int) -> Tensor:
        vocab = self.IDIOM_VOCAB
        mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
        pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
        encoded = tokenizer(text=vocab,
                            add_special_tokens=False,
                            padding='max_length',
                            max_length=k,
                            return_tensors="pt")
        input_ids = encoded['input_ids']
        input_ids[input_ids == pad_id] = mask_id
        return input_ids
    
    def S_word(self,word2subs, S_subword: Tensor) -> Tensor:
        word2subs = word2subs.T.repeat(S_subword.shape[0], 1, 1)
        S_word = S_subword.gather(dim=-1, index=word2subs)  
        S_word = S_word.sum(dim=1)
        return S_word
    
    def idiomify(self, descriptions: List[str], *args, **kwargs) -> List[List[Tuple[str, float]]]:
        # get the X
        idiom2def = [("", desc) for desc in descriptions]
        X = Idiom2DefDataset.build_X(idiom2def, tokenizer=self.tokenizer, k=self.rd.hparams['k'])
        S_subword = self.rd.forward(X)
        word2subs = self.build_word2subs(self.tokenizer, self.k)
        S_word = self.S_word(word2subs, S_subword) 
        S_word_probs = F.sigmoid(S_word) 
        results = list()
        for w_probs in S_word_probs.tolist():
            idiom2score = [
                (idiom, w_score)
                for idiom, w_score in zip(self.IDIOM_VOCAB, w_probs)
            ]
            # sort and append
            results.append(sorted(idiom2score, key=lambda x: x[1], reverse=True))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int,
                        default=5)
    parser.add_argument("--desc", type=str,
                        default="진흙에서 피는 꽃")
    parser.add_argument("--gpu", type=bool,
                        default=True)
    
    args = parser.parse_args()
    k:int = args.k
    desc: str = args.desc
    gpu:bool = args.gpu

    bert_mlm = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(['[IDIOM]'], special_tokens=True) #Add [IDIOM] token
    bert_mlm.resize_token_embeddings(len(tokenizer))
    
    rd = ReverseDict(bert_mlm, tokenizer, None, None, IDIOM_VOCAB, k, None, None) 
    checkpoint = torch.load(CKPT)
    rd.load_state_dict(checkpoint, strict=False)
    rd.eval()
    idiomifier = Idiomifier(rd, tokenizer, IDIOM_VOCAB, k, gpu)
    print("### desc: {} ###".format(desc))
    for results in idiomifier.idiomify(descriptions=[desc]):
        for idx, res in enumerate(results):
            print("{}:".format(idx), res)


if __name__ == '__main__':
    main()
