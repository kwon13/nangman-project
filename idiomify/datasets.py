from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from idiomify.config import IDIOM_VOCAB

class Idiom2DefDataset(Dataset):
    def __init__(self,
                 idiom2Def: List[Tuple[str, str]],
                 tokenizer: AutoTokenizer,
                 k: int,
                 classes: List[str]):
        # (N, 3(input_ids,token_type_ids,attention_mask), L)
        self.X = self.build_X(idiom2Def, tokenizer, k)
        # (N, L)
        self.y = self.build_y(idiom2Def, classes)

    @staticmethod
    def build_X(idiom2Def: List[Tuple[str, str]], tokenizer: AutoTokenizer, k: int):
        """
        Returning x_data
        Args:
            idiom2Def: List((idiom_1,def_1), (idiom_1,def_2), ... ,(idiom_n,def_m))
            tokenizer: AutoTokenizer
            k:idiom의 subword의 최대 길이
        
        Returns:
            {input_ids:[ids_token,...],
            token_type_ids:[1,1,1,..0,0...],
            attention_mask:[1,1,1...]}
        """
        defs = [def_ for _, def_ in idiom2Def]
        lefts = [" ".join(["[MASK]"] * k)] * len(defs)
        rights = defs
        encodings = tokenizer(text=lefts,
                              text_pair=rights,
                              return_tensors="pt",
                              add_special_tokens=True,
                              truncation=True,
                              padding=True)

        return torch.stack([encodings['input_ids'],
                            encodings['token_type_ids'],
                            encodings['attention_mask']], dim=1)

    @staticmethod
    def build_y(idiom2Def: List[Tuple[str, str]], classes: List[str]):
        """
        Returning y_data
        Args:
            idiom2Def: List((idiom_1,def_1), (idiom_1,def_2), ... ,(idiom_n,def_m))
            classes: List(idiom_1,idiom_2,...idiom_n)
        
        Returns:
            multi-hot encoding:[[1,0,1,0...],
                                [0,0,0,0...]
                                ...]
        """
        encodings = []
        for idioms, _ in idiom2Def:
            encoding = torch.tensor([0]*len(classes)).float()
            for idiom in idioms.split(','):
                idiom = idiom.strip()
                encoding[classes.index(idiom)] = 1
            encodings.append(encoding)
        
        return torch.stack(encodings).float()

    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.X[idx], self.y[idx]


class Idiom2Def(Idiom2DefDataset):

    def __init__(self, idiomify_dataset: List[List[str]], tokenizer: AutoTokenizer, k: int):
        classes = IDIOM_VOCAB
        idiom2Def = self.to_idiom2Def(idiomify_dataset)
        super().__init__(idiom2Def, tokenizer, k, classes)

    @staticmethod
    def to_idiom2Def(idiomify_dataset) -> List[Tuple[str, str]]:
        return [
            (str(row[0]).strip(), str(idiom_def).strip())
            for row in idiomify_dataset
            for idiom_def in row[1:] # idiom idx = [0], contents idx = [1:]
        ]
