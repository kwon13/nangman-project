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
        # (N, 3, L)
        self.X = self.build_X(idiom2Def, tokenizer, k)
        # (N,)
        self.y = self.build_y(idiom2Def, classes)

    @staticmethod
    def build_X(idiom2Def: List[Tuple[str, str]], tokenizer: AutoTokenizer, k: int):
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
                            # token type for the padded tokens? -> they are masked with the
                            # attention mask anyways
                            # https://github.com/google-research/bert/issues/635#issuecomment-491485521
                            encodings['token_type_ids'],
                            encodings['attention_mask']], dim=1)

    @staticmethod
    def build_y(idiom2Def: List[Tuple[str, str]], classes: List[str]):
        fruits = [fruit for fruit, _ in idiom2Def]
        return Tensor([
            classes.index(fruit)
            for fruit in fruits
        ]).long()

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
            for idiom_def in row[3:]
        ]
