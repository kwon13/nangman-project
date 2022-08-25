import os
import json
from fastNLP.io import Loader, DataBundle
from fastNLP import Instance, DataSet


class EnLoader(Loader):
    def __init__(self):
        super().__init__()
          
    def load(self, folder):
        data_bundle = DataBundle()
        for name in ['desc.json', 'dev.json', 'seen.json', 'train.json', 'unseen.json']:
            path = os.path.join(folder, name)
            dataset = DataSet() # SQL로 치면 테이블 생성
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for d in data:
                    word = d['word'].lower()
                    definition = d['definitions'].lower()
                    ins = Instance(word=word, definition=definition)
                    # ins: 
                    # >>> ins = Instance(word=[1, 1, 1], definition=[2, 2, 2])
                    # +-----------+------------+
                    # | word      | definition |
                    # +-----------+------------+
                    # | [1, 1, 1] | [2, 2, 2]  |
                    # +-----------+------------+
                    # >>> ins['word']
                    # [1, 1, 1]
                    dataset.append(ins)
                    # >>> ins = Instance(word=[4], definition=[3])
                    # >>> dataset.append(ins)
                    # +-----------+------------+
                    # | word      | definition |
                    # +-----------+------------+
                    # | [1, 1, 1] | [2, 2, 2]  |
                    # | [4]       | [3]        |
                    # +-----------+------------+
                data_bundle.set_dataset(dataset, name=name.split('.')[0])
                
        words = []
        with open(os.path.join(folder, 'target_words.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    words.append(line)
        setattr(data_bundle, 'target_words', words)
        return data_bundle

