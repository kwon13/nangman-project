import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICE'] = os.environ['p']

import torch
from torch import optim
from fastNLP import Tester, CrossEntropyLoss
from fastNLP import BucketSampler, cache_results, WarmupCallback, GradientClipCallback, FitlogCallback #fastNLP: high-level interface
from model.bert import ENRobertaReverseDict
import fitlog
from model.metrics import MonoMetric
from data.pipe import ENRobertaPipe
fitlog.set_log_dir('en_logs') 
fitlog.add_hyper_in_file(__file__) 
fitlog.add_other('uncased', name='note')

paths = './data/'
#######hyper
model_name = 'roberta'
max_word_len = 5
lr = 2e-5
batch_size = 64
n_epochs = 10
#######hyper
pre_name = 'klue/roberta-base'


def get_data():
    data_bundle = ENRobertaPipe(pre_name, max_word_len).process_from_file(paths)
    return data_bundle

data_bundle = get_data()
word2bpes = data_bundle.word2bpes
print(f"In total {len(word2bpes)} target words")
pad_id = data_bundle.pad_id

model = ENRobertaReverseDict(pre_name, word2bpes, pad_id=pad_id,
                          number_word_in_train=None)

if torch.cuda.is_available():
    model.cuda()

data = {}
for name in ['test']:
    data[name] = data_bundle.get_dataset(name)


sampler = BucketSampler()


tester = Tester(data=data['test'],model=model, metrics=MonoMetric())

eval_results = tester.test()