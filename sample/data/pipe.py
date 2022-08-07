from fastNLP.io import  DataBundle, Pipe
from fastNLP import DataSet
from transformers.tokenization_bert import BertTokenizer
from .loader import EnLoader

def _prepare_data_bundle(tokenizer, data_bundle, max_word_len):
    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    sep_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    cls_id = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    pad_id = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    unk_id = tokenizer.convert_tokens_to_ids(['[UNK]'])[0]

    target_words = data_bundle.target_words
    word2bpes = []
    word2idx = {}
    for ins in data_bundle.get_dataset('train'):
        word = ins['word']      # ins['word'] = 사전 단어 ex) '안녕하세요', '한국어'...
        if word not in word2idx:
            word2idx[word] = len(word2idx)
            bpes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            word2bpes.append(bpes)

    number_word_in_train = len(word2idx)

    for word in target_words:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
            bpes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            word2bpes.append(bpes)

    for name in data_bundle.get_dataset_names():
        # (1) tokenize word 열 생성
        # (2) tokenize definition 열 생성
        ds = data_bundle.get_dataset(name) # ['desc', 'dev', 'seen', 'train', 'unseen']
        new_ds = DataSet()
        for ins in ds:
            word = ins['word']
            idx = word2idx[word]
            bpes = word2bpes[idx]
            definition = []
            for word in ins['definition'].split():
                definition.extend(tokenizer.tokenize(word))
            definition = tokenizer.convert_tokens_to_ids(definition)

            input = [cls_id] + [mask_id] * max_word_len + \
                    [sep_id] + definition
            # input: [CLS], [MASK], [MASK], [MASK], [MASK], [MASK], [SEP], [definition_ids] (max_word_len : 5)
            #       -> [definition_ids]을 보고 [MASK]에 들어갈 단어 추론
            input = input[:256] 
            input.append(sep_id) # input 최대 길이 = 255([CLS], [], [MASK]*k, ) + 1('[SEP]') = 256
            ins['input'] = input

            if unk_id in bpes:
                if name == 'train':
                    continue # bpes = target word 토큰
                else:
                    bpes = [0] * (max_word_len + 1)  # +1이 있는 이유는 모르겠음. 아마 이후 if문에서 train을 거르기 위해서?
            if len(bpes) <= max_word_len:
                ins['target'] = idx
                new_ds.append(ins)
            else:
                if name != 'train':
                    ins['target'] = -1
                    new_ds.append(ins)
        data_bundle.set_dataset(new_ds, name)
    
    for i in range(len(word2bpes)):
        bpes = word2bpes[i] # bpes = target word 토큰
        # 전 bpes:  [11655, 4279, 8553]
        bpes = bpes[:max_word_len] + [mask_id] * max(0, max_word_len - len(bpes))
        word2bpes[i] = bpes
        # 후 bpes:  [11655, 4279, 8553, 4, 4]

    data_bundle.set_pad_val('input', pad_id)
    data_bundle.set_input('input')
    data_bundle.set_target('target')
    setattr(data_bundle, 'word2bpes', word2bpes)
    setattr(data_bundle, 'pad_id', pad_id)
    setattr(data_bundle, 'number_word_in_train', number_word_in_train)
    setattr(data_bundle, 'word2idx', word2idx)

    return data_bundle


class ENBertPipe(Pipe):
    
    def __init__(self, bert_name, max_word_len=6):
        super().__init__()
        self.bert_name = bert_name
        self.max_word_len = max_word_len
        self.lower = True

    def process(self, data_bundle):
        tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        tokenizer.do_basic_tokenize = False
        return _prepare_data_bundle(tokenizer, data_bundle, self.max_word_len) # 토큰화

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = EnLoader().load(paths)
        return self.process(data_bundle)
