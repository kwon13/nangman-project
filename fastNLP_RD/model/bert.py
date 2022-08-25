from torch import nn
import torch
from transformers import RobertaForMaskedLM
from transformers import BertForMaskedLM

class RDRobertaForMaskedLM(RobertaForMaskedLM):
    def set_start_end(self, start=1, end=5):
        self.start = start
        self.end = end

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output[:, self.start:self.end])

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

class ENRobertaReverseDict(nn.Module):
    def __init__(self, pre_name, word2bpes, pad_id, number_word_in_train):
        super().__init__()
        self.roberta_model = RDRobertaForMaskedLM.from_pretrained(pre_name)
        self.roberta_model.set_start_end(1, 1+len(word2bpes[0]))
        self.max_word_len = len(word2bpes[0])
        # 1 x 1 x vocab_size
        word2bpes = torch.LongTensor(word2bpes).transpose(0, 1).unsqueeze(0) 
        self.register_buffer('word2bpes', word2bpes)
        self.number_word_in_train = number_word_in_train
        self.pad_id = pad_id

    def forward(self, input):
        """
        input 형식: cls + mask + sep_id + definition
        """
        attention_mask = input.ne(self.pad_id)

        #  batch_size x max_len x vocab_size
        bpe_reps = self.roberta_model(input_ids=input, token_type_ids=None,
                                                        attention_mask=attention_mask)[0]

        # bsz x max_word_len x word_vocab_size
        word2bpes = self.word2bpes.repeat(bpe_reps.size(0), 1, 1)
        word_scores = bpe_reps.gather(dim=-1, index=word2bpes)   # bsz x max_word_len x word_vocab_size

        word_scores = word_scores.sum(dim=1)
        if self.training and self.number_word_in_train is not None:
            word_scores = word_scores[:, :self.number_word_in_train]

        return {'pred': word_scores}
