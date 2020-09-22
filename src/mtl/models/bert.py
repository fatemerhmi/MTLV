from transformers import BertPreTrainedModel, BertModel
from torch import nn

__all__ = ['bert_base_uncased', 'bert_large_uncased', 'bert_base_cased', 'bert_large_cased']

class BertCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return pooled_output
    
def bert_base_uncased():
    model =  BertCLS.from_pretrained('bert-base-uncased')
    model.embedding_size = 768
    return model

def bert_large_uncased():
    model = BertCLS.from_pretrained('bert-base-uncased')
    model.embedding_size = 1024
    return model


def bert_base_cased():
    model = BertCLS.from_pretrained('bert-base-cased')
    model.embedding_size = 768
    return model

def bert_large_cased():
    model = BertCLS.from_pretrained('bert-large-cased')
    model.embedding_size = 1024
    return model