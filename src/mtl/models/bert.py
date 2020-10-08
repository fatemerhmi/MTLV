from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput 

__all__ = ['bert_base_uncased', 'bert_large_uncased', 'bert_base_cased', 'bert_large_cased']

class BertCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
        self.hidden_size_BertCLS = config.hidden_size


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

class BertCLS_multilabel_singleHead(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size_BertCLS = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # nn.init.xavier_normal_(self.classifier.weight)
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
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        loss_func = BCEWithLogitsLoss() 
        # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        loss = loss_func(logits.view(-1, self.num_labels), labels.type_as(logits).view(-1, self.num_labels))
        

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # def freeze_bert_encoder(self):
    #     for param in self.bert.parameters():
    #         param.requires_grad = False
        # for name, param in self.bert.named_parameters():
        #     if 'classifier' not in name: # classifier layer
        #         param.requires_grad = False
    # def unfreeze_bert_encoder(self):
    #     for param in self.bert.parameters():
    #         param.requires_grad = True

class BertCLS_multilabel_MTL(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size_BertCLS = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # nn.init.xavier_normal_(self.classifier.weight)
        self.init_weights()

        self.nhead = config.nhead # number of heads
        self.classifiers = [nn.Linear(config.hidden_size, config.num_labels) for x in range(self.nhead)]

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
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits_heads = [clf(pooled_output) for clf in self.classifiers]
        # logits = self.classifier(pooled_output)

        loss = None
        loss_func = BCEWithLogitsLoss()
        losses = [loss_func(logits.view(-1, self.num_labels), labels.type_as(logits).view(-1, self.num_labels)) for logits in logits_heads]
        # loss += []
        # loss = loss_func(logits.view(-1, self.num_labels), labels.type_as(logits).view(-1, self.num_labels))
        

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=losses,
            logits=logits_heads,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def bert_base_uncased(num_labels, training_type, nhead = None):
    if training_type == "singlehead_cls":
        # model =  BertCLS.from_pretrained('bert-base-uncased')
        model =  BertCLS_multilabel_singleHead.from_pretrained('bert-base-uncased', num_labels = num_labels, return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS
        return model
    elif training_type == "MTL_cls":
        if nhead is None:
            raise Exception("number of heads 'nhead' must be more than 1!")
        else:
            model =  BertCLS_multilabel_MTL.from_pretrained('bert-base-uncased', num_labels = num_labels, nhead = nhead, return_dict=True)
            model.embedding_size = model.hidden_size_BertCLS
            return model

def bert_large_uncased():
    model = BertCLS.from_pretrained('bert-base-uncased')
    model.embedding_size = model.hidden_size_BertCLS
    return model


def bert_base_cased():
    model = BertCLS.from_pretrained('bert-base-cased')
    model.embedding_size = model.hidden_size_BertCLS
    return model

def bert_large_cased():
    model = BertCLS.from_pretrained('bert-large-cased')
    model.embedding_size = model.hidden_size_BertCLS
    return model