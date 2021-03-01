from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput 
import os

__all__ = ['bert_base_uncased', 'bert_base_cased', \
'BioBERT_Basev1_1', 'BioBERT_Basev1_0_PM', 'BioBERT_Basev1_0_PMC', 'BioBERT_Basev1_0_PM_PMC',  \
"bert_base_news", "bert_base_openI"]

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
        # pooled_output = self.dropout(pooled_output)
        return pooled_output

class BertCLS_binarycls(BertPreTrainedModel):
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
        if labels is not None:            
            loss_fct = BCEWithLogitsLoss() 
            loss = loss_fct(logits.view(-1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            loss_fct = BCEWithLogitsLoss() 
            loss = loss_fct(logits.view(-1, self.num_labels), labels.type_as(logits).view(-1, self.num_labels))

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
    def __init__(self, config, num_labels2):
        super().__init__(config)
        self.num_labels_list = num_labels2[0]
        self.device1 = num_labels2[1]

        self.hidden_size_BertCLS = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # nn.init.xavier_normal_(self.classifier.weight)
        self.init_weights()

        self.nhead = len(self.num_labels_list) # number of heads
        self.classifiers = [nn.Linear(config.hidden_size, num_labels).to(self.device1) for num_labels in self.num_labels_list]

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

        losses = None
        if labels is not None:
            loss_func = BCEWithLogitsLoss()
            # fix the labels here
            # losses = [loss_func(logits.view(-1, self.num_labels), labels.type_as(logits).view(-1, self.num_labels)) for logits, num_labels in zip(logits_heads, self.num_labels_list, )]
            # loss = loss_func(logits.view(-1, self.num_labels), labels.type_as(logits).view(-1, self.num_labels))
            losses = torch.zeros(self.nhead)
            for i, logits, num_labels in zip(range(0,self.nhead), logits_heads, self.num_labels_list):
                #remove -1 paddings:
                head_labels = labels[:,i,:]
                head_labels = head_labels[:,0:self.num_labels_list[i]]

                loss = loss_func(logits.view(-1, num_labels), head_labels.type_as(logits).view(-1, num_labels))
                losses[i] = loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=losses,
            logits=logits_heads,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

#------------------------dataset specific------------------------
def bert_base_openI(num_labels, training_type, device='gpu'):
    MODEL_PATH = "model_weights/openI_BERT3"

    if training_type == "singlehead_cls":
        model =  BertCLS_multilabel_singleHead.from_pretrained(MODEL_PATH, num_labels = num_labels, return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS

    elif training_type == "MTL_cls":
        model =  BertCLS_multilabel_MTL.from_pretrained(MODEL_PATH, num_labels2 = [num_labels, device], return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS

    elif training_type == "emb_cls":
        model = BertModel.from_pretrained(MODEL_PATH, return_dict=True)
    else:
        NameError(f"No further tuned model found at {MODEL_PATH}")

    return model

def bert_base_news(num_labels, training_type, device='gpu'):
    MODEL_PATH = "model_weights/news_BERT"

    if training_type == "singlehead_cls":
        model =  BertCLS_multilabel_singleHead.from_pretrained(MODEL_PATH, num_labels = num_labels, return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS

    elif training_type == "MTL_cls":
        model =  BertCLS_multilabel_MTL.from_pretrained(MODEL_PATH, num_labels2 = [num_labels, device], return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS

    elif training_type == "emb_cls":
        model = BertModel.from_pretrained(MODEL_PATH, return_dict=True)
    else:
        NameError(f"No further tuned model found at {MODEL_PATH}")

    return model

#-----------------------general --------------------------------
def bert_base_uncased(num_labels, training_type, device='gpu'):
    MODEL_PATH = "model_weights/bert-base-uncased"
    if not os.path.exists(MODEL_PATH):
        if training_type == "STL_cls":
            model =  BertCLS_binarycls.from_pretrained('bert-base-uncased', num_labels = 1, return_dict=True)
            model.embedding_size = model.hidden_size_BertCLS

        elif training_type == "MTL_cls" or training_type == "GMTL_cls":
            model =  BertCLS_multilabel_singleHead.from_pretrained('bert-base-uncased', num_labels = num_labels, return_dict=True)
            model.embedding_size = model.hidden_size_BertCLS

        elif training_type == "GMHL_cls":
            model =  BertCLS_multilabel_MTL.from_pretrained('bert-base-uncased', num_labels2 = [num_labels, device], return_dict=True)
            model.embedding_size = model.hidden_size_BertCLS

        elif training_type == "emb_cls":
            model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    
    else:
        if training_type == "STL_cls":
            model =  BertCLS_binarycls.from_pretrained(MODEL_PATH, num_labels = 1, return_dict=True)
            model.embedding_size = model.hidden_size_BertCLS

        elif training_type == "MTL_cls" or training_type == "GMTL_cls":
            model =  BertCLS_multilabel_singleHead.from_pretrained(MODEL_PATH, num_labels = num_labels, return_dict=True)
            model.embedding_size = model.hidden_size_BertCLS

        elif training_type == "GMHL_cls":
            model =  BertCLS_multilabel_MTL.from_pretrained(MODEL_PATH, num_labels2 = [num_labels, device], return_dict=True)
            model.embedding_size = model.hidden_size_BertCLS

        elif training_type == "emb_cls":
            model = BertModel.from_pretrained(MODEL_PATH, return_dict=True)


        # if training_type == "singlehead_cls":
        #     model =  BertCLS_multilabel_singleHead.from_pretrained(MODEL_PATH, num_labels = num_labels, return_dict=True)
        #     model.embedding_size = model.hidden_size_BertCLS

        # elif training_type == "MTL_cls":
        #     model =  BertCLS_multilabel_MTL.from_pretrained(MODEL_PATH, num_labels2 = [num_labels, device], return_dict=True)
        #     model.embedding_size = model.hidden_size_BertCLS

        # elif training_type == "emb_cls":
        #     model = BertModel.from_pretrained(MODEL_PATH, return_dict=True)
        
        # elif training_type == "STL_cls":
        #     model =  BertCLS_binarycls.from_pretrained(MODEL_PATH, num_labels = 1, return_dict=True)
        #     model.embedding_size = model.hidden_size_BertCLS

    return model

def bert_base_cased(num_labels, training_type, device='gpu'):
    MODEL_PATH = "model_weights/bert-base-cased"
    if not os.path.exists(MODEL_PATH):
        if training_type == "singlehead_cls":
            model =  BertCLS_multilabel_singleHead.from_pretrained('bert-base-cased', num_labels = num_labels, return_dict=True)
            model.embedding_size = model.hidden_size_BertCLS

        elif training_type == "MTL_cls":
            model =  BertCLS_multilabel_MTL.from_pretrained('bert-base-cased', num_labels2 = [num_labels, device], return_dict=True)
            model.embedding_size = model.hidden_size_BertCLS

        elif training_type == "emb_cls":
            model = BertModel.from_pretrained('bert-base-cased', return_dict=True)
    
    else:
        if training_type == "singlehead_cls":
            # model =  BertCLS.from_pretrained('bert-base-uncased')
            model =  BertCLS_multilabel_singleHead.from_pretrained(MODEL_PATH, num_labels = num_labels, return_dict=True)
            model.embedding_size = model.hidden_size_BertCLS
            
        elif training_type == "MTL_cls":
            model =  BertCLS_multilabel_MTL.from_pretrained(MODEL_PATH, num_labels2 = [num_labels, device], return_dict=True)
            model.embedding_size = model.hidden_size_BertCLS

        elif training_type == "emb_cls":
            model = BertModel.from_pretrained(MODEL_PATH, return_dict=True)

    return model


def BioBERT_Basev1_1(num_labels, training_type, device='gpu'):
    MODEL_PATH = "model_weights/biobert_v1.1_pubmed"
    if not os.path.exists(f"{MODEL_PATH}/pytorch_model.bin"):
        raise NameError(f'Could not find {MODEL_PATH}/pytorch_model.bin! Download the BioBERT model and store them in model_weights directory, then run the script to convert it to pytorch weights.')
    
    if training_type == "singlehead_cls":
        # model =  BertCLS.from_pretrained('bert-base-uncased')
        model =  BertCLS_multilabel_singleHead.from_pretrained(MODEL_PATH, num_labels = num_labels, return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS

    elif training_type == "MTL_cls":
        model =  BertCLS_multilabel_MTL.from_pretrained(MODEL_PATH, num_labels2 = [num_labels, device], return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS

    elif training_type == "emb_cls":
        model = BertModel.from_pretrained(MODEL_PATH, return_dict=True)

    return model
def BioBERT_Basev1_0_PM(num_labels, training_type, device='gpu'):
    MODEL_PATH = "model_weights/biobert_v1.0_pubmed"
    if not os.path.exists(f"{MODEL_PATH}/pytorch_model.bin"):
        raise NameError(f'Could not find {MODEL_PATH}/pytorch_model.bin! Download the BioBERT model and store them in model_weights directory, then run the script to convert it to pytorch weights.')
    
    if training_type == "singlehead_cls":
        # model =  BertCLS.from_pretrained('bert-base-uncased')
        model =  BertCLS_multilabel_singleHead.from_pretrained(MODEL_PATH, num_labels = num_labels, return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS

    elif training_type == "MTL_cls":
        model =  BertCLS_multilabel_MTL.from_pretrained(MODEL_PATH, num_labels2 = [num_labels, device], return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS
    elif training_type == "emb_cls":
        model = BertModel.from_pretrained(MODEL_PATH, return_dict=True)

    return model

def BioBERT_Basev1_0_PMC(num_labels, training_type, device='gpu'):
    MODEL_PATH = "model_weights/biobert_v1.0_pmc"
    if not os.path.exists(f"{MODEL_PATH}/pytorch_model.bin"):
        raise NameError(f'Could not find {MODEL_PATH}/pytorch_model.bin! Download the BioBERT model and store them in model_weights directory, then run the script to convert it to pytorch weights.')
    
    if training_type == "singlehead_cls":
        # model =  BertCLS.from_pretrained('bert-base-uncased')
        model =  BertCLS_multilabel_singleHead.from_pretrained(MODEL_PATH, num_labels = num_labels, return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS

    elif training_type == "MTL_cls":
        model =  BertCLS_multilabel_MTL.from_pretrained(MODEL_PATH, num_labels2 = [num_labels, device], return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS

    elif training_type == "emb_cls":
        model = BertModel.from_pretrained(MODEL_PATH, return_dict=True)

    return model

def BioBERT_Basev1_0_PM_PMC(num_labels, training_type, device='gpu'):
    MODEL_PATH = "model_weights/biobert_v1.0_pubmed_pmc"
    if not os.path.exists(f"{MODEL_PATH}/pytorch_model.bin"):
        raise NameError(f'Could not find {MODEL_PATH}/pytorch_model.bin! Download the BioBERT model and store them in model_weights directory, then run the script to convert it to pytorch weights.')
    
    if training_type == "singlehead_cls":
        # model =  BertCLS.from_pretrained('bert-base-uncased')
        model =  BertCLS_multilabel_singleHead.from_pretrained(MODEL_PATH, num_labels = num_labels, return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS

    elif training_type == "MTL_cls":
        model =  BertCLS_multilabel_MTL.from_pretrained(MODEL_PATH, num_labels2 = [num_labels, device], return_dict=True)
        model.embedding_size = model.hidden_size_BertCLS

    elif training_type == "emb_cls":
        model = BertModel.from_pretrained(MODEL_PATH, return_dict=True)

    return model
