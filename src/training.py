from tqdm import tqdm, trange
import ast
import torch
import numpy as np

import mtl.utils.configuration as configuration
import mtl.utils.logger as mlflowLogger
from mtl.heads.clsHeads import *
from mtl.utils.evaluate import *
import mtl.utils.logger as mlflowLogger 

def train(train_dataloader, val_dataloader, test_dataloader, model, training_args, use_cuda, cfg_optimizer):
    #-------training args:
    if use_cuda:
        print('[  training  ] Running on GPU.')
    else:
        print('[  training  ] Running on CPU.')

    training_type = training_args['type']
    epoch = training_args['epoch']

    #-------training type:
    if training_type == "multihead_cls":
        print(f"[  training  ] The training type is: Multi-head classification.")
        multihead_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer)


def multihead_cls(train_dataloader, validation_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer):
    #-------get params from mlflow
    col_names = ast.literal_eval(mlflowLogger.get_params("col_names"))
    heads_index = ast.literal_eval(mlflowLogger.get_params("heads_index"))

    #-------load heads
    head_count = [len(group) for group in heads_index]
    nheads = len(head_count)

    #-------load model
    if use_cuda:
        device = "cuda"
        model.cuda()
    else:
        device = "cpu"

    #-------load optimizer
    optimizer = configuration.setup_optimizer(cfg_optimizer, model.parameters())
    # optimizer = optimizer_obj(model.parameters(),lr=2e-5)  # Default optimization
    
    #-------FineTune model
    # trange is a tqdm wrapper around the normal python range
    for e in trange(epoch, desc="Epoch"):
        print("epoch", e)
        #==========Training======================
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0 #running loss
        train_headloss = np.zeros(3)
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()

            #Forward pass for multihead
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  
            head_losses = []
            for i in range(0,nheads):
                
                #remove -1 paddings:
                labels = b_labels[:,i,:]
                labels = labels[:,0:head_count[i]]
                
                hparams={
                    'labels' : labels,
                    'num_labels' : len(heads_index[i]),
                    'input_size' : outputs[0].size()[0],
                    'inputLayer' : outputs,
                    'device'     : device,
                }
                head_losses.append(HeadMultilabelCLS(hparams))
            
            loss = 0
            batch_traning_headloss = []
            for head in head_losses:
                loss += head.loss 
                batch_traning_headloss.append(head.loss)
            train_headloss = np.sum([train_headloss, batch_traning_headloss], axis=0)
            # Backward pass
            loss.backward()
            
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        train_headloss = train_headloss/nb_tr_steps
        index = 0
        for headloss in train_headloss:
            mlflowLogger.store_metric(f"training.headloss.{index}", headloss.item(), e)
            index +=1
        mlflowLogger.store_metric("training.loss", tr_loss/nb_tr_steps, e)
        # print("Train loss: {}".format(tr_loss/nb_tr_steps))

        #===========Validation==========
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Variables to gather full output
        true_labels_each_head,pred_labels_each_head = [],[]
        true_labels_all_head,pred_labels_all_head = [],[]
        
        # Predict
        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  

                pred_label_heads = []
                true_label_heads = []
                for i in range(0,nheads):
                    #remove -1 paddings:
                    labels = b_labels[:,i,:]
                    labels = labels[:,0:head_count[i]]

                    hparams={
                        'labels' : labels,
                        'num_labels' : len(heads_index[i]),
                        'input_size' : outputs[0].size()[0],
                        'inputLayer' : outputs,
                        'device'     : device,
                    }
                    pred_label_heads.append(HeadMultilabelCLS(hparams).pred_label)
                    true_label_heads.append(labels)
                
                #store batch labels 
                pred_label_b = np.array(pred_label_heads)
                true_labels_b = np.array(true_label_heads)

                #store each head label seperatly
                true_labels_each_head.append(true_labels_b)
                pred_labels_each_head.append(pred_label_b)

                #store all head labels together
                true_labels_all_head.append(
                    torch.cat([true_head_label for true_head_label in true_labels_b],1)
                    .to('cpu').numpy())
                pred_labels_all_head.append(
                    torch.cat([pred_head_label for pred_head_label in pred_label_b],1)
                    .to('cpu').numpy())

        # print("Results of all heads:")
        true_labels_all_head = np.concatenate([item for item in true_labels_all_head])
        pred_labels_all_head = np.concatenate([item for item in pred_labels_all_head])
        val_f1, val_acc, _ = calculate_f1_acc(pred_labels_all_head, true_labels_all_head)

        mlflowLogger.store_metric("validation.acc", val_acc, e)
        mlflowLogger.store_metric("validation.f1", val_f1, e)

        true_labels_each_head = np.array(true_labels_each_head)
        pred_labels_each_head = np.array(pred_labels_each_head)
        
        # print("Results of each head:")
        for i in range(0,nheads):
            # print(f"Head_{i}")
            i_head_true_labels = true_labels_each_head[:,i]
            i_head_true_labels = torch.cat([item for item in i_head_true_labels],0).to('cpu').numpy()
            
            i_head_pred_labels = pred_labels_each_head[:,i]
            i_head_pred_labels = torch.cat([item for item in i_head_pred_labels],0).to('cpu').numpy()

            val_head_f1, val_head_acc, _ = calculate_f1_acc(i_head_pred_labels, i_head_true_labels)
            mlflowLogger.store_metric(f"validation.headacc.{i}", val_head_acc, e)
            mlflowLogger.store_metric(f"validation.headf1.{i}", val_head_f1, e)

    #============test=============
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    #track variables
    true_labels_each_head,pred_labels_each_head = [],[]
    true_labels_all_head,pred_labels_all_head = [],[]
    # Predict
    for i, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  

            pred_label_heads = []
            true_label_heads = []
            for i in range(0,nheads):
                #remove -1 paddings:
                labels = b_labels[:,i,:]
                labels = labels[:,0:head_count[i]]

                hparams={
                    'labels' : labels,
                    'num_labels' : len(heads_index[i]),
                    'input_size' : outputs[0].size()[0],
                    'inputLayer' : outputs,
                    'device'     : device,
                }
                pred_label_heads.append(HeadMultilabelCLS(hparams).pred_label)
                true_label_heads.append(labels)

            #store batch labels 
            pred_label_b = np.array(pred_label_heads)
            true_labels_b = np.array(true_label_heads)

            #store each head label seperatly
            true_labels_each_head.append(true_labels_b)
            pred_labels_each_head.append(pred_label_b)

            #store all head labels together
            true_labels_all_head.append(
                torch.cat([true_head_label for true_head_label in true_labels_b],1)
                .to('cpu').numpy())
            pred_labels_all_head.append(
                torch.cat([pred_head_label for pred_head_label in pred_label_b],1)
                .to('cpu').numpy())

    true_labels_all_head = np.concatenate([item for item in true_labels_all_head])
    pred_labels_all_head = np.concatenate([item for item in pred_labels_all_head])
    
    test_f1, test_acc, test_LRAP, test_clf_report = calculate_f1_acc_test(pred_labels_all_head, true_labels_all_head, col_names)
    mlflowLogger.store_metric(f"test.f1", test_f1, e)
    mlflowLogger.store_metric(f"test.acc", test_acc, e)
    mlflowLogger.store_metric(f"test.LRAP", test_LRAP, e)
    mlflowLogger.store_param(f"test.report", str(test_clf_report))


    true_labels_each_head = np.array(true_labels_each_head)
    pred_labels_each_head = np.array(pred_labels_each_head)

    # print("########### Results of each head:")
    for i in range(0,nheads):
        i_head_true_labels = true_labels_each_head[:,i]
        i_head_true_labels = torch.cat([item for item in i_head_true_labels],0).to('cpu').numpy()

        i_head_pred_labels = pred_labels_each_head[:,i]
        i_head_pred_labels = torch.cat([item for item in i_head_pred_labels],0).to('cpu').numpy()

        testhead_f1, testhead_acc, testhead_LRAP = calculate_f1_acc(i_head_pred_labels, i_head_true_labels)
        mlflowLogger.store_metric(f"test.headf1.{i}", testhead_f1, e)
        mlflowLogger.store_metric(f"test.headacc.{i}", testhead_acc, e)
        mlflowLogger.store_metric(f"test.headLRAP.{i}", testhead_LRAP, e)

    mlflowLogger.finish_mlflowrun()