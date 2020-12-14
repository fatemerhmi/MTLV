from tqdm import tqdm, trange
import ast
import torch
import numpy as np
from sklearn import metrics

import mtl.utils.configuration as configuration
import mtl.utils.logger as mlflowLogger
from mtl.heads.clsHeads import *
from mtl.utils.evaluate import *

def train(train_dataloader, val_dataloader, test_dataloader, model, cfg, use_cuda, training_type, fold_i = None):
    #-------config
    training_args = cfg['training']
    cfg_optimizer = cfg['optimizer']

    #-------training args:
    if use_cuda:
        print('[  training  ] Running on GPU.')
    else:
        print('[  training  ] Running on CPU.')

    configuration.log_training_args(training_args)
    # training_type = training_args['type']
    epoch = training_args['epoch']

    #-------training type:
    if training_type == "MTL_cls": 
        cfg_loss = cfg['loss'] 
        print(f"[  training  ] The training type is: Multi-head classification.")
        if fold_i != None:
            print(f"[training] Fold {fold_i}")
            test_f1_micro, test_f1_macro, test_acc = mtl_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, cfg_loss, fold_i)
        else:
            mtl_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, cfg_loss)
    elif training_type == "singlehead_cls":
        if fold_i != None:
            print(f"[training] Fold {fold_i}")
            test_f1_micro, test_f1_macro, test_acc = singlehead_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, fold_i)
        else:
            singlehead_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer)
    
    if fold_i != None:
        return test_f1_micro, test_f1_macro, test_acc

def mtl_cls(train_dataloader, validation_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, cfg_loss, fold_i = None):
    #-------get params from mlflow
    col_names = ast.literal_eval(mlflowLogger.get_params("col_names")) 
    heads_index = ast.literal_eval(mlflowLogger.get_params("heads_index")) #if fold_i == None else ast.literal_eval(mlflowLogger.get_params(f"heads_index.Fold{fold_i}"))


    head_index_flatten = [i for head_index in heads_index for i in head_index]
    new_col_names_order = [col_names[index] for index in head_index_flatten]
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

    #-------load loss
    loss_func = configuration.setup_loss(cfg_loss)

    # #-------load Head
    # hparams={
    #     # 'labels' : b_labels,
    #     'num_labels' : col_count,
    #     'input_size' : model.embedding_size,
    #     # 'inputLayer' : outputs,
    #     'device'     : device,
    # }
    
    # #TODO: replace this with setup head
    # classification_head = HeadMultilabelCLS(hparams)

    #-------FineTune model
    for e in trange(epoch, desc="Epoch"):
        #==========Training======================
        model.train()

        tr_loss = 0 #MTL loss
        train_headloss = np.zeros(nheads) # head losses
        nb_tr_steps =  0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            optimizer.zero_grad()

            #Forward pass for MTL
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            
            #--------head loss
            train_headloss = np.sum([train_headloss, outputs.loss], axis=0)

            #------total loss and backprop
            if 'weights' in cfg_loss:
                loss = loss_func(outputs.loss, cfg_loss['weights'])
            else:
                loss = loss_func(outputs.loss)
            #--------- Backward pass
            loss.backward()
            
            optimizer.step()
            # scheduler.step()
            
            tr_loss += loss.item()
            # nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        train_headloss = train_headloss/nb_tr_steps
        index = 0
        for headloss in train_headloss:
            mlflowLogger.store_metric(f"training.headloss.{index}", headloss.item(), e) if fold_i == None else mlflowLogger.store_metric(f"training.Fold{fold_i}.headloss.{index}", headloss.item(), e)
            index +=1
        mlflowLogger.store_metric("training.loss", tr_loss/nb_tr_steps, e) if fold_i == None else mlflowLogger.store_metric(f"training.Fold{fold_i}.loss", tr_loss/nb_tr_steps, e)
        # print("Train loss: {}".format(tr_loss/nb_tr_steps))

        #===========Validation==========
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Variables to gather full output
        true_labels_each_head,pred_labels_each_head = [],[]
        true_labels_all_head,pred_labels_all_head = [],[]
        
        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                pred_label_heads = []
                true_label_heads = []
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                for i in range(0,nheads):
                    #remove -1 paddings:
                    labels = b_labels[:,i,:]
                    labels = labels[:,0:head_count[i]]

                    true_label_heads.append(labels)
                    pred = torch.sigmoid(outputs.logits[i])
                    pred_label_heads.append(pred)
                
                #------store validation loss
                if 'weights' in cfg_loss:
                    val_loss = loss_func(outputs.loss, cfg_loss['weights']).item()
                else:
                    val_loss = loss_func(outputs.loss).item()
                mlflowLogger.store_metric("validation.loss", val_loss, e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.loss", val_loss, e)
                for head_indx, val_loss_head in enumerate(outputs.loss):
                    mlflowLogger.store_metric(f"validation.loss.head{head_indx}", val_loss_head.item(), e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.loss.head{head_indx}", val_loss_head.item(), e)

                # store batch labels 
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
        val_f1_micro, val_f1_macro, val_acc, LRAP, prf = calculate_f1_acc(pred_labels_all_head, true_labels_all_head)

        mlflowLogger.store_metric("validation.f1_micro", val_f1_micro, e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.f1_micro", val_f1_micro, e)
        mlflowLogger.store_metric("validation.f1_macro", val_f1_macro, e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.f1_macro", val_f1_macro, e)
        mlflowLogger.store_metric("validation.acc", val_acc, e)           if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.acc", val_acc, e)

        #log percision, recall, f1 for each label
        if fold_i == None:
            for indx, _label in enumerate(new_col_names_order):
                #index 2 becuz it has percision, recall, f1
                mlflowLogger.store_metric(f"validation.Label.{_label}.f1", prf[2][indx], e)  #else mlflowLogger.store_metric(f"validation.Fold{fold_i}.Label.{_label}.f1", prf[2][indx], e)

        true_labels_each_head = np.array(true_labels_each_head)
        pred_labels_each_head = np.array(pred_labels_each_head)
        
        # print("Results of each head:")
        for i in range(0,nheads):
            i_head_true_labels = true_labels_each_head[:,i]
            i_head_true_labels = torch.cat([item for item in i_head_true_labels],0).to('cpu').numpy()
            
            i_head_pred_labels = pred_labels_each_head[:,i]
            i_head_pred_labels = torch.cat([item for item in i_head_pred_labels],0).to('cpu').numpy()

            val_head_f1_micro, val_head_f1_macro, val_head_acc, _ , _ = calculate_f1_acc(i_head_pred_labels, i_head_true_labels)
            mlflowLogger.store_metric(f"validation.headacc.{i}", val_head_acc, e)           if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.headacc.{i}", val_head_acc, e)
            mlflowLogger.store_metric(f"validation.headf1_micro.{i}", val_head_f1_micro, e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.headf1_micro.{i}", val_head_f1_micro, e)
            mlflowLogger.store_metric(f"validation.headf1_macro.{i}", val_head_f1_macro, e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.headf1_macro.{i}", val_head_f1_macro, e)

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
            pred_label_heads = []
            true_label_heads = []
            
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            for i in range(0,nheads):
                #remove -1 paddings:
                labels = b_labels[:,i,:]
                labels = labels[:,0:head_count[i]]

                true_label_heads.append(labels)
                pred = torch.sigmoid(outputs.logits[i])
                pred_label_heads.append(pred)

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
    

    
    test_f1_micro, test_f1_macro, test_acc, test_LRAP, test_clf_report = calculate_f1_acc_test(pred_labels_all_head, true_labels_all_head, new_col_names_order)
    mlflowLogger.store_metric(f"test.f1_micro", test_f1_micro, e)     if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.f1_micro", test_f1_micro, e)
    mlflowLogger.store_metric(f"test.f1_macro", test_f1_macro, e)     if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.f1_macro", test_f1_macro, e)
    mlflowLogger.store_metric(f"test.acc", test_acc, e)               if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.acc", test_acc, e)     
    mlflowLogger.store_metric(f"test.LRAP", test_LRAP, e)             if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.LRAP", test_LRAP, e)  
    mlflowLogger.store_artifact(test_clf_report, "cls_report", "txt") if fold_i == None else mlflowLogger.store_artifact(test_clf_report, f"cls_report.Fold{fold_i}", "txt")

    true_labels_each_head = np.array(true_labels_each_head)
    pred_labels_each_head = np.array(pred_labels_each_head)

    # print("########### Results of each head:")
    for i in range(0,nheads):
        i_head_true_labels = true_labels_each_head[:,i]
        i_head_true_labels = torch.cat([item for item in i_head_true_labels],0).to('cpu').numpy()

        i_head_pred_labels = pred_labels_each_head[:,i]
        i_head_pred_labels = torch.cat([item for item in i_head_pred_labels],0).to('cpu').numpy()

        testhead_f1_micro, testhead_f1_macro, testhead_acc, testhead_LRAP, _ = calculate_f1_acc(i_head_pred_labels, i_head_true_labels)
        mlflowLogger.store_metric(f"test.headf1_micro.{i}", testhead_f1_micro, e) if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.headf1_micro.{i}", testhead_f1_micro, e)
        mlflowLogger.store_metric(f"test.headf1_macro.{i}", testhead_f1_macro, e) if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.headf1_macro.{i}", testhead_f1_macro, e)
        mlflowLogger.store_metric(f"test.headacc.{i}", testhead_acc, e) if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.headacc.{i}", testhead_acc, e)
        mlflowLogger.store_metric(f"test.headLRAP.{i}", testhead_LRAP, e) if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.headLRAP.{i}", testhead_LRAP, e)

    if fold_i == None: 
        mlflowLogger.finish_mlflowrun()
    if fold_i != None: 
        return test_f1_micro, test_f1_macro, test_acc

def singlehead_cls(train_dataloader, validation_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, fold_i = None):
    #-------get params from mlflow
    col_names = ast.literal_eval(mlflowLogger.get_params("col_names"))
    # print("training", col_names)
    col_count = len(col_names)
    # print("training", col_count)

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
        #==========Training======================
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0 #running loss
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
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)  

            # loss, _ = classification_head.run(outputs, b_labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        mlflowLogger.store_metric("training.loss", tr_loss/nb_tr_steps, e)  if fold_i == None else mlflowLogger.store_metric(f"training.Fold{fold_i}.loss", tr_loss/nb_tr_steps, e)

        #===========Validation==========
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Variables to gather full output
        true_labels_signlehead, pred_labels_singlehead = [],[]
        
        # Predict
        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)  

                #----store validation loss
                val_loss = outputs.loss.item()
                mlflowLogger.store_metric("validation.loss", val_loss, e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.loss", val_loss, e)

                pred_label_b = torch.sigmoid(outputs.logits)

                true_labels_signlehead.append(b_labels)
                pred_labels_singlehead.append(pred_label_b)

        true_labels_signlehead = np.concatenate([item.to('cpu').numpy() for item in true_labels_signlehead])
        pred_labels_singlehead = np.concatenate([item.to('cpu').numpy() for item in pred_labels_singlehead])

        val_f1_micro, val_f1_macro, val_acc, _ , prf = calculate_f1_acc(pred_labels_singlehead, true_labels_signlehead)

        mlflowLogger.store_metric("validation.f1_micro", val_f1_micro, e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.f1_micro", val_f1_micro, e)
        mlflowLogger.store_metric("validation.f1_macro", val_f1_macro, e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.f1_macro", val_f1_macro, e)
        mlflowLogger.store_metric("validation.acc", val_acc, e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.acc", val_acc, e)

        #log percision, recall, f1 for each label
        if fold_i == None:
            for indx, _label in enumerate(col_names): 
                #index 2 becuz it has 0:percision, 1:recall, 2:f1
                mlflowLogger.store_metric(f"validation.Label.{_label}.f1", prf[2][indx], e)  #else mlflowLogger.store_metric(f"validation.Fold{fold_i}.Label.{_label}.f1", prf[2][indx], e)

    #============test=============
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    true_labels_signlehead, pred_labels_singlehead = [],[]
    
    # Predict
    for i, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)  
            pred_label_b = torch.sigmoid(outputs.logits)
            true_labels_signlehead.append(b_labels)
            pred_labels_singlehead.append(pred_label_b)
    
    true_labels_signlehead = np.concatenate([item.to('cpu').numpy() for item in true_labels_signlehead])
    pred_labels_singlehead = np.concatenate([item.to('cpu').numpy() for item in pred_labels_singlehead])

    test_f1_micro, test_f1_macro, test_acc, test_LRAP, test_clf_report = calculate_f1_acc_test(pred_labels_singlehead, true_labels_signlehead, col_names)
    mlflowLogger.store_metric(f"test.f1_micro", test_f1_micro)        if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.f1_micro", test_f1_micro)
    mlflowLogger.store_metric(f"test.f1_macro", test_f1_macro)        if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.f1_macro", test_f1_macro)
    mlflowLogger.store_metric(f"test.acc", test_acc)                  if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.acc", test_acc) 
    mlflowLogger.store_metric(f"test.LRAP", test_LRAP)                if fold_i == None else mlflowLogger.store_metric(f"test.Fold{fold_i}.LRAP", test_LRAP)
    mlflowLogger.store_artifact(test_clf_report, "cls_report", "txt") if fold_i == None else mlflowLogger.store_artifact(test_clf_report, f"cls_report.Fold{fold_i}.", "txt")

    if fold_i !=None: 
        return test_f1_micro, test_f1_macro, test_acc
    if fold_i == None: 
        mlflowLogger.finish_mlflowrun()


