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
    training_type_experiment = training_args['type']
    # epoch = training_args['epoch']

    #-------training type:
    if training_type == "MTL_cls": 
        if training_type_experiment == "ttest":
            epoch = training_args['epoch_mtl']
        else: 
            epoch = training_args['epoch']
        cfg_loss = cfg['loss'] 
        print(f"[  training  ] The training type is: Multi-head classification.")
        if fold_i != None:
            print(f"[training] Fold {fold_i}")
            test_f1_micro, test_f1_macro, test_subset_accuracy, test_hamming_score_ = mtl_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, cfg_loss, fold_i)
        else:
            mtl_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, cfg_loss)

    elif training_type == "singlehead_cls":
        if training_type_experiment == "ttest":
            epoch = training_args['epoch_s']
        else: 
            epoch = training_args['epoch']

        if fold_i != None:
            print(f"[training] Fold {fold_i}")
            test_f1_micro, test_f1_macro, test_subset_accuracy, test_hamming_score_ = singlehead_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, fold_i)
        else:
            singlehead_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer)
    
    if fold_i != None:
        return test_f1_micro, test_f1_macro, test_subset_accuracy, test_hamming_score_

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

    #-------load loss
    loss_func = configuration.setup_loss(cfg_loss)

    #-------FineTune model
    for e in trange(epoch, desc="Epoch"):
        #============================Training======================================
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
            mlflowLogger.store_metric(f"mtl.training.headloss.{index}", headloss.item(), e) if fold_i == None else mlflowLogger.store_metric(f"mtl.training.Fold{fold_i}.headloss.{index}", headloss.item(), e)
            index +=1
        mlflowLogger.store_metric("mtl.training.loss", tr_loss/nb_tr_steps, e) if fold_i == None else mlflowLogger.store_metric(f"mtl.training.Fold{fold_i}.loss", tr_loss/nb_tr_steps, e)
        # print("Train loss: {}".format(tr_loss/nb_tr_steps))

        #==============================Validation======================================
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
                # if 'weights' in cfg_loss:
                #     val_loss = loss_func(outputs.loss, cfg_loss['weights']).item()
                # else:
                #     val_loss = loss_func(outputs.loss).item()
                # mlflowLogger.store_metric("validation.loss", val_loss, e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.loss", val_loss, e)
                # for head_indx, val_loss_head in enumerate(outputs.loss):
                #     mlflowLogger.store_metric(f"validation.loss.head{head_indx}", val_loss_head.item(), e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.loss.head{head_indx}", val_loss_head.item(), e)

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

        #-------------------------calculate and storing VALIDATION result for ALL heads----------------------
        true_labels_all_head = np.concatenate([item for item in true_labels_all_head])
        pred_labels_all_head = np.concatenate([item for item in pred_labels_all_head])

        val_head_f1_micro, val_head_f1_macro, val_head_hamming_loss_, val_head_hamming_score_, val_head_subset_accuracy, prf = calculate_scores(pred_labels_all_head, true_labels_all_head)

        store_results_to_mlflow("mtl.validation", fold_i, e , val_head_f1_micro, val_head_f1_macro, val_head_hamming_loss_, val_head_hamming_score_, val_head_subset_accuracy)

        #log percision, recall, f1 for each label
        if fold_i == None:
            for indx, _label in enumerate(new_col_names_order):
                #index 2 becuz it has percision, recall, f1
                mlflowLogger.store_metric(f"mtl.validation.Label.{_label}.f1", prf[2][indx], e)  #else mlflowLogger.store_metric(f"validation.Fold{fold_i}.Label.{_label}.f1", prf[2][indx], e)

        #-------------------------calculate and storing VALIDATION result for EACH head----------------------
        true_labels_each_head = np.array(true_labels_each_head)
        pred_labels_each_head = np.array(pred_labels_each_head)
        for i in range(0,nheads):
            i_head_true_labels = true_labels_each_head[:,i]
            i_head_true_labels = torch.cat([item for item in i_head_true_labels],0).to('cpu').numpy()
            
            i_head_pred_labels = pred_labels_each_head[:,i]
            i_head_pred_labels = torch.cat([item for item in i_head_pred_labels],0).to('cpu').numpy()

            val_head_f1_micro, val_head_f1_macro, val_head_hamming_loss_, val_head_hamming_score_, val_head_subset_accuracy, _ = calculate_scores(i_head_pred_labels, i_head_true_labels)
            store_results_to_mlflow(f"mtl.validation.head{i}", fold_i, e , val_head_f1_micro, val_head_f1_macro, val_head_hamming_loss_, val_head_hamming_score_, val_head_subset_accuracy)


    #===========================test============================
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

    #-------------------------calculate and storing TEST result for ALL heads----------------------
    true_labels_all_head = np.concatenate([item for item in true_labels_all_head])
    pred_labels_all_head = np.concatenate([item for item in pred_labels_all_head])
    
    test_f1_score_micro, test_f1_score_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy, test_clf_report= calculate_f1_acc_test(pred_labels_all_head, true_labels_all_head, new_col_names_order)
    store_results_to_mlflow(f"mtl.test", fold_i, e , test_f1_score_micro, test_f1_score_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy, test_clf_report)
    

    true_labels_each_head = np.array(true_labels_each_head)
    pred_labels_each_head = np.array(pred_labels_each_head)

    #-------------------------calculate and storing TEST result for EACH head----------------------
    for i in range(0,nheads):
        i_head_true_labels = true_labels_each_head[:,i]
        i_head_true_labels = torch.cat([item for item in i_head_true_labels],0).to('cpu').numpy()

        i_head_pred_labels = pred_labels_each_head[:,i]
        i_head_pred_labels = torch.cat([item for item in i_head_pred_labels],0).to('cpu').numpy()

        test_head_f1_micro, test_head_f1_macro, test_head_hamming_loss_, test_head_hamming_score_, test_head_subset_accuracy, _ = calculate_scores(i_head_pred_labels, i_head_true_labels)
        store_results_to_mlflow(f"mtl.test.head{i}", fold_i, e , test_head_f1_micro, test_head_f1_macro, test_head_hamming_loss_, test_head_hamming_score_, test_head_subset_accuracy)

    #-------------------------
    if fold_i == None: 
        mlflowLogger.finish_mlflowrun()
    if fold_i != None: 
        return test_f1_score_micro, test_f1_score_macro, test_hamming_score_, test_subset_accuracy

def singlehead_cls(train_dataloader, validation_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, fold_i = None):

    col_names = ast.literal_eval(mlflowLogger.get_params("col_names"))
    col_count = len(col_names)

    #-------load model
    if use_cuda:
        device = "cuda"
        model.cuda()
    else:
        device = "cpu"

    #-------load optimizer
    optimizer = configuration.setup_optimizer(cfg_optimizer, model.parameters())

    #-------FineTune model
    for e in trange(epoch, desc="Epoch"):
        #==========Training======================
        model.train()
        # Tracking variables
        tr_loss = 0 #running loss
        nb_tr_steps = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)  
            optimizer.zero_grad()
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_steps += 1

        mlflowLogger.store_metric("stl.training.loss", tr_loss/nb_tr_steps, e)  if fold_i == None else mlflowLogger.store_metric(f"stl.training.Fold{fold_i}.loss", tr_loss/nb_tr_steps, e)

        #===========Validation==========
        model.eval()
        true_labels_signlehead, pred_labels_singlehead = [],[]
        with torch.no_grad():
            for i, batch in enumerate(validation_dataloader):
                batch = tuple(t.to(device, dtype = torch.long) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  
                # val_loss = outputs.loss.item()
                # mlflowLogger.store_metric("validation.loss", val_loss, e) if fold_i == None else mlflowLogger.store_metric(f"validation.Fold{fold_i}.loss", val_loss, e)

                true_labels_signlehead.extend(b_labels.cpu().detach().numpy().tolist())
                pred_labels_singlehead.extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())

        val_f1_micro, val_f1_macro, val_hamming_loss_, val_hamming_score_, val_subset_accuracy, prf = calculate_scores(pred_labels_singlehead, true_labels_signlehead)
        store_results_to_mlflow("stl.validation", fold_i, e , val_f1_micro, val_f1_macro, val_hamming_loss_, val_hamming_score_, val_subset_accuracy)

        #log percision, recall, f1 for each label
        # if fold_i == None:
        #     for indx, _label in enumerate(col_names): 
        #         #index 2 becuz it has 0:percision, 1:recall, 2:f1
        #         mlflowLogger.store_metric(f"validation.Label.{_label}.f1", prf[2][indx], e)  #else mlflowLogger.store_metric(f"validation.Fold{fold_i}.Label.{_label}.f1", prf[2][indx], e)

    #============test=============
    model.eval()

    true_labels_signlehead, pred_labels_singlehead = [],[]
    
    # Predict
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  

            true_labels_signlehead.extend(b_labels.cpu().detach().numpy().tolist())
            pred_labels_singlehead.extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())
    
    test_f1_micro, test_f1_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy, test_clf_report = calculate_f1_acc_test(pred_labels_singlehead, true_labels_signlehead, col_names)
    store_results_to_mlflow("stl.test", fold_i, e , test_f1_micro, test_f1_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy)
    mlflowLogger.store_artifact(test_clf_report, "stl.test.cls_report", "txt") if fold_i == None else mlflowLogger.store_artifact(test_clf_report, f"stl.test.cls_report.Fold{fold_i}.", "txt")

    if fold_i !=None: 
        return test_f1_micro, test_f1_macro, test_hamming_score_, test_subset_accuracy
    if fold_i == None: 
        mlflowLogger.finish_mlflowrun()

def store_results_to_mlflow(prefix, fold_i, e , f1_micro, f1_macro, hamming_loss_, hamming_score_, subset_accuracy, clf_report = None):
    if fold_i == None:
        mlflowLogger.store_metric(f"{prefix}.f1_micro", f1_micro, e)
        mlflowLogger.store_metric(f"{prefix}.f1_macro", f1_macro, e)
        mlflowLogger.store_metric(f"{prefix}.Hamming_score", hamming_score_, e)
        # mlflowLogger.store_metric(f"{prefix}.Hamming_loss", hamming_loss_, e)
        mlflowLogger.store_metric(f"{prefix}.subset_accuracy", subset_accuracy, e)
        if clf_report != None: mlflowLogger.store_artifact(clf_report, "cls_report", "txt")
    else:
        mlflowLogger.store_metric(f"{prefix}.Fold{fold_i}.f1_micro", f1_micro, e)
        mlflowLogger.store_metric(f"{prefix}.Fold{fold_i}.f1_macro", f1_macro, e)
        mlflowLogger.store_metric(f"{prefix}.Fold{fold_i}.Hamming_score", hamming_score_, e)
        # mlflowLogger.store_metric(f"{prefix}.Fold{fold_i}.Hamming_loss", hamming_loss_, e)
        mlflowLogger.store_metric(f"{prefix}.Fold{fold_i}.subset_accuracy", subset_accuracy, e)
        if clf_report != None: mlflowLogger.store_artifact(clf_report, f"cls_report.Fold{fold_i}", "txt")

