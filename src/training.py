from tqdm import tqdm, trange
import ast
import torch
import numpy as np
from sklearn import metrics

import mtl.utils.configuration as configuration
import mtl.utils.logger as mlflowLogger
from mtl.heads.clsHeads import *
from mtl.utils.evaluate import *
from mtl.utils.training import mtl_validation_test

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
            test_f1_micro, test_f1_macro, test_subset_accuracy, test_hamming_score_ = mtl_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, cfg_loss, fold_i, training_type_experiment)
        else:
            mtl_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, cfg_loss)

    elif training_type == "singlehead_cls":
        if training_type_experiment == "ttest":
            epoch = training_args['epoch_s']
        else: 
            epoch = training_args['epoch']

        if fold_i != None:
            print(f"[training] Fold {fold_i}")
            test_f1_micro, test_f1_macro, test_subset_accuracy, test_hamming_score_ = ml_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer, fold_i, training_type_experiment)
        else:
            ml_cls(train_dataloader, val_dataloader, test_dataloader, model, epoch, use_cuda, cfg_optimizer)
    
    if fold_i != None:
        return test_f1_micro, test_f1_macro, test_subset_accuracy, test_hamming_score_

def GMHL(train_dataloader, validation_dataloader, test_dataloader, cfg, use_cuda, fold_i = None):
    """
    GMHL: Group multi head learning

    A function to make a multi-head architecture (an example of multi-task learning).
    To learn all the tasks in same network but with groupings of tasks in seperate heads. 
    """
    #-------config
    training_args = cfg['training']
    training_type = training_args['type']
    cfg_optimizer = cfg['optimizer']
    model_cfg = cfg['model']
    epoch = training_args['epoch']
    cfg_loss = cfg['loss'] 

    #-------get params from mlflow
    col_names = ast.literal_eval(mlflowLogger.get_params("col_names")) 
    heads_index = ast.literal_eval(mlflowLogger.get_params("heads_index")) 
    # num_labels = [len(labels) for labels in heads_index]
    
    #-------get mlflow prefix for storing variables === TODO: figure out ttest
    training_type_experiment = cfg['training']['type']
    prefix_logger = ""
    if training_type_experiment == 'ttest':
        prefix_logger = "gmhl."

    head_index_flatten = [i for head_index in heads_index for i in head_index]
    new_col_names_order = [col_names[index] for index in head_index_flatten]
    #-------load heads
    head_count = [len(group) for group in heads_index]
    nheads = len(head_count)

    #-------load model
    if use_cuda:
        device = "cuda"
    else:
        device = "cpu"
    model = configuration.setup_model(cfg['model'])(head_count, training_type, device)
    if use_cuda:
        model.cuda()


    #-------load optimizer
    optimizer = configuration.setup_optimizer(cfg_optimizer, model.parameters())

    #-------load loss
    loss_func = configuration.setup_loss(cfg_loss)

    #-------FineTune model
    for e in trange(epoch, desc="Epoch"):
        #============================Training======================================
        model.train()

        tr_loss = 0 #MTL loss
        train_headloss = torch.zeros(nheads) # head losses
        nb_tr_steps =  0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            optimizer.zero_grad()

            #Forward pass for MTL
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            
            #--------head loss
            # train_headloss = np.sum([train_headloss, outputs.loss], axis=0)
            train_headloss = train_headloss.add(outputs.loss)

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
            mlflowLogger.store_metric(f"{prefix_logger}training.headloss.{index}", headloss.item(), e) if fold_i == None else mlflowLogger.store_metric(f"{prefix_logger}training.Fold{fold_i}.headloss.{index}", headloss.item(), e)
            index +=1
        mlflowLogger.store_metric(f"{prefix_logger}training.loss", tr_loss/nb_tr_steps, e) if fold_i == None else mlflowLogger.store_metric(f"{prefix_logger}training.Fold{fold_i}.loss", tr_loss/nb_tr_steps, e)
        # print("Train loss: {}".format(tr_loss/nb_tr_steps))

        #==============================Validation======================================
        model.eval()
        pred_labels_all_head, true_labels_all_head, true_labels_each_head, pred_labels_each_head = mtl_validation_test(validation_dataloader, head_count, device, nheads, model)

        #-------------------------calculate and storing VALIDATION result for ALL heads----------------------
        val_head_f1_micro, val_head_f1_macro, val_head_hamming_loss_, val_head_hamming_score_, val_head_subset_accuracy, prf = calculate_scores(pred_labels_all_head, true_labels_all_head)

        store_results_to_mlflow(f"{prefix_logger}validation", fold_i, e , val_head_f1_micro, val_head_f1_macro, val_head_hamming_loss_, val_head_hamming_score_, val_head_subset_accuracy)

        #log percision, recall, f1 for each label
        if fold_i == None:
            for indx, _label in enumerate(new_col_names_order):
                #index 2 becuz it has percision, recall, f1
                mlflowLogger.store_metric(f"{prefix_logger}validation.Label.{_label}.f1", prf[2][indx], e)  #else mlflowLogger.store_metric(f"validation.Fold{fold_i}.Label.{_label}.f1", prf[2][indx], e)

        #-------------------------calculate and storing VALIDATION result for EACH head----------------------
        true_labels_each_head = np.array(true_labels_each_head)
        pred_labels_each_head = np.array(pred_labels_each_head)
        for i in range(0,nheads):
            i_head_true_labels = true_labels_each_head[:,i]
            i_head_true_labels = np.concatenate([item for item in i_head_true_labels],0)
            
            i_head_pred_labels = pred_labels_each_head[:,i]
            i_head_pred_labels = np.concatenate([item for item in i_head_pred_labels],0)

            val_head_f1_micro, val_head_f1_macro, val_head_hamming_loss_, val_head_hamming_score_, val_head_subset_accuracy, _ = calculate_scores(i_head_pred_labels, i_head_true_labels)
            store_results_to_mlflow(f"{prefix_logger}validation.head{i}", fold_i, e , val_head_f1_micro, val_head_f1_macro, val_head_hamming_loss_, val_head_hamming_score_, val_head_subset_accuracy)

    #===========================test============================
    model.eval()

    pred_labels_all_head, true_labels_all_head, true_labels_each_head, pred_labels_each_head = mtl_validation_test(test_dataloader, head_count, device, nheads, model)
    
    #-------------------------calculate and storing TEST result for ALL heads----------------------    
    test_f1_score_micro, test_f1_score_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy, test_clf_report= calculate_scores_test(pred_labels_all_head, true_labels_all_head, new_col_names_order)
    store_results_to_mlflow(f"{prefix_logger}test", fold_i, e , test_f1_score_micro, test_f1_score_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy, test_clf_report)
    
    #-------------------------calculate and storing TEST result for EACH head----------------------

    true_labels_each_head = np.array(true_labels_each_head)
    pred_labels_each_head = np.array(pred_labels_each_head)
    for i in range(0,nheads):
        i_head_true_labels = true_labels_each_head[:,i]
        i_head_true_labels = np.concatenate([item for item in i_head_true_labels],0)
        
        i_head_pred_labels = pred_labels_each_head[:,i]
        i_head_pred_labels = np.concatenate([item for item in i_head_pred_labels],0)

        test_head_f1_micro, test_head_f1_macro, test_head_hamming_loss_, test_head_hamming_score_, test_head_subset_accuracy, _ = calculate_scores(i_head_pred_labels, i_head_true_labels)
        store_results_to_mlflow(f"{prefix_logger}test.head{i}", fold_i, e , test_head_f1_micro, test_head_f1_macro, test_head_hamming_loss_, test_head_hamming_score_, test_head_subset_accuracy)

    #-------------------------
    if fold_i == None: 
        mlflowLogger.finish_mlflowrun()
    if fold_i != None: 
        return test_f1_score_micro, test_f1_score_macro, test_hamming_score_, test_subset_accuracy

def MTL(train_dataloader, validation_dataloader, test_dataloader, num_labels, cfg, use_cuda, fold_i = None, training_type_experiment = None):
    """
    MTL: Multi-task leanring
    Multi-label classification or Multi-task learning classification
    """
    #-------config
    training_args = cfg['training']
    cfg_optimizer = cfg['optimizer']
    model_cfg = cfg['model']
    epoch = training_args['epoch']

    #-------get mlflow prefix for storing variables
    prefix_logger = ""
    if training_type_experiment == 'ttest':
        prefix_logger = "mtl."

    #-------training args:
    training_type = "MTL_cls"

    #------- model
    model = configuration.setup_model(model_cfg)(num_labels, training_type) 
    if use_cuda:
        device = "cuda"
        model.cuda()
    else:
        device = "cpu"
    
    configuration.log_training_args(training_args)

    col_names = ast.literal_eval(mlflowLogger.get_params("col_names"))
    col_count = len(col_names)

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
            # batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device, dtype = torch.long)
            b_input_mask = b_input_mask.to(device, dtype = torch.long)
            b_labels = b_labels.to(device, dtype = torch.float)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)  
            optimizer.zero_grad()
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_steps += 1

        mlflowLogger.store_metric(f"{prefix_logger}training.loss", tr_loss/nb_tr_steps, e)  if fold_i == None else mlflowLogger.store_metric(f"{prefix_logger}training.Fold{fold_i}.loss", tr_loss/nb_tr_steps, e)

        #===========Validation==========
        model.eval()
        true_labels_signlehead, pred_labels_singlehead = [],[]
        with torch.no_grad():
            for i, batch in enumerate(validation_dataloader):
                # batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                b_input_ids = b_input_ids.to(device, dtype = torch.long)
                b_input_mask = b_input_mask.to(device, dtype = torch.long)
                b_labels = b_labels.to(device, dtype = torch.float)

                #multi-label cls
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  

                true_labels_signlehead.extend(b_labels.cpu().detach().numpy().tolist())
                pred_labels_singlehead.extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())

        val_f1_micro, val_f1_macro, val_hamming_loss_, val_hamming_score_, val_subset_accuracy, prf = calculate_scores(pred_labels_singlehead, true_labels_signlehead)
        store_results_to_mlflow(f"{prefix_logger}validation", fold_i, e , val_f1_micro, val_f1_macro, val_hamming_loss_, val_hamming_score_, val_subset_accuracy)

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
            # batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device, dtype = torch.long)
            b_input_mask = b_input_mask.to(device, dtype = torch.long)
            b_labels = b_labels.to(device, dtype = torch.float)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  

            true_labels_signlehead.extend(b_labels.cpu().detach().numpy().tolist())
            pred_labels_singlehead.extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())
    
    test_f1_micro, test_f1_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy, test_clf_report = calculate_scores_test(pred_labels_singlehead, true_labels_signlehead, col_names)
    store_results_to_mlflow(f"{prefix_logger}test", fold_i, e , test_f1_micro, test_f1_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy, test_clf_report)

    if fold_i !=None: 
        return test_f1_micro, test_f1_macro, test_hamming_score_, test_subset_accuracy
    if fold_i == None: 
        mlflowLogger.finish_mlflowrun()

def STL(train_dataloader, validation_dataloader, test_dataloader, num_labels, cfg, use_cuda, fold_i = None, training_type_experiment = None):
    """
    STL: Single Task Learning Classificaiton
    A binary classificaiton per label
    """
    #-------training args:
    training_args = cfg['training']
    training_type = cfg['training']['type']
    
    configuration.log_training_args(training_args)

    col_names = ast.literal_eval(mlflowLogger.get_params("col_names"))
    col_count = len(col_names)

    #-------get mlflow prefix for storing variables
    prefix_logger = ""
    if training_type_experiment == 'ttest':
        prefix_logger = "stl."

    #-------Binary classificaiton per label
    true_label_all, preds_label_all = [],[]
    for index_label, single_label_name in enumerate(col_names):
        true_label, preds_label = BC(train_dataloader, validation_dataloader, test_dataloader, index_label, single_label_name, training_type, cfg, use_cuda, fold_i = fold_i, prefix_logger = prefix_logger)

        #storying test target, pred for each label
        true_label_all.append(true_label)
        preds_label_all.append(preds_label)

        # if index_label == 2:
        #     break

    #store labels as a multi-label problem
    #concatiante 
    true_label_all = np.concatenate(true_label_all, axis=1)
    preds_label_all = np.concatenate(preds_label_all, axis=1)
    test_f1_micro, test_f1_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy, test_clf_report = calculate_scores_test(preds_label_all, true_label_all, col_names)
    store_results_to_mlflow(f"{prefix_logger}test", fold_i, 1 , test_f1_micro, test_f1_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy, clf_report = test_clf_report)
    # mlflowLogger.store_artifact(test_clf_report, f"{prefix_logger}test.cls_report", "txt") if fold_i == None else mlflowLogger.store_artifact(test_clf_report, f"{prefix_logger}test.cls_report.Fold{fold_i}.", "txt")

    if fold_i !=None: 
        return test_f1_micro, test_f1_macro, test_hamming_score_, test_subset_accuracy
    if fold_i == None: 
        mlflowLogger.finish_mlflowrun()

def GMTL(train_dataloader, validation_dataloader, test_dataloader, cfg , use_cuda, fold_i= None):
    """
    GMTL: Grouping Multi-task learning
    A set of models to to perform multi-task learning 
    Each model learn a set of labels(tasks) that is generated form the clustering (few seperate models)
    """
    #-------training args:
    training_args = cfg['training']
    training_type = "GMTL_cls"
    
    configuration.log_training_args(training_args)

     #-------get params from mlflow
    col_names = ast.literal_eval(mlflowLogger.get_params("col_names")) 
    all_heads_indexes = ast.literal_eval(mlflowLogger.get_params("heads_index")) #if fold_i == None else ast.literal_eval(mlflowLogger.get_params(f"heads_index.Fold{fold_i}"))
    col_count = len(col_names)

    head_indexes_flatten = [i for head_index in all_heads_indexes for i in head_index]
    new_col_names_order = [col_names[index] for index in head_indexes_flatten]

    #-------get mlflow prefix for storing variables
    prefix_logger = ""
    if training_type == 'ttest':
        prefix_logger = "gmtl."

    #-------multi-label cls per group 
    true_label_all, preds_label_all = [],[]
    for index_head, per_head_indexes in enumerate(all_heads_indexes):
        true_label, preds_label = MLC(train_dataloader, validation_dataloader, test_dataloader, index_head, per_head_indexes, training_type, cfg, use_cuda, fold_i = fold_i, prefix_logger = prefix_logger)

        #storying test target, pred for each label
        true_label_all.append(true_label)
        preds_label_all.append(preds_label)

    #store labels as a multi-label problem
    #concatiante 
    true_label_all = np.concatenate(true_label_all, axis=1)
    preds_label_all = np.concatenate(preds_label_all, axis=1)
    test_f1_micro, test_f1_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy, test_clf_report = calculate_scores_test(preds_label_all, true_label_all, new_col_names_order)
    store_results_to_mlflow(f"{prefix_logger}test", fold_i, 1 , test_f1_micro, test_f1_macro, test_hamming_loss_, test_hamming_score_, test_subset_accuracy, clf_report = test_clf_report)

    if fold_i !=None: 
        return test_f1_micro, test_f1_macro, test_hamming_score_, test_subset_accuracy
    if fold_i == None: 
        mlflowLogger.finish_mlflowrun()

def MLC(train_dataloader, validation_dataloader, test_dataloader, index_head, per_head_indexes, training_type, cfg, use_cuda,fold_i = None, prefix_logger = None):
    """
    MLC: multi-label classification
    A helper funciton for other functions to call when they are doing a multi-label classification in their settings
    """
   #-------config
    training_args = cfg['training']
    cfg_optimizer = cfg['optimizer']
    model_cfg = cfg['model']
    epoch = training_args['epoch']

    #-------load model
    num_labels = len(per_head_indexes)
    model = configuration.setup_model(model_cfg)(num_labels, training_type) #fix this for binary cls
    if use_cuda:
        device = "cuda"
        model.cuda()
    else:
        device = "cpu"

    #-------load optimizer
    optimizer = configuration.setup_optimizer(cfg_optimizer, model.parameters())

    #-------FineTune model
    for e in trange(epoch, desc="Epoch"):
        #============================Training======================================
        model.train()
        tr_loss ,nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device, dtype = torch.long)
            b_input_mask = b_input_mask.to(device, dtype = torch.long)
            b_labels = b_labels.to(device, dtype = torch.float)

            #get label of each label for binary classification
            b_labels_head = b_labels[:,index_head]
            #remove -1
            b_labels_head = b_labels_head[:,0:num_labels]
            #binary cls
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels_head)  
            
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_steps += 1

        mlflowLogger.store_metric(f"{prefix_logger}training.headloss{index_head}", tr_loss/nb_tr_steps, e) if fold_i == None else mlflowLogger.store_metric(f"{prefix_logger}training.Fold{fold_i}.headloss{index_head}", tr_loss/nb_tr_steps, e)
        #==============================Validation======================================
        model.eval()
        true_label, preds_label = [],[]
        with torch.no_grad():
            for i, batch in enumerate(validation_dataloader):
                # batch = tuple(t.to(device, dtype = torch.long) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                b_input_ids = b_input_ids.to(device, dtype = torch.long)
                b_input_mask = b_input_mask.to(device, dtype = torch.long)
                b_labels = b_labels.to(device, dtype = torch.float)

                # get label of each label for binary classification
                b_labels_head = b_labels[:,index_head]
                #remove -1
                b_labels_head = b_labels_head[:,0:num_labels]

                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  
                true_label.extend(b_labels_head.cpu().detach().numpy().tolist())
                preds_label.extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())

        val_head_f1_micro, val_head_f1_macro, val_head_hamming_loss_, val_head_hamming_score_, val_head_subset_accuracy, prf = calculate_scores(preds_label, true_label)
        store_results_to_mlflow(f"{prefix_logger}validation.head{index_head}", fold_i, e , val_head_f1_micro, val_head_f1_macro, val_head_hamming_loss_, val_head_hamming_score_, val_head_subset_accuracy)

        #log percision, recall, f1 for each label
        # if fold_i == None:
        #     for indx, _label in enumerate(new_col_names_order):
        #         #index 2 becuz it has percision, recall, f1
        #         mlflowLogger.store_metric(f"{prefix_logger}validation.Label.{_label}.f1", prf[2][indx], e)  #else mlflowLogger.store_metric(f"validation.Fold{fold_i}.Label.{_label}.f1", prf[2][indx], e)

    #============test=============
    model.eval()
    true_labels, pred_labels = [],[]
    
    # Predict
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device, dtype = torch.long)
            b_input_mask = b_input_mask.to(device, dtype = torch.long)
            b_labels = b_labels.to(device, dtype = torch.float)

            # get label of each label for binary classification
            b_labels_head = b_labels[:,index_head]
            #remove -1
            b_labels_head = b_labels_head[:,0:num_labels]
            
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  

            true_labels.extend(b_labels_head.cpu().detach().numpy().tolist())
            pred_labels.extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())
    
    test_head_f1_micro, test_head_f1_macro, test_head_hamming_loss_, test_head_hamming_score_, test_head_subset_accuracy, _ = calculate_scores_test(true_labels, true_labels)
    store_results_to_mlflow(f"{prefix_logger}test.head{index_head}", fold_i, e , test_head_f1_micro, test_head_f1_macro, test_head_hamming_loss_, test_head_hamming_score_, test_head_subset_accuracy)

    return true_labels, pred_labels

def BC(train_dataloader, validation_dataloader, test_dataloader, index_label, single_label_name, training_type, cfg, use_cuda,fold_i = None, prefix_logger = None):
    """
    BC: Binary classification
    """
    #-------config
    training_args = cfg['training']
    cfg_optimizer = cfg['optimizer']
    model_cfg = cfg['model']
    epoch = training_args['epoch']

    #------- model
    model = configuration.setup_model(model_cfg)(1, training_type) #fix this for binary cls
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
        tr_loss ,nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device, dtype = torch.long)
            b_input_mask = b_input_mask.to(device, dtype = torch.long)
            b_labels = b_labels.to(device, dtype = torch.float)

            #get label of each label for binary classification
            b_labels_single = b_labels[:,index_label].view(-1,1)

            #binary cls
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels_single)  
            
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_steps += 1

        mlflowLogger.store_metric(f"{prefix_logger}{single_label_name}.training.loss", tr_loss/nb_tr_steps, e)  if fold_i == None else mlflowLogger.store_metric(f"{prefix_logger}training.Fold{fold_i}.loss", tr_loss/nb_tr_steps, e)

        #===========Validation==========
        model.eval()
        true_label, preds_label = [],[]
        with torch.no_grad():
            for i, batch in enumerate(validation_dataloader):
                # batch = tuple(t.to(device, dtype = torch.long) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                b_input_ids = b_input_ids.to(device, dtype = torch.long)
                b_input_mask = b_input_mask.to(device, dtype = torch.long)
                b_labels = b_labels.to(device, dtype = torch.float)

                # get label of each label for binary classification
                b_labels_single = b_labels[:,index_label].view(-1,1)

                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  
                true_label.extend(b_labels_single.cpu().detach().numpy().tolist())
                preds_label.extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())

        val_f1_micro, val_f1_macro = calculate_scores_binarycls(preds_label, true_label)
        store_results_to_mlflow_binarycls(f"{prefix_logger}{single_label_name}.validation", fold_i, e , val_f1_micro, val_f1_macro)

    #============test=============
    model.eval()
    true_label, preds_label = [],[]
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device, dtype = torch.long)
            b_input_mask = b_input_mask.to(device, dtype = torch.long)
            b_labels = b_labels.to(device, dtype = torch.float)


            # get label of each label for binary classification
            b_labels_single = b_labels[:,index_label].view(-1,1)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  

            true_label.extend(b_labels_single.cpu().detach().numpy().tolist())
            preds_label.extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())
    
    test_f1_micro, test_f1_macro = calculate_scores_binarycls(preds_label, true_label)
    store_results_to_mlflow_binarycls(f"{prefix_logger}{single_label_name}.test", fold_i, e , test_f1_micro, test_f1_macro)

    return true_label, preds_label

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

def store_results_to_mlflow_binarycls(prefix, fold_i, e , f1_micro, f1_macro):
    if fold_i == None:
        mlflowLogger.store_metric(f"{prefix}.f1_micro", f1_micro, e)
        mlflowLogger.store_metric(f"{prefix}.f1_macro", f1_macro, e)
    else:
        mlflowLogger.store_metric(f"{prefix}.Fold{fold_i}.f1_micro", f1_micro, e)
        mlflowLogger.store_metric(f"{prefix}.Fold{fold_i}.f1_macro", f1_macro, e)


