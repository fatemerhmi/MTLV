


def train():
    print("training function")



def multihead_cls(data_path, split_path, PreTrainedModel, epochs, batch_size, 
                  max_length ,ModelTokenizer, tokenizer_name, model_name, 
                  use_data_loader, heads_index, col_names):

    head_count = [len(group) for group in heads_index]
    nheads = len(head_count)
    #-----------load model----------------
    # Load model, the pretrained model will include a single linear classification layer on top for classification. 
    model = PreTrainedModel.from_pretrained(model_name)
    model.cuda()
    optimizer = AdamW(model.parameters(),lr=2e-5)  # Default optimization
    
    #---------FineTune model-----------
    # device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # n_gpu = torch.cuda.device_count()
    
    # Store our loss and accuracy for plotting
    train_loss_set = []

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):

        #-------Training-------
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
            for head in head_losses:
                loss += head.loss 
            
            # Backward pass
            loss.backward()
            
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        # ---------Validation--------

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
        
        print("Results of all heads:")
        true_labels_all_head = np.concatenate([item for item in true_labels_all_head])
        pred_labels_all_head = np.concatenate([item for item in pred_labels_all_head])
        calculate_f1_acc(pred_labels_all_head, true_labels_all_head)
        
        true_labels_each_head = np.array(true_labels_each_head)
        pred_labels_each_head = np.array(pred_labels_each_head)
        
        print("Results of each head:")
        for i in range(0,nheads):
            print(f"Head_{i}")
            i_head_true_labels = true_labels_each_head[:,i]
            i_head_true_labels = torch.cat([item for item in i_head_true_labels],0).to('cpu').numpy()
            
            i_head_pred_labels = pred_labels_each_head[:,i]
            i_head_pred_labels = torch.cat([item for item in i_head_pred_labels],0).to('cpu').numpy()
            calculate_f1_acc(i_head_pred_labels, i_head_true_labels)

    # ---------test--------
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

    print("########## Results of all heads:")
    true_labels_all_head = np.concatenate([item for item in true_labels_all_head])
    pred_labels_all_head = np.concatenate([item for item in pred_labels_all_head])
    calculate_f1_acc_test(pred_labels_all_head, true_labels_all_head, col_names)

    true_labels_each_head = np.array(true_labels_each_head)
    pred_labels_each_head = np.array(pred_labels_each_head)

    print("########### Results of each head:")
    for i in range(0,nheads):
        print(f"Head_{i}")
        i_head_true_labels = true_labels_each_head[:,i]
        i_head_true_labels = torch.cat([item for item in i_head_true_labels],0).to('cpu').numpy()

        i_head_pred_labels = pred_labels_each_head[:,i]
        i_head_pred_labels = torch.cat([item for item in i_head_pred_labels],0).to('cpu').numpy()
        calculate_f1_acc(i_head_pred_labels, i_head_true_labels)
