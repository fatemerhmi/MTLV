import numpy as np
import torch 
def mtl_validation_test(validation_dataloader, head_count, device, nheads, model):
    # Variables to gather full output
    true_labels_each_head,pred_labels_each_head = [],[]
    true_labels_all_head,pred_labels_all_head = [],[]
    
    for i, batch in enumerate(validation_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            pred_label_b = []
            true_labels_b = []
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            for i in range(0,nheads):
                #remove -1 paddings:
                labels = b_labels[:,i,:]
                labels = labels[:,0:head_count[i]]

                true_labels_b.append(labels.to('cpu').numpy())
                pred = torch.sigmoid(outputs.logits[i])
                pred_label_b.append(pred.to('cpu').numpy())

            #store each head label seperatly
            true_labels_each_head.append(true_labels_b)
            pred_labels_each_head.append(pred_label_b)

            #store all head labels together
            true_labels_all_head.append(np.concatenate(true_labels_b,1))
            pred_labels_all_head.append(np.concatenate(pred_label_b,1))

    true_labels_all_head = np.concatenate(true_labels_all_head, axis=0)
    pred_labels_all_head = np.concatenate(pred_labels_all_head, axis=0)

    return pred_labels_all_head, true_labels_all_head, true_labels_each_head, pred_labels_each_head

