from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score, label_ranking_average_precision_score, precision_recall_fscore_support
import numpy as np
from sklearn import metrics

__all__ = ["calculate_scores_test", "calculate_scores"]

# def calculate_f1_acc(pred_labels, true_labels, threshold = 0.50):
#     target = true_labels
#     pred = np.array(pred_labels) >= 0.5

#     f1_micro = f1_score(target,pred,average='micro')*100
#     f1_macro = f1_score(target,pred,average='macro')*100
#     subset_accuracy = accuracy_score(target, pred)*100

#     prf = precision_recall_fscore_support(target, pred, average=None)
#     LRAP = label_ranking_average_precision_score(target, pred_labels)
#     return f1_micro, f1_macro, acc, LRAP, prf


def calculate_scores(outputs, targets ): #threshold = 0.50
    # outputs = np.array(outputs) >= 0.5
    
    threshold = calculate_f1_macro_opt(outputs, targets)
    outputs = np.array(outputs) >= threshold
    
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_micro = round(f1_score_micro*100,2)

    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    f1_score_macro = round(f1_score_macro*100,2)

    hamming_loss_ = metrics.hamming_loss(targets, outputs)
    hamming_loss_ = round(hamming_loss_*100,2)

    hamming_score_ = hamming_score(np.array(targets), np.array(outputs))
    hamming_score_ = round(hamming_score_*100,2)

    subset_accuracy = metrics.accuracy_score(targets, outputs)
    subset_accuracy = round(subset_accuracy*100,2)

    prf = precision_recall_fscore_support(targets, outputs, average=None)

    return f1_score_micro, f1_score_macro, hamming_loss_, hamming_score_, subset_accuracy, prf


def calculate_scores_test(pred_labels, true_labels, test_label_cols):
    threshold = calculate_f1_macro_opt(pred_labels, true_labels)

    target = true_labels
    pred = np.array(pred_labels) >= threshold
    
    f1_score_micro = metrics.f1_score(target, pred, average='micro')
    f1_score_micro = round(f1_score_micro*100,2)

    f1_score_macro = metrics.f1_score(target,pred,average='macro')
    f1_score_macro = round(f1_score_macro*100,2)

    subset_accuracy = metrics.accuracy_score(target, pred)
    subset_accuracy = round(subset_accuracy*100,2)
    
    hamming_loss_ = metrics.hamming_loss(target, pred)
    hamming_loss_ = round(hamming_loss_*100,2)

    hamming_score_ = hamming_score(np.array(target), np.array(pred))
    hamming_score_ = round(hamming_score_*100,2)

    clf_report = classification_report(target,pred,target_names=test_label_cols)

    prf = precision_recall_fscore_support(target, pred, average=None)

    return f1_score_micro, f1_score_macro, hamming_loss_, hamming_score_, subset_accuracy, clf_report

def calculate_f1_macro_opt(pred_labels, true_labels):
    # Calculate Accuracy - maximize F1 accuracy by tuning threshold values. First with 'macro_thresholds' on the order of e^-1 then with 'micro_thresholds' on the order of e^-2

    macro_thresholds = np.array(range(1,10))/10

    f1_results = []
    for th in macro_thresholds:
        # pred_bools = [pl>th for pl in pred_labels]
        pred = np.array(pred_labels) >= th
        test_f1_accuracy = f1_score(true_labels,pred,average='macro')
        f1_results.append(test_f1_accuracy)

    best_macro_th = macro_thresholds[np.argmax(f1_results)] #best macro threshold value
    return best_macro_th

    # micro_thresholds = (np.array(range(10))/100)+best_macro_th #calculating micro threshold values

    # f1_results, flat_acc_results = [], []
    # for th in micro_thresholds:
    #     pred_bools = [pl>th for pl in pred_labels]
    #     test_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')
    #     test_flat_accuracy = accuracy_score(true_bools, pred_bools)
    #     f1_results.append(test_f1_accuracy)
    #     flat_acc_results.append(test_flat_accuracy)

    # best_f1_idx = np.argmax(f1_results) #best threshold value

    # # Printing and saving classification report
    # print('Best Threshold: ', micro_thresholds[best_f1_idx])
    # print('Test F1 Score: ', f1_results[best_f1_idx])
    # print('Test Accuracy: ', flat_acc_results[best_f1_idx])

    # best_pred_bools = [pl>micro_thresholds[best_f1_idx] for pl in pred_labels]
    # clf_report_optimized = classification_report(true_bools,best_pred_bools, target_names=test_label_cols)
    # # pickle.dump(clf_report_optimized, open('classification_report_optimized.txt','wb'))
    # print(clf_report_optimized)

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)