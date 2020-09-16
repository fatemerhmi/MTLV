from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score, label_ranking_average_precision_score
import numpy as np

__all__ = ["calculate_f1_acc", "calculate_f1_acc_test"]

def calculate_f1_acc(pred_labels, true_labels, threshold = 0.50):
    pred_bools = [pl>threshold for pl in pred_labels]
    true_bools = [tl==1 for tl in true_labels]
    val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')*100
    val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100

    print('Validation F1: ', val_f1_accuracy)
    print('Validation Accuracy: ', val_flat_accuracy)

def calculate_f1_acc_test(pred_labels, true_labels, test_label_cols, threshold = 0.50):
    pred_bools = [pl>threshold for pl in pred_labels]
    true_bools = [tl==1 for tl in true_labels] 
    print("-----------test-----------")
    print("Threshold: 0.5")
    print('Test F1 Score: ', f1_score(true_bools, pred_bools,average='micro'))
    print('Test Accuracy: ', accuracy_score(true_bools, pred_bools))
    print('LRAP: ', label_ranking_average_precision_score(true_labels, pred_labels) ,'\n')
    clf_report = classification_report(true_bools,pred_bools,target_names=test_label_cols)
    # pickle.dump(clf_report, open('classification_report.txt','wb')) #save report
    print(clf_report)

    # Calculate Accuracy - maximize F1 accuracy by tuning threshold values. First with 'macro_thresholds' on the order of e^-1 then with 'micro_thresholds' on the order of e^-2
    print("-----Optimizing threshold value for micro F1 score-----")

    macro_thresholds = np.array(range(1,10))/10

    f1_results, flat_acc_results = [], []
    for th in macro_thresholds:
        pred_bools = [pl>th for pl in pred_labels]
        test_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')
        test_flat_accuracy = accuracy_score(true_bools, pred_bools)
        f1_results.append(test_f1_accuracy)
        flat_acc_results.append(test_flat_accuracy)

    best_macro_th = macro_thresholds[np.argmax(f1_results)] #best macro threshold value

    micro_thresholds = (np.array(range(10))/100)+best_macro_th #calculating micro threshold values

    f1_results, flat_acc_results = [], []
    for th in micro_thresholds:
        pred_bools = [pl>th for pl in pred_labels]
        test_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')
        test_flat_accuracy = accuracy_score(true_bools, pred_bools)
        f1_results.append(test_f1_accuracy)
        flat_acc_results.append(test_flat_accuracy)

    best_f1_idx = np.argmax(f1_results) #best threshold value

    # Printing and saving classification report
    print('Best Threshold: ', micro_thresholds[best_f1_idx])
    print('Test F1 Score: ', f1_results[best_f1_idx])
    print('Test Accuracy: ', flat_acc_results[best_f1_idx])
    print('LRAP: ', label_ranking_average_precision_score(true_labels, pred_labels) , '\n')

    best_pred_bools = [pl>micro_thresholds[best_f1_idx] for pl in pred_labels]
    clf_report_optimized = classification_report(true_bools,best_pred_bools, target_names=test_label_cols)
    # pickle.dump(clf_report_optimized, open('classification_report_optimized.txt','wb'))
    print(clf_report_optimized)