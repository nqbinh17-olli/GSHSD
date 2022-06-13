from typing import Tuple
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

class F1Score:
    def __init__(self):
        pass
    @staticmethod
    def __calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.
        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score
    @staticmethod
    def __calc_f1_count_for_label(predictions: torch.Tensor,
                                labels: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label
        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label
        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()
        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(true_positive),
                                precision)
        recall = torch.div(true_positive, true_count)
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        return f1, true_count
    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor, average: str = 'weighted') -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.
        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            average: averaging method
        Returns:
            f1 score
        """
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')
            
        if average == 'micro':
            return self.__calc_f1_micro(predictions, labels)
        f1_score = 0
        f1_all = []
        for label_id in range(1, len(labels.unique()) + 1):
            f1, true_count = self.__calc_f1_count_for_label(predictions, labels, label_id)
            f1_all.append(f1.data.item())
            if average == 'weighted':
                f1_score += f1 * true_count
            elif average == 'macro':
                f1_score += f1
            
        if average == 'weighted':
            f1_score = torch.div(f1_score, len(labels))
        elif average == 'macro':
            f1_score = torch.div(f1_score, len(labels.unique()))
            
        return f1_score, np.array(f1_all)

def gmean(TP, FN, FP, TN):
    return np.sqrt((TP/(TP+FN))*(TN/(TN+FP)))

def cal_gmean(y_true, y_pred, return_type="mean"):
    assert isinstance(return_type, str) and return_type.lower() in ["total", "harmonic_mean", "mean", "class"],\
        f'return_type expected in ["total", "harmonic_mean", "mean", "class"], but got {return_type}'
    
    return_type = return_type.lower()
    labels = set(y_true)
    cf_metric = confusion_matrix(y_true, y_pred)
    TP_list, FN_list, FP_list, TN_list = [],[],[],[]
    gmean_list = []
    for l in range(len(labels)):
        TP = cf_metric[l,l]
        FN = cf_metric[l,:].sum() - cf_metric[l,l]
        FP = cf_metric[:,l].sum() - cf_metric[l,l]
        TN = cf_metric.sum() - (TP + FN + FP)
        TP_list.append(TP)
        FN_list.append(FN)
        FP_list.append(FP)
        TN_list.append(TN)
        gmean_list.append(gmean(TP, FN, FP, TN))
    
    if return_type == "total":
        TP = np.array(TP_list).sum()
        FN = np.array(FN_list).sum()
        FP = np.array(FP_list).sum()
        TN = np.array(TN_list).sum()
        return gmean(TP, FN, FP, TN)

    elif return_type == "harmonic_mean":
        zero_check = ((np.array(gmean_list) == 0.0)*1.0).sum()
        if zero_check >= 1:
            return 0.0
        invert = 1/np.array(gmean_list)
        invert = invert.sum()
        return len(gmean_list)/invert

    elif return_type == "mean":
        return np.array(gmean_list).mean()

    elif return_type == "class":
        result_dict = {}
        for i, ln in enumerate(list(labels)):
            result_dict[ln] = gmean_list[i]
        return result_dict