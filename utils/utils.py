"""
Utils module

The code from this file comes from:
    * https://github.com/taikiinoue45/PaDiM
"""
from collections import OrderedDict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray as NDArray
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support, average_precision_score, fbeta_score
from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn.functional as F
import os
import json


def embeddings_concat(x0: Tensor, x1: Tensor) -> Tensor:
    b0, c0, h0, w0 = x0.size()
    _, c1, h1, w1 = x1.size()
    s = h0 // h1
    #print("shape before unfold", str(x0.shape))
    #print(x0)
    x0 = F.unfold(x0, kernel_size=(s, s), dilation=(1, 1), stride=(s, s))
    #print(x0)
    #print("------")
    #print("shape after unfold", str(x0.shape))
    x0 = x0.view(b0, c0, -1, h1, w1)
    #print(x0)
    #print("shape after unfold_view", str(x0.shape))
    #print(x1)
    z = torch.zeros(b0, c0 + c1, x0.size(2), h1, w1).to(x0.device)
    for i in range(x0.size(2)):
        z[:, :, i, :, :] = torch.cat((x0[:, :, i, :, :], x1), 1)
    #print(z)
    #print("shape before view", str(z.shape))
    z = z.view(b0, -1, h1 * w1)
    #print("shape after view", str(z.shape))
    #print(z)
    z = F.fold(z, kernel_size=(s, s), output_size=(h0, w0), stride=(s, s))
    #print(z)
    #print("shape after fold", str(z.shape))
    return z


def mean_smoothing(amaps: Tensor, kernel_size: int = 21) -> Tensor:
    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def get_performance(y_trues, y_preds, manual_threshold):
    fpr, tpr, t = roc_curve(y_trues, y_preds)
    roc_score = auc(fpr, tpr)
    ap = average_precision_score(y_trues, y_preds, pos_label=1)
    recall_dict = dict()
    precisions = [0.996, 0.99, 0.95, 0.9]
    temp_dict=dict()
    min_thresh = 0.9*min(y_preds)
    max_thresh = 1.1*max(y_preds)
    print(max_thresh)
    mov_thresh = np.random.default_rng().uniform(min_thresh, max_thresh, 400)
    print(mov_thresh.shape)
    for th in sorted(mov_thresh, reverse=True):
        y_preds_new = [1 if ele >= th else 0 for ele in y_preds] 
        if len(set(y_preds_new)) == 1:
            print("y_preds_new did only contain the element {}... Continuing with next iteration!".format(y_preds_new[0]))
            continue
        
        precision, recall, _, _ = precision_recall_fscore_support(y_trues, y_preds_new, average="binary", pos_label=1)
        temp_dict[str(precision)] = recall
        print("writing")
    p_dict = OrderedDict(sorted(temp_dict.items(), reverse=False))
    # interploation
    print("interpolation steps", len(list(p_dict.keys())))
    for i in range(len(list(p_dict.keys())), 0, -1):
        print(i)
        try:
            if p_dict[list(p_dict.keys())[i-1]]>p_dict[list(p_dict.keys())[i-2]]:
                p_dict[list(p_dict.keys())[i-2]] = p_dict[list(p_dict.keys())[i-1]]
        except IndexError:
            print("finished interpolation")
    p_dict = OrderedDict(sorted(p_dict.items(), reverse=True))
    for p in precisions:   
        for precision, recall in p_dict.items(): 
            recall_dict["recall at pr="+str(p)] = 0.0
            recall_dict["true pr="+str(p)] = 0.0
            while float(precision)>p:
                recall_dict["recall at pr="+str(p)] = recall
                recall_dict["true pr="+str(p)] = float(precision)
                continue
            else:
                break

    
    # auroc Threshold
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(t, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    auc_threshold = roc_t['threshold']
    auc_threshold = list(auc_threshold)[0]
    
    
    
    y_preds_auc_thresh = [1 if ele >= auc_threshold else 0 for ele in y_preds] 
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_trues, y_preds_auc_thresh, average="binary", pos_label=1)
    f05_score = fbeta_score(y_trues, y_preds_auc_thresh, beta=0.5, average="binary", pos_label=1)
    #### conf_matrix = [["true_normal", "false_abnormal"], ["false_normal", "true_abnormal"]]     
    conf_matrix = confusion_matrix(y_trues, y_preds_auc_thresh)
    performance = OrderedDict([ ('auc', roc_score), ("ap", ap), ('precision', precision),
                                ("recall", recall), ("f1_score", f1_score), ("f05_score", f05_score), ("conf_matrix", conf_matrix),
                                ("threshold", auc_threshold)])
    
    if manual_threshold:
        man_dict = dict()
        y_preds_man_thresh = [1 if ele >= manual_threshold else 0 for ele in y_preds]
        precision_man, recall_man, f1_score_man, _ = precision_recall_fscore_support(y_trues, y_preds_man_thresh, average="binary", pos_label=1)
        f05_score_man = fbeta_score(y_trues, y_preds_man_thresh, beta=0.5, average="binary", pos_label=1)
        #### conf_matrix = [["true_normal", "false_abnormal"], ["false_normal", "true_abnormal"]]     
        conf_matrix_man = confusion_matrix(y_trues, y_preds_man_thresh)
        man_dict["manual_threshold"] = manual_threshold
        man_dict["precision_man"] = precision_man
        man_dict["recall_man"] = recall_man
        man_dict["f1_score_man"] = f1_score_man
        man_dict["f05_score_man"] = f05_score_man
        man_dict["conf_matrix_man"] = conf_matrix_man
        performance.update(man_dict)
    performance.update(recall_dict)
                                
    return performance, t, y_preds_man_thresh if manual_threshold else None, y_preds_auc_thresh

def get_values_for_pr_curve(y_trues, y_preds, thresholds):
    precisions = []
    recalls = []
    tn_counts = []
    fp_counts = []
    fn_counts = []
    tp_counts = []
    for threshold in thresholds:
        y_preds_new = [1 if ele >= threshold else 0 for ele in y_preds] 
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds_new).ravel()
        if len(set(y_preds_new)) == 1:
            print("y_preds_new did only contain the element {}... Continuing with next iteration!".format(y_preds_new[0]))
            continue
        
        precision, recall, _, _ = precision_recall_fscore_support(y_trues, y_preds_new, average="binary", pos_label=1)
        precisions.append(precision)
        recalls.append(recall)
        tn_counts.append(tn)
        fp_counts.append(fp)
        fn_counts.append(fn)
        tp_counts.append(tp)
        
        
    
    return np.array(tp_counts), np.array(fp_counts), np.array(tn_counts), np.array(fn_counts), np.array(precisions), np.array(recalls), len(thresholds)


def denormalize(img: NDArray) -> NDArray:
    mean = np.array([0.4209137, 0.42091936, 0.42130423])
    std = np.array([0.34266332, 0.34264612, 0.3432589])
    img = (img * std + mean) * 255.0
    return img.astype(np.uint8)


def write_inference_result(file_names, y_preds, y_trues, outf):
        classification_result = {"tp": [], "fp": [], "tn": [], "fn": []}
        for file_name, gt, anomaly_score in zip(file_names, y_trues, y_preds):
            anomaly_score=int(anomaly_score)
            if gt == anomaly_score == 0:
                classification_result["tp"].append(file_name)
            if anomaly_score == 0 and gt != anomaly_score:
                classification_result["fp"].append(file_name)
            if gt == anomaly_score == 1:
                classification_result["tn"].append(file_name)
            if anomaly_score == 1 and gt != anomaly_score:
                classification_result["fn"].append(file_name)
        print("clas_res", classification_result)
        with open(outf, "w") as file:
            json.dump(classification_result, file, indent=4)
        
def get_values_for_roc_curve(y_trues, y_preds):
        fpr, tpr, _ = roc_curve(y_trues, y_preds) 
        return fpr, tpr, auc(fpr, tpr)
