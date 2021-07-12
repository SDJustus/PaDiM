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
    x0 = F.unfold(x0, kernel_size=(s, s), dilation=(1, 1), stride=(s, s))
    #print("shape after unfold", str(x0.shape))
    x0 = x0.view(b0, c0, -1, h1, w1)
    #print("shape after unfold_view", str(x0.shape))
    z = torch.zeros(b0, c0 + c1, x0.size(2), h1, w1).to(x0.device)
    for i in range(x0.size(2)):
        z[:, :, i, :, :] = torch.cat((x0[:, :, i, :, :], x1), 1)
    #print("shape before view", str(z.shape))
    z = z.view(b0, -1, h1 * w1)
    #print("shape after view", str(z.shape))
    z = F.fold(z, kernel_size=(s, s), output_size=(h0, w0), stride=(s, s))
    #print("shape after fold", str(z.shape))
    return z


def mean_smoothing(amaps: Tensor, kernel_size: int = 21) -> Tensor:
    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def compute_roc_score(amaps: NDArray, y_trues: NDArray, stems: List[str]) -> float:

    num_data = len(stems)
    y_scores = amaps.reshape(num_data, -1).max(axis=1)
    fprs, tprs, thresholds = roc_curve(y_trues, y_scores, pos_label=1, drop_intermediate=False)

    # Save roc_curve.csv
    keys = [f"threshold_{i}" for i in range(len(thresholds))]
    roc_df = pd.DataFrame({"key": keys, "fpr": fprs, "tpr": tprs, "threshold": thresholds})
    roc_df.to_csv("roc_curve.csv", index=False)

    # Update test_dataset.csv
    # pred_csv = pd.merge(
    #     pd.DataFrame({"stem": stems, "y_score": y_scores, "y_true": y_trues}),
    #     pd.read_csv("test_dataset.csv"),
    #     on="stem",
    # )
    # for i, th in enumerate(thresholds):
    #     pred_csv[f"threshold_{i}"] = pred_csv["y_score"].apply(lambda x: 1 if x >= th else 0)
    # pred_csv.to_csv("test_dataset.csv", index=False)

    print("np.unique", np.unique(y_trues))

    return roc_auc_score(y_trues, y_scores)


def compute_pro_score(amaps: NDArray, masks: NDArray) -> float:

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    max_step = 200
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / max_step

    for th in tqdm(np.arange(min_th, max_th, delta), desc="compute pro"):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                TP_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(TP_pixels / region.area)

        inverse_masks = 1 - masks
        FP_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = FP_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    df.to_csv("pro_curve.csv", index=False)
    return auc(df["fpr"], df["pro"])

def get_performance(y_trues, y_preds):
        fpr, tpr, t = roc_curve(y_trues, y_preds)
        roc_score = auc(fpr, tpr)
        ap = average_precision_score(y_trues, y_preds, pos_label=1)
        recall_dict = dict()
        precisions = [0.996, 0.99, 0.95, 0.9]
        temp_dict=dict()
        
        for th in t:
            y_preds_new = [1 if ele >= th else 0 for ele in y_preds] 
            if len(set(y_preds_new)) == 1:
                print("y_preds_new did only contain the element {}... Continuing with next iteration!".format(y_preds_new[0]))
                continue
            
            precision, recall, _, _ = precision_recall_fscore_support(y_trues, y_preds_new, average="binary", pos_label=1)
            temp_dict[str(precision)] = recall
        p_dict = OrderedDict(sorted(temp_dict.items(), reverse=True))
        for p in precisions:   
            for precision, recall in p_dict.items(): 
                if float(precision)<=p:
                    print(f"writing {p}; {precision}")
                    recall_dict["recall at pr="+str(p)+"(real_value="+str(precision)+")"] = recall
                    break
                else:
                    continue
    
        
        #Threshold
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(t, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
        threshold = roc_t['threshold']
        threshold = list(threshold)[0]
        
        
        
        y_preds = [1 if ele >= threshold else 0 for ele in y_preds] 
        
        
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_trues, y_preds, average="binary", pos_label=1)
        f05_score = fbeta_score(y_trues, y_preds, beta=0.5, average="binary", pos_label=1)
        #### conf_matrix = [["true_normal", "false_abnormal"], ["false_normal", "true_abnormal"]]     
        conf_matrix = confusion_matrix(y_trues, y_preds)
        performance = OrderedDict([ ('auc', roc_score), ("ap", ap), ('precision', precision),
                                    ("recall", recall), ("f1_score", f1_score), ("f05_score", f05_score), ("conf_matrix", conf_matrix),
                                    ("threshold", threshold)])
        performance.update(recall_dict)
                                    
        return performance, t, y_preds

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


def draw_roc_and_pro_curve(roc_score: float, pro_score: float) -> None:

    grid = ImageGrid(
        fig=plt.figure(figsize=(8, 8)),
        rect=111,
        nrows_ncols=(1, 1),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.15,
    )

    roc_df = pd.read_csv("roc_curve.csv")
    fpr = roc_df["fpr"]
    tpr = roc_df["tpr"]
    th = roc_df["threshold"]
    v_min = th.min()
    grid[0].plot(fpr, tpr, color="k", label=f"ROC Score: {round(roc_score, 3):.3f}", zorder=1)
    im = grid[0].scatter(fpr, tpr, s=8, c=th, cmap="jet", vmin=v_min, vmax=1, zorder=2)
    grid[0].set_xlim(-0.05, 1.05)
    grid[0].set_ylim(-0.05, 1.05)
    grid[0].set_xticks(np.arange(0, 1.1, 0.1))
    grid[0].set_yticks(np.arange(0, 1.1, 0.1))
    grid[0].tick_params(axis="both", labelsize=14)
    grid[0].set_xlabel("FPR: FP / (TN + FP)", fontsize=24)
    grid[0].set_ylabel("TPR: TP / (TP + FN)", fontsize=24)
    grid[0].xaxis.set_label_coords(0.5, -0.1)
    grid[0].yaxis.set_label_coords(-0.1, 0.5)
    grid[0].legend(fontsize=24)
    grid[0].grid(which="both", linestyle="dotted", linewidth=1)
    cb = plt.colorbar(im, cax=grid.cbar_axes[0])
    cb.ax.tick_params(labelsize="large")
    plt.savefig("roc_curve.png")
    plt.close()

    grid = ImageGrid(
        fig=plt.figure(figsize=(8, 8)),
        rect=111,
        nrows_ncols=(1, 1),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.15,
    )

    pro_df = pd.read_csv("pro_curve.csv")
    fpr = pro_df["fpr"]
    pro = pro_df["pro"]
    th = pro_df["threshold"]
    grid[0].plot(fpr, pro, color="k", label=f"PRO Score: {round(pro_score, 3):.3f}", zorder=1)
    im = grid[0].scatter(fpr, pro, s=8, c=th, cmap="jet", vmin=v_min, vmax=1, zorder=2)
    grid[0].set_xlim(-0.05, 1.05)
    grid[0].set_ylim(-0.05, 1.05)
    grid[0].set_xticks(np.arange(0, 1.1, 0.1))
    grid[0].set_yticks(np.arange(0, 1.1, 0.1))
    grid[0].tick_params(axis="both", labelsize=14)
    grid[0].set_xlabel("FPR: FP / (TN + FP)", fontsize=24)
    grid[0].set_ylabel("PRO: Per-Region Overlap", fontsize=24)
    grid[0].xaxis.set_label_coords(0.5, -0.1)
    grid[0].yaxis.set_label_coords(-0.1, 0.5)
    grid[0].legend(fontsize=24)
    grid[0].grid(which="both", linestyle="dotted", linewidth=1)
    cb = plt.colorbar(im, cax=grid.cbar_axes[0])
    cb.ax.tick_params(labelsize="large")
    plt.savefig("pro_curve.png")
    plt.close()


def savegif(imgs: NDArray, amaps: NDArray, masks: NDArray, stems: List[str]) -> None:

    os.mkdir("results")
    pbar = tqdm(enumerate(zip(stems, imgs, masks, amaps)), desc="savefig")
    for i, (stem, img, mask, amap) in pbar:

        # How to get two subplots to share the same y-axis with a single colorbar
        # https://stackoverflow.com/a/38940369
        grid = ImageGrid(
            fig=plt.figure(figsize=(12, 4)),
            rect=111,
            nrows_ncols=(1, 3),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.15,
        )

        img = denormalize(img)

        grid[0].imshow(img)
        grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[0].set_title("Input Image", fontsize=24)

        grid[1].imshow(img)
        grid[1].imshow(mask, alpha=0.3, cmap="Reds")
        grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[1].set_title("Ground Truth", fontsize=24)

        grid[2].imshow(img)
        im = grid[2].imshow(amap, alpha=0.3, cmap="jet", vmin=0, vmax=1)
        grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[2].cax.toggle_label(True)
        grid[2].set_title("Anomaly Map", fontsize=24)

        plt.colorbar(im, cax=grid.cbar_axes[0])
        plt.savefig(f"results/{stem}.png", bbox_inches="tight")
        plt.close()

    # NOTE(inoue): The gif files converted by PIL or imageio were low-quality.
    #              So, I used the conversion command (ImageMagick) instead.
    subprocess.run("convert -delay 100 -loop 0 results/*.png result.gif", shell=True)


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