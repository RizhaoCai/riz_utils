import sklearn as skl
import numpy as numpy
import math
import numpy as np

def eval_stat(scores, labels, thr=0.5):
    # label 1 for the postive, and the label 2 for the negatvie
    pred = scores >= thr
    TN = np.sum((labels == 0) & (pred == False))  # True Negative   -- True Reject 
    FN = np.sum((labels == 1) & (pred == False))  # False Negative  -- False Reject
    FP = np.sum((labels == 0) & (pred == True))   # False Positive  -- False Accept
    TP = np.sum((labels == 1) & (pred == True))   # True Positive   -- True Accept
    return TN, FN, FP, TP

def get_thresholds(scores, grid_density):
    """
        @scores: a vector of scores with shape [n,1] or [n,]
    """
    # uniform thresholds in [min, max]
    Min, Max = min(scores), max(scores)
    thresholds = []
    for i in range(grid_density + 1):
        thresholds.append(Min + i * (Max - Min) / float(grid_density))
    return thresholds


def get_eer_stats(scores, labels, grid_density = 10000):
    thresholds = get_thresholds(scores, grid_density)
    min_dist = 1.0
    min_dist_stats = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_stat(scores, labels, thr)
        far = FP / float(TN + FP)  
        frr = FN / float(TP + FN)  
        dist = math.fabs(far - frr)
        if dist < min_dist:
            min_dist = dist
            min_dist_stats = [far, frr, thr]
    eer = (min_dist_stats[0] + min_dist_stats[1]) / 2.0
    thr = min_dist_stats[2]
    return eer, thr

def get_hter_at_thr(scores, labels, thr):
    TN, FN, FP, TP = eval_stat(scores, labels, thr)
    far = FP / float(TN + FP)   
    frr = FN / float(TP + FN)
    
    hter = (far + frr) / 2.0
    return hter,far,frr

def get_hter_at_thr_riz(scores, labels, thr):
    TN, FN, FP, TP = eval_stat(scores, labels, thr)
    #far = FP / float(TN + FP)   # wrong！！
    #frr = FN / float(TP + FN)
    far = FP / float(FP+TP)
    frr = FN / float(FN+TN)
    hter = (far + frr) / 2.0
    return hter,far,frr


def get_accuracy(scores, labels, thr):
    TN, FN, FP, TP = eval_stat(scores, labels, thr)
    accuracy = float(TP + TN) / len(scores)
    return accuracy

def get_best_thr(scores, labels, grid_density = 10000):
    thresholds = get_thresholds(scores, grid_density)
    acc_best = 0.0
    for thr in thresholds:
        acc = get_accuracy(scores, labels, thr)
        if acc > acc_best:
            acc_best = acc
            thr_best = thr
    return thr_best, acc_best