import numpy as np
from skimage.metrics import structural_similarity

def prep_clf(obs, pre, threshold=0.5):
    """
    func: 二分類結果，混淆矩陣的四個元素
    inputs:
        obs: 觀測值；
        pre: 預測值；
        threshold: 閾值 判別正負樣本的閾值 默認0.1,氣象上默認格點>=0.1才判定存在降水
    returns:
        hits, misses, falsealarms, correctnegatives
        # aliases: TP, FN, FP, TN
    """
    
    # 根據閾值分類為0,1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    bias = (hits + misses) / (hits + falsealarms)

    return hits, misses, falsealarms, correctnegatives, bias

def FAR_POD(obs, pre, threshold=0.5):
    TP, FN, FP, TN, bias = prep_clf(obs=obs, pre=pre, threshold=threshold)

    far = FP / (TP + FP)
    pod = TP / (TP + FN)
    csi = TP / (TP + FN + FP)
    sr = 1 - far

    # 計算HSS
    # hss_num = 2 * ((TP * TN) - (FP * FN))
    # hss_den = ((TP + FP) * (FP + TN)) + ((TP + FN) * (FN + TN))
    # hss = hss_num / hss_den

    # num = (TP + FP) * (TP + FN)
    # den = TP + FP + TN + FN
    # Dr = num / den
    # ets = (TP - Dr) / (TP + FN + FP - Dr)

    # return csi, hss, far, pod, ets, bias
    return csi, far, pod, bias, sr