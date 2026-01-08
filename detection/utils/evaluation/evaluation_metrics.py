import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def _calc_sMAPE(n,m):
    if n == 0 and m == 0:
        return 0
    elif (n == 0 and m > 0) or (n > 0 and m == 0):
        return 1
    elif n > 0 and m > 0:
        # from https://arxiv.org/pdf/2108.01234.pdf
        return abs(n-m)/(abs(n) + abs(m))
    else: 
        return 1
    
def _calc_mae(n, m):
    return np.abs(n-m)

def _compute_pairwise_distances(gt, pred):
    pairwise_distances = cdist(gt, pred, metric="euclidean")

    if np.any(pairwise_distances):
        pairwise_distances = np.stack(pairwise_distances)
    return pairwise_distances

def _dev_percentage(n,m):
    if n > 0 and m > 0:
        return np.abs(((n - m)/n))
    elif (n > 0 and m == 0) or (n == 0 and m > 0):
        return 1
    elif n == 0 and m == 0:
        return 0
    else:
        raise Exception("Number of ground truths and/or preds are negative: gts = {n}, preds = {m}.")
        

def metric_coords(gts, preds, match_distance=45):
    """
    gt: [(x,y), (...), ...]
    pred: [(x,y), (...), ...]
    """
    n = len(gts)
    m = len(preds)

    if n == 0 and m == 0:
        return 1, 1, 1, 0, 0, 0
    
    elif (n == 0 and m > 0) or (n > 0 and m == 0):
        return 0, 0, 0, _dev_percentage(n, m), 1, _calc_mae(n, m)
    
    elif n > 0 and m > 0:
        pairwise_distances = _compute_pairwise_distances(gts, preds)
        if np.any(pairwise_distances):
            trivial = not np.any(pairwise_distances < match_distance)
            if trivial:
                true_positives = 0
            else:
                # match the predicted points to labels via linear cost assignment ('hungarian matching')
                max_distance = pairwise_distances.max()
                # the costs for matching: the first term sets a low cost for all pairs that are in
                # the matching distance, the second term sets a lower cost for shorter distances,
                # so that the closest points are matched
                costs = -(pairwise_distances < match_distance).astype(float) - (max_distance - pairwise_distances) / max_distance
                # perform the matching, returns the indices of the matched coordinates
                label_ind, pred_ind = linear_sum_assignment(costs)
                # check how many of the matches are smaller than the match distance
                # these are the true positives
                match_ok = pairwise_distances[label_ind, pred_ind] < match_distance
                true_positives = np.count_nonzero(match_ok)

            # compute false positives and false negatives
            false_positives = m - true_positives
            false_negatives = n - true_positives

            precision = true_positives / (true_positives + false_positives) if true_positives > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if true_positives > 0 else 0
            # from https://www.v7labs.com/blog/f1-score-guide#:~:text=The%20F1%20score%20is%20calculated,denotes%20a%20better%20quality%20classifier.
            f1 =  (2*precision*recall)/(precision+recall) if (precision > 0 and recall > 0) else 0
            return precision, recall, f1, _dev_percentage(n, m), _calc_sMAPE(n, m), _calc_mae(n, m)  
        else:
            return 0,0,0, _dev_percentage(n, m), _calc_sMAPE(n, m), _calc_mae(n, m)
    else:
        raise Exception(f"Number of ground truths and/or predictions are negative (metric): len(gts) = {n}, len(preds) = {m}.")
