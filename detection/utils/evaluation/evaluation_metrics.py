import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from detection.config import CZII_MATCH_DISTANCE_VOXELS, DEFAULT_MATCH_DISTANCE


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


def resolve_match_distance(n_gts, match_distance=None, gt_classes=None):
    """
    Resolve the match tolerance into an array of one distance per ground-truth point.

    Coordinates are in voxels (picks in Angstrom / VOXEL_SIZE_A), so the tolerance is too.
    Precedence: an explicit `match_distance` (scalar or per-GT array) wins; otherwise the
    CZII criterion 0.5 * particle radius is looked up per class from `gt_classes`;
    with neither, DEFAULT_MATCH_DISTANCE is used for every point.
    """
    if match_distance is not None:
        md = np.asarray(match_distance, dtype=float)
        if md.ndim == 0:
            return np.full(n_gts, float(md))
        if len(md) != n_gts:
            raise ValueError(f"match_distance has {len(md)} entries but there are {n_gts} ground truths.")
        return md

    if gt_classes is None:
        return np.full(n_gts, DEFAULT_MATCH_DISTANCE)

    if len(gt_classes) != n_gts:
        raise ValueError(f"gt_classes has {len(gt_classes)} entries but there are {n_gts} ground truths.")
    return np.array(
        [CZII_MATCH_DISTANCE_VOXELS.get(str(cls), DEFAULT_MATCH_DISTANCE) for cls in gt_classes],
        dtype=float,
    )


def hungarian_match(gts, preds, match_distance=None, gt_classes=None):
    """
    Match ground-truth points to predictions one-to-one via linear cost assignment.

    The tolerance may differ per ground-truth point (see `resolve_match_distance`), which is
    what the CZII criterion needs: a ribosome is found within 7.5 voxels, an apo-ferritin
    within 3.0.

    Returns:
        gt_idx, pred_idx: index arrays of the pairs that are within tolerance.
    """
    n, m = len(gts), len(preds)
    if n == 0 or m == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    distances = cdist(np.asarray(gts, dtype=float), np.asarray(preds, dtype=float), metric="euclidean")
    tolerance = resolve_match_distance(n, match_distance, gt_classes)[:, np.newaxis]

    # The first term gives every in-tolerance pair a lower cost than any out-of-tolerance pair,
    # the second term prefers the closer pair among those, so the assignment is driven by the
    # tolerance first and by distance only as a tie-break.
    max_distance = distances.max()
    if max_distance == 0:  # all points coincide
        max_distance = 1.0
    costs = -(distances < tolerance).astype(float) - (max_distance - distances) / max_distance

    gt_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = distances[gt_ind, pred_ind] < tolerance[gt_ind, 0]
    return gt_ind[match_ok], pred_ind[match_ok]


def metric_coords_per_class(gts, gt_classes, preds, match_distance=None):
    """
    Per-class detection metrics from a single global matching, so a prediction can only be a
    true positive for one class.

    Detection is class-agnostic: a prediction carries no class, so per-class *precision* (and
    everything derived from it) is undefined and deliberately not reported. Only recall and the
    raw counts are meaningful here; use `metric_coords` for the tomogram-level precision/F1.

    gts: list/array of GT coordinates (voxels)
    gt_classes: list of class label strings, one per GT
    preds: array of prediction coordinates (voxels)
    match_distance: override for the tolerance; by default the CZII 0.5 * radius per class.

    returns: dict mapping class -> {"recall", "matched", "label_count"}
    """
    matched_gt_indices, _ = hungarian_match(gts, preds, match_distance, gt_classes)
    matched_gt_indices = set(matched_gt_indices.tolist())

    class_gt_indices = {}
    for i, cls in enumerate(gt_classes):
        class_gt_indices.setdefault(cls, []).append(i)

    results = {}
    for cls, gt_indices in class_gt_indices.items():
        n_cls = len(gt_indices)
        tp = sum(1 for i in gt_indices if i in matched_gt_indices)
        results[cls] = {
            "recall": tp / n_cls if n_cls > 0 else 0,
            "matched": tp,
            "label_count": n_cls,
        }

    return results


def metric_coords(gts, preds, match_distance=None, gt_classes=None):
    """
    Tomogram-level detection metrics.

    gt: [(z,y,x), (...), ...] in voxels
    pred: [(z,y,x), (...), ...] in voxels
    match_distance: override for the tolerance; by default the CZII 0.5 * radius looked up
        per point from `gt_classes` (and DEFAULT_MATCH_DISTANCE where no classes are given).
    """
    n = len(gts)
    m = len(preds)

    if n < 0 or m < 0:
        raise Exception(f"Number of ground truths and/or predictions are negative (metric): len(gts) = {n}, len(preds) = {m}.")

    if n == 0 and m == 0:
        return 1, 1, 1, 0, 0, 0

    if n == 0 or m == 0:
        return 0, 0, 0, _dev_percentage(n, m), 1, _calc_mae(n, m)

    matched_gt, _ = hungarian_match(gts, preds, match_distance, gt_classes)
    true_positives = len(matched_gt)

    false_positives = m - true_positives
    false_negatives = n - true_positives

    precision = true_positives / (true_positives + false_positives) if true_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives > 0 else 0
    # from https://www.v7labs.com/blog/f1-score-guide#:~:text=The%20F1%20score%20is%20calculated,denotes%20a%20better%20quality%20classifier.
    f1 = (2*precision*recall)/(precision+recall) if (precision > 0 and recall > 0) else 0
    return precision, recall, f1, _dev_percentage(n, m), _calc_sMAPE(n, m), _calc_mae(n, m)
