"""
metrics.py — Shared Metric Functions

Provides:
    cosine_distance_matrix  — pairwise cosine distance [N_probe, N_gallery]
    compute_rank_k          — Rank-k accuracy
    compute_map             — mean Average Precision
    compute_cmc_curve       — full CMC curve (Rank-1 through Rank-K)
    compute_eer             — Equal Error Rate for verification
    compute_gender_metrics  — accuracy, balanced accuracy, F1 per class

All functions accept torch tensors and return Python floats or numpy arrays.
They are used by both gait_eval.py and gender_eval.py.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple


# ── Distance ────────────────────────────────────────────────────────────────

def cosine_distance_matrix(
    probe_emb: torch.Tensor,
    gallery_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise cosine distance matrix.

    Args:
        probe_emb:   [N_probe,   D] — L2 normalised inside this function
        gallery_emb: [N_gallery, D]

    Returns:
        dist: [N_probe, N_gallery]
              entry (i, j) = 1 - cosine_similarity(probe_i, gallery_j)
              range [0, 2]; lower = more similar
    """
    probe_norm   = F.normalize(probe_emb,   dim=1)
    gallery_norm = F.normalize(gallery_emb, dim=1)
    sim  = torch.mm(probe_norm, gallery_norm.t())
    dist = 1.0 - sim
    return dist


# ── Retrieval metrics ────────────────────────────────────────────────────────

def compute_rank_k(
    dist_matrix: torch.Tensor,
    probe_ids:   List[int],
    gallery_ids: List[int],
    k: int,
) -> float:
    """
    Rank-k accuracy: fraction of probes whose correct subject
    appears in the top-k nearest gallery entries.

    Args:
        dist_matrix: [N_probe, N_gallery]
        probe_ids:   [N_probe]   subject IDs
        gallery_ids: [N_gallery] subject IDs
        k:           rank cutoff

    Returns:
        rank_k: float in [0, 1]
    """
    gallery_t = torch.tensor(gallery_ids)
    probe_t   = torch.tensor(probe_ids)
    correct   = 0

    for i in range(len(probe_t)):
        sorted_idx = dist_matrix[i].argsort()
        top_k_ids  = gallery_t[sorted_idx[:k]]
        if probe_t[i] in top_k_ids:
            correct += 1

    return correct / len(probe_t)


def compute_map(
    dist_matrix: torch.Tensor,
    probe_ids:   List[int],
    gallery_ids: List[int],
) -> float:
    """
    mean Average Precision (mAP).

    For each probe compute AP over the ranked gallery,
    then average across all probes.

    Args:
        dist_matrix: [N_probe, N_gallery]
        probe_ids:   [N_probe]
        gallery_ids: [N_gallery]

    Returns:
        mAP: float in [0, 1]
    """
    gallery_t = torch.tensor(gallery_ids)
    probe_t   = torch.tensor(probe_ids)
    aps       = []

    for i in range(len(probe_t)):
        sorted_idx    = dist_matrix[i].argsort()
        sorted_labels = gallery_t[sorted_idx]
        is_match      = (sorted_labels == probe_t[i]).float()

        if is_match.sum() == 0:
            continue

        precisions = []
        n_correct  = 0
        for rank, match in enumerate(is_match):
            if match:
                n_correct += 1
                precisions.append(n_correct / (rank + 1))
        aps.append(float(np.mean(precisions)))

    return float(np.mean(aps)) if aps else 0.0


def compute_cmc_curve(
    dist_matrix: torch.Tensor,
    probe_ids:   List[int],
    gallery_ids: List[int],
    max_rank: int = 20,
) -> np.ndarray:
    """
    Compute the full CMC (Cumulative Match Characteristic) curve.

    Args:
        dist_matrix: [N_probe, N_gallery]
        probe_ids:   [N_probe]
        gallery_ids: [N_gallery]
        max_rank:    maximum rank to compute (default 20)

    Returns:
        cmc: [max_rank] array where cmc[k] = Rank-(k+1) accuracy
    """
    max_rank  = min(max_rank, len(gallery_ids))
    gallery_t = torch.tensor(gallery_ids)
    probe_t   = torch.tensor(probe_ids)
    cmc       = np.zeros(max_rank)

    for i in range(len(probe_t)):
        sorted_idx    = dist_matrix[i].argsort()
        sorted_labels = gallery_t[sorted_idx]
        is_match      = (sorted_labels == probe_t[i])

        # Find first match rank (0-indexed)
        match_ranks = is_match.nonzero(as_tuple=False)
        if len(match_ranks) == 0:
            continue
        first_match = match_ranks[0].item()
        if first_match < max_rank:
            cmc[first_match:] += 1

    cmc = cmc / len(probe_t)
    return cmc


# ── Verification metric: EER ─────────────────────────────────────────────────

def compute_eer(
    dist_matrix: torch.Tensor,
    probe_ids:   List[int],
    gallery_ids: List[int],
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) for identity verification.

    EER is the threshold at which FAR (False Accept Rate) equals
    FRR (False Reject Rate). Lower EER = better verification.

    For gait recognition with gallery size > 1, we treat each
    probe-gallery pair as a verification trial:
        - Genuine pair:  same subject ID
        - Impostor pair: different subject IDs

    The cosine distance threshold that equates FAR and FRR is the EER point.

    Args:
        dist_matrix: [N_probe, N_gallery] — cosine distances
        probe_ids:   [N_probe]
        gallery_ids: [N_gallery]

    Returns:
        eer:       float — Equal Error Rate in [0, 1]
        threshold: float — distance threshold at EER
    """
    gallery_t = torch.tensor(gallery_ids)
    probe_t   = torch.tensor(probe_ids)

    genuine_scores   = []
    impostor_scores  = []

    for i in range(len(probe_t)):
        for j in range(len(gallery_t)):
            dist = dist_matrix[i, j].item()
            if probe_t[i] == gallery_t[j]:
                genuine_scores.append(dist)
            else:
                impostor_scores.append(dist)

    genuine_scores  = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Sweep thresholds from min to max distance
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 1000)

    frr_list = []
    far_list = []

    for thresh in thresholds:
        # FRR: genuine pairs rejected (distance > threshold)
        frr = (genuine_scores  > thresh).mean()
        # FAR: impostor pairs accepted (distance <= threshold)
        far = (impostor_scores <= thresh).mean()
        frr_list.append(frr)
        far_list.append(far)

    frr_arr = np.array(frr_list)
    far_arr = np.array(far_list)

    # EER: point where |FAR - FRR| is minimised
    diff      = np.abs(far_arr - frr_arr)
    eer_idx   = diff.argmin()
    eer       = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2)
    threshold = float(thresholds[eer_idx])

    return eer, threshold


# ── Gender metrics ───────────────────────────────────────────────────────────

def compute_gender_metrics(
    preds:  torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """
    Compute gender classification metrics robust to class imbalance.

    Args:
        preds:  [N] predicted class indices (0=Male, 1=Female)
        labels: [N] ground truth class indices

    Returns:
        dict with:
            accuracy:          overall % correct
            balanced_accuracy: mean of per-class recalls
            F1_Male:           F1 for Male class
            F1_Female:         F1 for Female class
            precision_Male:    precision for Male
            recall_Male:       recall for Male
            precision_Female:  precision for Female
            recall_Female:     recall for Female
    """
    correct = (preds == labels)
    accuracy = correct.float().mean().item()

    results  = {}
    recalls  = []

    for cls, name in [(0, 'Male'), (1, 'Female')]:
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results[f'precision_{name}'] = precision
        results[f'recall_{name}']    = recall
        results[f'F1_{name}']        = f1
        recalls.append(recall)

    results['accuracy']          = accuracy
    results['balanced_accuracy'] = float(np.mean(recalls))
    return results
