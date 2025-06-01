# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import differential_evolution
from itertools import combinations
import autocsfmask.utils as utils


def get_mask_simple(metrics, weights, thres):
    assert np.isclose(np.sum(weights), 1.0), 'Weights must sum to 1.0'
    mask = utils.get_mask(metrics, weights, thres=thres)
    return mask


def ensure_sum_to_one(weights):
    if not np.isclose(np.sum(weights), 1.0):
        return utils.softmax(weights)
    else:
        return weights


def compute_corrscore(mask, func_data_window):
    nslice = func_data_window.shape[2]
    slicewise_scores = np.zeros(nslice)
    for islice in range(nslice):
        data_slice = func_data_window[:, :, islice, :]
        data_2d = data_slice.reshape(-1, data_slice.shape[-1])
        mask_flat = mask[:, :, islice].flatten()
        masked_voxels = data_2d[mask_flat > 0]
        full_voxels = data_2d
        if masked_voxels.shape[0] < 2:
            continue  # skip if mask is too small
        norm_full = (full_voxels - full_voxels.mean(axis=1, keepdims=True)) / full_voxels.std(axis=1, keepdims=True)
        norm_masked = (masked_voxels - masked_voxels.mean(axis=1, keepdims=True)) / masked_voxels.std(axis=1, keepdims=True)
        corr_full = np.corrcoef(norm_full)
        corr_masked = np.corrcoef(norm_masked)
        i, j = np.triu_indices_from(corr_masked, k=1)
        i_all, j_all = np.triu_indices_from(corr_full, k=1)
        masked_corr = corr_masked[i, j]
        full_corr = corr_full[i_all, j_all]
        masked_mean = masked_corr.mean()
        full_mean = full_corr.mean()
        score = masked_mean / full_mean if full_mean > 0 else 0.0     
        slicewise_scores[islice] = score
        
    def normalize_corrscore_ratio(score_array):
        score_array = np.clip(score_array, 0, np.inf)
        max_val = np.max(score_array)
        return score_array / (max_val + 1e-8) if max_val > 0 else score_array    
    
    return normalize_corrscore_ratio(slicewise_scores)


def objective_corr(params, metrics_list, func_data_window):
    n_metrics = len(metrics_list)
    weights = params[:n_metrics]
    thres = params[n_metrics:]
    mask = utils.get_mask(metrics_list, weights, thres=thres)
    corr_score = compute_corrscore(mask, func_data_window)
    return 1 - np.mean(corr_score)


def objective_da_sum(params, metrics_list, func_data_window):
    n_metrics = len(metrics_list)
    weights = ensure_sum_to_one(params[:n_metrics])
    thres = params[n_metrics:]
    mask = utils.get_mask(metrics_list, weights, thres=thres)
    s = utils.get_signal(func_data_window, mask)
    sproc = utils.scale_epi(s)
    da = utils.compute_deattenuation_matrix(sproc)
    num_upper = (da.shape[1] * (da.shape[2] - 1)) / 2
    upper_mask = np.triu(np.ones((da.shape[1], da.shape[2]), dtype=bool), k=1)
    scores = (da[:, upper_mask] > 0).sum(axis=1) / num_upper
    score = scores.sum()/scores.size
    return 1 - score


def objective_dice(params, metrics_list, func_data_window, size_penalty_weight=1.0):
    n_metrics = len(metrics_list)
    weights = ensure_sum_to_one(params[:n_metrics])
    thres = params[n_metrics:]
    mask = utils.get_mask(metrics_list, weights, thres=thres)
    masks = list(mask)
    pairwise_dice = [utils.dice_score(m1, m2) for m1, m2 in combinations(masks, 2)]
    mean_dice = np.mean(pairwise_dice)
    mask_sizes = np.array([m.sum() for m in masks])
    normalized_sizes = mask_sizes / mask[0].size
    mean_size = np.mean(normalized_sizes)
    penalty = size_penalty_weight * mean_size
    return min(1.0, (1 - mean_dice) + penalty)


def objective_mixed(params, metrics_list, func_data_window):
    corr_cost = objective_corr(params, metrics_list, func_data_window)
    da_cost = objective_da_sum(params, metrics_list, func_data_window)
    dice_cost = objective_dice(params, metrics_list, func_data_window)
    return np.mean([corr_cost, da_cost, dice_cost])

    
def get_mask_optim(metrics, func_data_window):
    nslice = func_data_window.shape[2]
    n_metrics = len(metrics)
    thres_bounds = [(0.2, 0.8)] * nslice
    bounds = [(-5, 5)] * n_metrics + thres_bounds 
    result = differential_evolution(objective_mixed, bounds,
                        args=(metrics, func_data_window), 
                        strategy='best1bin', maxiter=100, 
                        polish=True, disp=True)
    raw_weights = result.x[:n_metrics]
    optimal_weights = utils.softmax(raw_weights)
    optimal_thres = result.x[n_metrics:]
    final_mask = utils.get_mask(metrics, optimal_weights, thres=optimal_thres)
    print("=== Final Optimization Results ===")
    print(f"Final Weights: {np.array2string(optimal_weights, precision=4, separator=', ')}")
    print(f"Final Thresholds: {np.array2string(optimal_thres, precision=4, separator=', ')}")
    return final_mask, optimal_weights, optimal_thres

