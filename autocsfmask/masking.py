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


def compute_corrscore(mask, func_voxels):
    nslice = len(func_voxels)
    slicewise_scores = np.zeros(nslice)
    for islice in range(nslice):
        data_2d = func_voxels[islice]
        mask_flat = mask[islice]
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


def objective_amplitude_similarity(params, metrics_list, func_voxels, top_percent=25):
    """
    Objective: Encourage voxels within each slice to have similar amplitudes,
    restricted to the top X% (default 5%) of strongest voxels.
    """
    n_metrics = len(metrics_list)
    weights = ensure_sum_to_one(params[:n_metrics])
    thres = params[n_metrics:]
    
    mask = utils.get_mask(metrics_list, weights, thres=thres)
    nslice = len(func_voxels)
    
    slice_scores = []
    for islice in range(nslice):
        data_2d = func_voxels[islice]  # shape: (n_voxels, n_timepoints)
        mask_flat = mask[islice]
        masked_voxels = data_2d[mask_flat > 0]
        
        if masked_voxels.shape[0] < 2:
            continue  # skip slices with too few voxels
        
        # Compute amplitude per voxel (RMS over time)
        amplitudes = np.sqrt(np.mean(masked_voxels**2, axis=1))
        
        if amplitudes.size < 2 or np.mean(amplitudes) == 0:
            continue
        
        # Keep only top X% strongest voxels
        k = max(1, int(np.ceil(amplitudes.size * top_percent / 100.0)))
        top_amplitudes = np.sort(amplitudes)[-k:]
        
        # Coefficient of variation (std/mean) → lower is better
        cv = np.std(top_amplitudes) / np.mean(top_amplitudes)
        slice_scores.append(cv)
    
    if len(slice_scores) == 0:
        return 1.0  # worst case if no slices are valid
    
    return np.mean(slice_scores)



def objective_corr(params, metrics_list, func_voxels):
    n_metrics = len(metrics_list)
    weights = params[:n_metrics]
    thres = params[n_metrics:]
    mask = utils.get_mask(metrics_list, weights, thres=thres)
    corr_score = compute_corrscore(mask, func_voxels)
    return 1 - np.mean(corr_score)


def objective_da_sum(params, metrics_list, func_voxels):
    n_metrics = len(metrics_list)
    weights = ensure_sum_to_one(params[:n_metrics])
    thres = params[n_metrics:]
    mask = utils.get_mask(metrics_list, weights, thres=thres)
    s = utils.get_signal(func_voxels, mask)
    sproc = utils.scale_data(s)
    da = utils.compute_deattenuation_matrix(sproc)
    num_upper = (da.shape[1] * (da.shape[2] - 1)) / 2
    upper_mask = np.triu(np.ones((da.shape[1], da.shape[2]), dtype=bool), k=1)
    scores = (da[:, upper_mask] > 0).sum(axis=1) / num_upper
    score = scores.sum()/scores.size
    return 1 - score


def objective_dice(params, metrics_list, func_voxels, size_penalty_weight=1.0):
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


def objective_num_voxel(params, metrics_list, func_voxels, alpha=2):
    n_metrics = len(metrics_list)
    weights = ensure_sum_to_one(params[:n_metrics])
    thres = params[n_metrics:]
    mask = utils.get_mask(metrics_list, weights, thres=thres)
    masks = list(mask)
    nvoxels = [np.sum(m) for m in masks]
    return np.std(nvoxels)/np.mean(nvoxels) ** alpha



def voxel_list_to_volume(voxel_list, coords_list, full_shape):
    # Check dimensionality from first element
    example = voxel_list[0]
    if example.ndim == 1:
        volume = np.zeros(full_shape, dtype=example.dtype)
        for voxels, coords in zip(voxel_list, coords_list):
            volume[coords[:, 0], coords[:, 1], coords[:, 2]] = voxels

    elif example.ndim == 2:
        n_timepoints = example.shape[1]
        volume = np.zeros(full_shape + (n_timepoints,), dtype=example.dtype)
        for voxels, coords in zip(voxel_list, coords_list):
            for t in range(n_timepoints):
                volume[coords[:, 0], coords[:, 1], coords[:, 2], t] = voxels[:, t]

    else:
        raise ValueError("voxel_list elements must be 1D (static) or 2D (timeseries).")

    return volume


from scipy.ndimage import label

def objective_connectivity(params, metrics_list, func_voxels, coords, vol_shape, penalty_scale=1e6):
    n_metrics = len(metrics_list)
    weights = ensure_sum_to_one(params[:n_metrics])
    thres = params[n_metrics:]
    
    mask_list = utils.get_mask(metrics_list, weights, thres=thres)
    mask_vol = voxel_list_to_volume(mask_list, coords, vol_shape)
    
    total_voxels = mask_vol.sum()
    if total_voxels == 0:
        return penalty_scale  # empty mask is invalid
    
    # label 3D connected components
    structure = np.ones((3,3,3), dtype=bool)
    labeled, n_components = label(mask_vol, structure=structure)
    
    if n_components == 1:
        return 0.0  # fully connected
    
    # fraction of voxels not in largest component
    counts = np.bincount(labeled.flatten())
    counts[0] = 0  # ignore background
    largest = counts.max()
    fraction_disconnected = (total_voxels - largest) / total_voxels
    
    # additional penalty for multiple islands (each extra component)
    multi_island_penalty = (n_components - 1) / n_components  # between 0 and 1
    
    # combined penalty, scaled high
    penalty = (fraction_disconnected + multi_island_penalty) * penalty_scale
    return penalty


def objective_mixed(params, metrics_list, func_voxels, coords, vol_shape, penalty_scale=1e9):
    """
    Ensures the mask is fully connected within each slice AND across slices.
    Any disconnected mask receives an extremely high cost.
    """
    n_metrics = len(metrics_list)
    weights = ensure_sum_to_one(params[:n_metrics])
    thres = params[n_metrics:]
    
    mask_list = utils.get_mask(metrics_list, weights, thres=thres)
    mask_vol = voxel_list_to_volume(mask_list, coords, vol_shape)
    
    if mask_vol.sum() == 0:
        return penalty_scale  # empty mask is invalid
    
    # ---- 1. Check full 3D connectivity ----
    structure_3d = np.ones((3,3,3), dtype=bool)  # 26-neighbor connectivity
    labeled_3d, n_components_3d = label(mask_vol, structure=structure_3d)
    if n_components_3d > 1:
        return penalty_scale  # disconnected in 3D
    
    # ---- 2. Check connectivity within each slice ----
    structure_2d = np.ones((3,3), dtype=bool)  # 8-neighbor connectivity within slice
    for z in range(vol_shape[2]):
        slice_mask = mask_vol[:,:,z]
        if slice_mask.sum() == 0:
            continue  # empty slice, ignore
        labeled_slice, n_comp_slice = label(slice_mask, structure=structure_2d)
        if n_comp_slice > 1:
            return penalty_scale  # disconnected within this slice
    
    # ---- Fully connected → evaluate other objectives ----
    corr_cost = objective_corr(params, metrics_list, func_voxels)
    da_cost = objective_da_sum(params, metrics_list, func_voxels)
    # amp_cost = objective_amplitude_similarity(params, metrics_list, func_voxels)
    # voxel_cost = objective_num_voxel(params, metrics_list, func_voxels)
    
    # return mean of main objectives
    return np.mean([corr_cost, da_cost])

    
def get_mask_optim(metrics, func_voxels, coords, vol_shape, objective_func=objective_mixed):
    nslice = len(func_voxels)
    n_metrics = len(metrics)
    thres_bounds = [(0.25, 0.75)] * nslice
    bounds = [(-5, 5)] * n_metrics + thres_bounds 
    result = differential_evolution(objective_func, bounds,
                        args=(metrics, func_voxels, coords, vol_shape), 
                        strategy='best1bin', maxiter=1000, 
                        polish=True, disp=True)
    raw_weights = result.x[:n_metrics]
    optimal_weights = utils.softmax(raw_weights)
    optimal_thres = result.x[n_metrics:]
    final_mask = utils.get_mask(metrics, optimal_weights, thres=optimal_thres)
    print("=== Final Optimization Results ===")
    print(f"Final Weights: {np.array2string(optimal_weights, precision=4, separator=', ')}")
    print(f"Final Thresholds: {np.array2string(optimal_thres, precision=4, separator=', ')}")
    return final_mask, optimal_weights, optimal_thres, result.fun

