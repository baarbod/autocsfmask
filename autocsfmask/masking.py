import numpy as np
from scipy.optimize import differential_evolution
import autocsfmask.utils as utils

def compute_corrscore(mask, func_voxels, precomputed_full_corr=None):
    nslice = len(func_voxels)
    slicewise_scores = np.zeros(nslice)
    for islice in range(nslice):
        data_2d = func_voxels[islice]
        mask_flat = mask[islice]
        masked_voxels = data_2d[mask_flat > 0]
        if masked_voxels.shape[0] < 3:
            continue 
        corr_masked = np.corrcoef(masked_voxels)
        i, j = np.triu_indices(corr_masked.shape[0], k=1)
        masked_mean = corr_masked[i, j].mean()
        if precomputed_full_corr is not None:
            full_mean = precomputed_full_corr[islice]
        else:
            corr_full = np.corrcoef(data_2d)
            if_all, j_all = np.triu_indices(corr_full.shape[0], k=1)
            full_mean = corr_full[if_all, j_all].mean()
        slicewise_scores[islice] = masked_mean / (full_mean + 1e-8)
    max_val = np.max(slicewise_scores)
    if max_val > 0:
        slicewise_scores /= max_val
    return slicewise_scores

def objective_mixed(params, metrics_list, func_voxels, precomputed_full_corr):
    n_metrics = len(metrics_list)
    weights = utils.softmax(params[:n_metrics])
    thres = params[n_metrics:]
    mask = utils.get_mask(metrics_list, weights, thres=thres)
    
    # --- penalty for empty or near-empty slices ---
    empty_penalty = 0
    for islice in range(len(func_voxels)):
        # Require at least 3 voxels per slice
        if np.sum(mask[islice] > 0) < 3:
            empty_penalty += 100  # Massive penalty to ruin this solution's score
            
    corr_scores = compute_corrscore(mask, func_voxels, precomputed_full_corr)
    corr_cost = 1 - np.mean(corr_scores)
    
    return corr_cost + empty_penalty
    
def get_mask_optim(metrics, func_voxels):
    n_metrics = len(metrics)
    nslice = len(func_voxels)
    precomputed_full_corr = []
    for slice_data in func_voxels:
        c = np.corrcoef(slice_data)
        indices = np.triu_indices(c.shape[0], k=1)
        precomputed_full_corr.append(c[indices].mean())
    bounds = [(-1, 1)] * n_metrics + [(0.0, 3.0)] * nslice
    result = differential_evolution(
        objective_mixed, 
        bounds,
        args=(metrics, func_voxels, precomputed_full_corr),
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True, 
        disp=True
    )
    optimal_weights = utils.softmax(result.x[:n_metrics])
    optimal_thres = result.x[n_metrics:]
    final_mask = utils.get_mask(metrics, optimal_weights, thres=optimal_thres)
    return final_mask, optimal_weights, optimal_thres, result.fun