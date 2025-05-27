# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import differential_evolution
import autocsfmask.utils as utils
from scipy.signal import coherence


def get_mask_simple(metrics, weights, thres):
    assert np.sum(weights) == 1.0, 'Warning: Weights must sum to 1.0'
    mask = utils.get_mask(metrics, weights, thres=thres)
    return mask


# quantify the correlation across mask voxels, penalizing smaller masks
def compute_masked_correlation0(mask, func_data_window, use_penalty=False, penalty_scale=5):

    nslice = func_data_window.shape[2]
    slicewise_corrs = []

    for z in range(nslice):
        mask_slice = mask[:, :, z]
        data_slice = func_data_window[:, :, z, :]  # shape: (x, y, time)
        
        masked_voxels = data_slice[mask_slice > 0]
        
        if masked_voxels.shape[0] < 2:
            continue  # Skip slices with <2 voxels in mask

        normed = (masked_voxels - masked_voxels.mean(axis=1, keepdims=True)) / masked_voxels.std(axis=1, keepdims=True)
        corr_matrix = np.corrcoef(normed)

        i, j = np.triu_indices_from(corr_matrix, k=1)
        
        # # test
        # x = np.linspace(0, 1, 100)
        # plt.plot(x, np.exp(-5 * (x)))
        
        # scale mean correlation by fraction of masked voxels
        mask_fraction = masked_voxels.shape[0] / np.prod(mask.shape[:2])
        scaling = np.exp(-penalty_scale * (1 - mask_fraction)) if use_penalty else 1.0
        slicewise_corrs.append((1 - corr_matrix[i, j].mean()) * scaling)
        
        # slicewise_corrs.append((1 - corr_matrix[i, j].mean()) * masked_voxels.shape[0])
        
    # penalty = masked_voxels.shape[0] / np.prod(mask.shape)  # fraction of volume included
    # Average across slices (if any were valid)
    return np.mean(slicewise_corrs) if slicewise_corrs else 0


def compute_masked_correlation(mask, func_data_window, use_penalty=True, penalty_scale=10):
    nslice = func_data_window.shape[2]
    separation_scores = []

    for z in range(nslice):
        mask_slice = mask[:, :, z]
        data_slice = func_data_window[:, :, z, :]  # shape: (x, y, time)

        group1 = data_slice[mask_slice > 0]    # Masked
        group2 = data_slice[mask_slice == 0]   # Non-masked

        if group1.shape[0] < 2 or group2.shape[0] < 2:
            continue

        # Normalize time series (z-scoring over time)
        group1 = (group1 - group1.mean(axis=1, keepdims=True)) / group1.std(axis=1, keepdims=True)
        group2 = (group2 - group2.mean(axis=1, keepdims=True)) / group2.std(axis=1, keepdims=True)

        # Intra-group correlation matrices
        corr1 = np.corrcoef(group1)
        corr2 = np.corrcoef(group2)
        corr12 = np.corrcoef(group1, group2)
        inter = corr12[:group1.shape[0], group1.shape[0]:]  # Cross-group

        def upper_mean(mat):
            i, j = np.triu_indices_from(mat, k=1)
            return mat[i, j].mean()

        intra1 = upper_mean(corr1)
        intra2 = upper_mean(corr2)
        inter_mean = inter.mean()

        # Discriminability: higher = more separation
        discriminability = (0.5 * (intra1 + intra2) - inter_mean)

        # Optionally penalize small group sizes
        group_frac = group1.shape[0] / np.prod(mask.shape[:2])
        scaling = np.exp(-penalty_scale * (1 - group_frac)) if use_penalty else 1.0

        # Convert to a cost (lower is better for optimization)
        cost = -discriminability * scaling
        separation_scores.append(cost)

    return np.mean(separation_scores) if separation_scores else 0


# quantify the coherence across mask voxels, penalizing smaller masks
def compute_masked_coherence(mask, func_data_window, use_penalty=False, penalty_scale=5, fs=1/0.378):
    nslice = func_data_window.shape[2]
    slicewise_cohs = []

    for z in range(nslice):
        mask_slice = mask[:, :, z]
        data_slice = func_data_window[:, :, z, :]  # shape: (x, y, time)
        masked_voxels = data_slice[mask_slice > 0]
        
        if masked_voxels.shape[0] < 2:
            continue  # Skip slices with <2 voxels in mask

        # compute pairwise coherence
        n_vox = masked_voxels.shape[0]
        coh_values = []

        for i in range(n_vox):
            for j in range(i + 1, n_vox):
                f, Cxy = coherence(masked_voxels[i], masked_voxels[j], fs=fs)
                coh_values.append(np.mean(Cxy))  # average coherence across frequency

        if not coh_values:
            continue

        mean_coh = np.mean(coh_values)

        # scale by mask size
        mask_fraction = masked_voxels.shape[0] / np.prod(mask.shape[:2])
        scaling = np.exp(-penalty_scale * (1 - mask_fraction)) if use_penalty else 1.0
        slicewise_cohs.append((1 - mean_coh) * scaling)

    return np.mean(slicewise_cohs) if slicewise_cohs else 0


def objective_da_sum(params, metrics, func_data_window):
    # Split weights and threshold from params
    raw_weights  = params[:4]
    # force decay rate to 0
    raw_weights[2] = 0
    weights = utils.softmax(raw_weights)
    thres = params[4]
    # Compute the mask based on metrics and threshold
    mask = utils.get_mask(metrics, weights, thres=thres)
    # Compute da_sum
    s = utils.get_signal(func_data_window, mask)
    sproc = utils.scale_epi(s)
    sproc = utils.smooth_timeseries(sproc, window_size=5)
    da = utils.compute_deattenuation_matrix(sproc)
    da_sum = np.float64((np.triu(da) < 0).sum())
    return da_sum  # We want to minimize this


def objective_corr(params, metrics, func_data_window):
    # Split weights and threshold from params
    raw_weights  = params[:4]
    # # force decay rate to 0
    # raw_weights[2] = 0
    weights = utils.softmax(raw_weights)
    thres = params[4]
    # Compute the mask based on metrics and threshold
    mask = utils.get_mask(metrics, weights, thres=thres)
    corr_score = compute_masked_correlation(mask, func_data_window)
    # corr_score = compute_masked_coherence(mask, func_data_window)
    return corr_score


def get_mask_optim_all(metrics, func_data_window):    
    # find weights and threshold using global optimization

    bounds = [(-5, 5)] * 4 + [(0.01, 0.85)]

    n_iter = 10
    results = []
    
    weights_list = []
    thres_list = []
    
    for i in range(n_iter):
        result = differential_evolution(objective_corr, 
                                        bounds,
                                        args=(metrics, func_data_window), 
                                        strategy='best1bin', maxiter=100, polish=True)
        raw_weights = result.x[:4]
        weights = utils.softmax(raw_weights)
        thres = result.x[4]
        weights_list.append(weights)
        thres_list.append(thres)
        results.append(result)
        print(f"Run {i+1}/{n_iter}:")
        print(f"  Weights     = {weights}")
        print(f"  Threshold   = {thres:.4f}")
        print(f"  cost      = {result.fun:.6f}")

    avg_weights = utils.softmax(np.mean(weights_list, axis=0))
    avg_thres = np.mean(thres_list)

    print("\nAveraged Parameters:")
    print(f"  Weights   = {avg_weights}")
    print(f"  Threshold = {avg_thres:.4f}")

    # re-evaluate da_sum using averaged parameters
    mask = utils.get_mask(metrics, avg_weights, thres=avg_thres)
    s = utils.get_signal(func_data_window, mask)
    sproc = utils.scale_epi(s)
    sproc = utils.smooth_timeseries(sproc, window_size=5)
    da = utils.compute_deattenuation_matrix(sproc)
    da_sum = np.float64((np.triu(da) < 0).sum())

    print(f"Final da_sum (from averaged parameters): {da_sum:.2f}")

    return mask, avg_weights
    

def get_mask_hybrid(metrics, func_data_window):
    # optimize weights, then sweep to get best threshold

    max_iter=10
    tol=1e-3
    weight_list = []
    thres_list = []
    prev_thres = None
    
    for iter in range(max_iter):
        print(f"ITER = {iter}")

        if prev_thres is None:
            thres_bounds = (0.01, 1.0)
        else:
            lower = max(0.0, 0.9 * prev_thres)
            upper = min(1.0, 1.1 * prev_thres)
            thres_bounds = (lower, upper)
            print(f"Adaptive threshold bounds: {lower:.4f} to {upper:.4f}")
            
        # perform global optimization
        bounds = [(-5, 5)] * 4 + [thres_bounds]
        result = differential_evolution(objective_da_sum, 
                            bounds,
                            args=(metrics, func_data_window), 
                            strategy='best1bin', maxiter=100, polish=True)

        raw_weights = result.x[:4]
        
        # force decay rate to 0
        raw_weights[2] = 0
        
        optimal_weights = utils.softmax(raw_weights)
        optimal_thres = result.x[4]
        weight_list.append(optimal_weights)
                    
        print("Optimal Weights:", optimal_weights)
        print("Optimal Threshold:", optimal_thres)
        print("Minimum score:", result.fun)
        
        # define thresholds around globably optimized threshold
        thresholds = np.linspace(0.8*optimal_thres, 1.2*optimal_thres, 20)
        thresholds = thresholds[thresholds <= 1.0]  # Remove values > 1
        thresholds = thresholds[thresholds >= 0.0]  # Optional, in case future tweaks go < 0
        print(f"thres between {0.8*optimal_thres} and {1.2*optimal_thres}") 
        
        # optimize threshold with fixed weights
        scores = []
        for threshold in thresholds:
            mask = utils.get_mask(metrics, optimal_weights, thres=threshold)
            score = compute_masked_correlation(mask, func_data_window)
            score *= -1 * 10**6
            scores.append(score)
            print(f"Threshold {threshold:.2f} → Score = {score:.4f}")
            
        # get threshold with best correlation score
        best_idx = np.argmax(scores)
        best_thres = thresholds[best_idx]
        thres_list.append(best_thres)
        print(f"Best threshold: {best_thres}, Score: {scores[best_idx]}")
    
        # Early stopping if threshold stabilizes
        if prev_thres is not None and abs(best_thres - prev_thres) < tol:
            print(f"Early stopping: threshold changed < {tol}")
            break
        prev_thres = best_thres
    
    # average weights, and choose threshold after all iterations
    avg_weights = utils.softmax(np.mean(weight_list, axis=0))
    avg_thres = np.median(thres_list)

    print("Final Weights:", avg_weights)
    print("Final Threshold:", avg_thres)
    final_mask = utils.get_mask(metrics, avg_weights, thres=avg_thres)
    score = compute_masked_correlation(final_mask, func_data_window)
    print(f"Final Score: {score * -1 * 10**6}")

    return final_mask, avg_weights
