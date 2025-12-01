# -*- coding: utf-8 -*-

import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / e_x.sum()


def dice_score(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    size1 = mask1.sum()
    size2 = mask2.sum()
    if size1 + size2 == 0:
        return 1.0  # Treat two empty masks as perfectly overlapping
    return 2.0 * intersection / (size1 + size2)


def smooth_timeseries(data, window_size=5):
        """
        Smooths a time series or multiple time series using a moving average.

        Parameters:
            data (np.ndarray): 1D array (T,) or 2D array (T, N) where T is time and N is number of series.
            window_size (int): Size of the moving average window.

        Returns:
            np.ndarray: Smoothed array of the same shape as input.
        """
        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        data = np.asarray(data)
        if data.ndim == 1:
            padded = np.pad(data, (window_size//2, window_size-1-window_size//2), mode='edge')
            return np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
        elif data.ndim == 2:
            smoothed = np.empty_like(data, dtype=np.float64)
            for i in range(data.shape[1]):
                padded = np.pad(data[:, i], (window_size//2, window_size-1-window_size//2), mode='edge')
                smoothed[:, i] = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
            return smoothed
        else:
            raise ValueError("Input must be a 1D or 2D array")


def compute_deattenuation_matrix(s):
    # s: shape (ntime, nslice)
    # Result: da of shape (ntime, nslice, nslice), where da[t, i, j] = s[t, i] - s[t, j]
    return s[:, :, np.newaxis] - s[:, np.newaxis, :]


def get_mask(metrics, weights, thres=0.5):
    
    mean_metric_voxels = []
    nslice = len(metrics[0])
    for islice in range(nslice):
        slice_metrics = [metrics[i][islice] for i in range(len(metrics))]
        stacked = np.stack(slice_metrics, axis=0)
        weights = np.array(weights)
        mean_data = np.average(stacked, axis=0, weights=weights)
        min_val, max_val = np.min(mean_data), np.max(mean_data)
        data_norm = (mean_data - min_val) / (max_val - min_val) if max_val > min_val else 0
        mean_metric_voxels.append(data_norm)
        
    if np.isscalar(thres) or len(thres) == 1:
        mask = [(mean_metric_voxels[i] > thres).astype(np.uint8) for i in range(nslice)]
    else:
        if len(thres) != nslice:
            raise ValueError(f"Length of threshold vector ({thres.shape[2]}) must match number of slices ({nslice})")
        mask = [(mean_metric_voxels[i] > thres[i]).astype(np.uint8) for i in range(nslice)]
    return mask


def get_signal(func_voxels, mask):
    nslice = len(func_voxels)
    n_timepoints = func_voxels[0].shape[1]  # assume all slices have same number of timepoints
    s = np.zeros((n_timepoints, nslice))
    
    for i in range(nslice):
        slice_voxels = func_voxels[i]          # shape (n_voxels, n_timepoints)
        slice_mask = mask[i].astype(bool)      # shape (n_voxels,)
        
        if np.any(slice_mask):  # only if there are masked voxels
            masked_data = slice_voxels[slice_mask, :]  # select masked voxels
            s[:, i] = masked_data.mean(axis=0)
        else:
            s[:, i] = 0  # or np.nan if you prefer
    return s


def scale_data(s, bottom_pct=2.5, top_pct=2.5, divide_by_top=True):
    def mean_pct_portion(x, pct, fromtop=False):
        x_sorted = np.sort(x)
        if fromtop:
            x_sorted = x_sorted[::-1]
        n = max(1, round(len(x_sorted) * pct / 100))
        return np.mean(x_sorted[:n])
    s_copy = s.copy()
    for ch in range(s_copy.shape[1]):
        baseline = mean_pct_portion(s_copy[:, ch], bottom_pct)
        s_copy[:, ch] -= baseline
    if divide_by_top:
        top_val = mean_pct_portion(s_copy[:, 0], top_pct, fromtop=True)
        s_copy /= top_val
    return s_copy