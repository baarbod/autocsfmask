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
        

def scale_epi(s, startind=0, remove_offset=True):
    
    def mean_pct_portion(x, pct, fromtop=False):
        # compute the mean top/bottom percent of signal

        x = np.sort(x, axis=0)
        if fromtop:
            x = np.flipud(x)
        num_rows, _ = x.shape
        n = round(num_rows * pct / 100)
        return np.mean(x[:n, :], axis=0)
    
    # input csf after loading from file
    s = s[startind:, :]
    if remove_offset:
        s -= mean_pct_portion(s, 15)
    slice1_maximum = mean_pct_portion(s[:, [0]], 5, fromtop=True)
    s /= slice1_maximum
    return s


def compute_deattenuation_matrix(s):
    # s: shape (ntime, nslice)
    # Result: da of shape (ntime, nslice, nslice), where da[t, i, j] = s[t, i] - s[t, j]
    return s[:, :, np.newaxis] - s[:, np.newaxis, :]


def get_mask(metrics, weights, thres=0.5):
    weights = np.array(weights, dtype=float).reshape(-1, 1, 1, 1)  # Reshape for broadcasting
    stacked_arrays = np.stack(metrics, axis=0)
    # Compute unnormalized mean metric
    mean_metric_unnorm = np.sum(stacked_arrays * weights, axis=0)
    # Normalize mean metric slice-wise
    mean_metric = np.zeros_like(mean_metric_unnorm, dtype=float)
    for i in range(mean_metric.shape[2]):  # Loop over slices
        slice_data = mean_metric_unnorm[:, :, i]
        min_val, max_val = np.min(slice_data), np.max(slice_data)
        mean_metric[:, :, i] = (slice_data - min_val) / (max_val - min_val) if max_val > min_val else 0
    # threshold can be sequence to apply to each slice
    if np.isscalar(thres):
        mask = (mean_metric > thres).astype(np.uint8)
    else:
        thres = np.array(thres).reshape(1, 1, -1)  # For broadcasting over slices
        if thres.shape[2] != mean_metric.shape[2]:
            raise ValueError(f"Length of threshold vector ({thres.shape[2]}) must match number of slices ({mean_metric.shape[2]})")
        mask = (mean_metric > thres).astype(np.uint8)
    return mask


def get_signal(func_data_window, mask):
    s = np.zeros((func_data_window.shape[-1], func_data_window.shape[-2]))
    for i in range(func_data_window.shape[-2]):
        masked_data = func_data_window[:, :, i, :] * mask[:, :, i][:, :, np.newaxis]
        s[:, i] = np.mean(masked_data, axis=(0, 1)) 
    return s