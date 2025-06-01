# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import skew
from scipy.optimize import curve_fit


def normalize_slicewise(arr, lower_percentile=2, upper_percentile=98):
    p_low = np.percentile(arr, lower_percentile, axis=(0, 1), keepdims=True)
    p_high = np.percentile(arr, upper_percentile, axis=(0, 1), keepdims=True)
    denom = p_high - p_low
    denom[denom == 0] = 1.0  # Avoid divide by zero
    arr_clipped = np.clip(arr, p_low, p_high)
    return (arr_clipped - p_low) / denom


def compute_std(func_data_window):
    return normalize_slicewise(func_data_window.std(axis=-1))


def compute_mean(func_data_window):
    return normalize_slicewise(func_data_window.mean(axis=-1))


def compute_skew(func_data_window):
    sk_arr = np.zeros((func_data_window.shape[0], func_data_window.shape[1], func_data_window.shape[2]))
    for islice in range(func_data_window.shape[2]):
        sk = skew(func_data_window[:, :, islice, :], axis=-1)
        sk_arr[:, :, islice] = np.abs(sk)
    return normalize_slicewise(sk_arr)


def compute_decay(func_data_window):
    def exp_decay(x, b):
        return np.exp(-b * x)
    padded_data = np.pad(func_data_window, ((1, 1), (1, 1), (0, 0), (0, 0)), mode='edge')
    DR = np.zeros((func_data_window.shape[0], func_data_window.shape[1], func_data_window.shape[2]))
    for islice in range(func_data_window.shape[2] - 1): # skip last slice - decay rate needs multiple subsequent slices
        for i in range(func_data_window.shape[0]):
            for j in range(func_data_window.shape[1]):
                smax = np.max(padded_data[i:i+2, j:j+2, islice:, :], axis=-1)
                smax = smax.mean(axis=(0, 1), keepdims=True).flatten()
                signal_scaled = smax / smax[0]
                slice_indices = np.arange(func_data_window.shape[2] - islice, dtype=np.float32)  # Indices for the slices  
                popt, _ =  curve_fit(exp_decay, slice_indices, signal_scaled)
                DR[i, j, islice] = popt[0]
    dr_norm = normalize_slicewise(DR)
    return dr_norm


def compute_sbref(sbref_data_window):
    if len(sbref_data_window.shape) > 3:
        sbref_data_window = sbref_data_window[:, :, :, 0]
    return normalize_slicewise(sbref_data_window)