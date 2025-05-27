# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import skew
from scipy.optimize import curve_fit


def compute_amp(func_data_window):
    tmax = np.max(func_data_window, axis=-1) / np.min(func_data_window, axis=-1)
    max_values = tmax.max(axis=(0, 1), keepdims=True)
    tmax_norm = tmax / max_values
    return tmax_norm


def compute_skew(func_data_window):
    sk_arr = np.zeros((func_data_window.shape[0], func_data_window.shape[1], func_data_window.shape[2]))
    for islice in range(func_data_window.shape[2]):
        sk = skew(func_data_window[:, :, islice, :], axis=-1)
        sk_arr[:, :, islice] = np.abs(sk)
    max_values = sk_arr.max(axis=(0, 1), keepdims=True)
    sk_norm = sk_arr / max_values
    return sk_norm


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
    fact_div = DR.max(axis=(0, 1), keepdims=True)
    fact_div[0][0][-1] = 1.0 # avoid divide by zero in the last slice
    dr_norm = DR / fact_div
    dr_norm[dr_norm < 0] = 0.0
    return dr_norm


def compute_sbref(sbref_data_window):
    if len(sbref_data_window.shape) > 3:
        sbref_data_window = sbref_data_window[:, :, :, 0]
    sbref_norm = sbref_data_window / sbref_data_window.max(axis=(0, 1), keepdims=True)
    return sbref_norm