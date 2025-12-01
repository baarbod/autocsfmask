# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import skew


def normalize_slicewise(voxel_list, lower_percentile=2, upper_percentile=98):
    for i, slice_data in enumerate(voxel_list):
        p_low = np.percentile(slice_data, lower_percentile)
        p_high = np.percentile(slice_data, upper_percentile)
        voxel_list[i] = (slice_data - p_low)/p_high
    return voxel_list


def compute_std(voxel_list):
    voxel_stds = []
    for i, slice_data in enumerate(voxel_list):
        voxel_stds.append(slice_data.std(axis=-1)) 
    return normalize_slicewise(voxel_stds)


def compute_mean(voxel_list):
    voxel_means = []
    for i, slice_data in enumerate(voxel_list):
        voxel_means.append(slice_data.mean(axis=-1)) 
    return normalize_slicewise(voxel_means)


def compute_bottom_mean(voxel_list, frac=0.05):
    voxel_means = []
    for slice_data in voxel_list:
        # Flatten along last axis, keep per-voxel timecourses
        values = slice_data
        cutoff = np.quantile(values, frac, axis=-1, keepdims=True)
        bottom_mask = values <= cutoff
        bottom_mean = (values * bottom_mask).sum(axis=-1) / bottom_mask.sum(axis=-1)
        voxel_means.append(bottom_mean)
    return normalize_slicewise(voxel_means)


def compute_skew(voxel_list):
    voxel_skews = []
    for i, slice_data in enumerate(voxel_list):
        voxel_skews.append(np.abs(skew(slice_data, axis=-1)))
    return normalize_slicewise(voxel_skews)    


def compute_sbref(voxel_list):
    return normalize_slicewise(voxel_list)