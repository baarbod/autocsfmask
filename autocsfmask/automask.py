# -*- coding: utf-8 -*-

import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import argparse
from scipy.stats import skew
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass
import matplotlib.patches as patches
from scipy.optimize import differential_evolution

import autocsfmask.utils as utils
import autocsfmask.metrics as metrics
import autocsfmask.masking as masking


def main():
    
    parser = argparse.ArgumentParser(description='Automatic 4th ventricle CSF flow segmentation')
    parser.add_argument('--func', type=str, help='path to functional')
    parser.add_argument('--sbref', type=str, help='path to sbref')
    parser.add_argument('--aseg', type=str, help='path to freesurfer aseg')
    parser.add_argument('--reg', type=str, help='path to registration matrix)')
    parser.add_argument('--outdir', default='', type=str, help='path to output directory')
    parser.add_argument('--span', default=4, type=int, help='length of window')
    parser.add_argument('--nslice', default=5, type=int, help='number of slices in window')
    parser.add_argument('--w_amp', default=0.25, type=float, help='weight of amplitude metric')
    parser.add_argument('--w_sk', default=0.25, type=float, help='weight of skew metric')
    parser.add_argument('--w_dr', default=0.25, type=float, help='weight of decay rate metric')
    parser.add_argument('--w_sbref', default=0.25, type=float, help='weight of sbref metric')
    parser.add_argument('--thres', default=0.50, type=float, help='threshold for mask definition')
    parser.add_argument('--method', default='simple', type=str, help='algorithm to use')
    
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print("Loading data...")
    func_data, sbref_data, aseg_data, regmat, func_affine, func_header = load_data(args.func, args.sbref, args.aseg, args.reg)
    print(f"Functional data shape: {func_data.shape}")
    print(f"SBRef shape: {sbref_data.shape}")
    print(f"Aseg shape: {aseg_data.shape}")
    print("Registration Matrix (regmat):")
    print(np.array2string(regmat, formatter={'float_kind':lambda x: f"{x:7.4f}"}))
    print("Extracting windowed data...")
    func_data_window, sbref_data_window, window_coords, aseg_windows, centroids =  \
        window_data(func_data, sbref_data, aseg_data, regmat, args.span, args.nslice)
    print(f"Windowed data shape: {func_data_window.shape}")
    
    print("Plotting windowed data...")
    fig_windows, axes = plt.subplots(nrows=args.nslice, ncols=3, figsize=(4.5, args.nslice * 1.5))
    lower_ind = np.array(window_coords).min(axis=0)
    upper_ind = np.array(window_coords).max(axis=0)
    xmin = lower_ind[0] - int(0.5*args.span)
    xmax = upper_ind[1] + int(0.5*args.span)
    ymin = lower_ind[2] - int(0.5*args.span) 
    ymax = upper_ind[3] + int(0.5*args.span)
    for islice in range(args.nslice):
        (fc_y, fc_x), (ac_y, ac_x) = centroids[islice]
        axes[islice, 0].imshow(aseg_windows[islice], cmap='gray')
        axes[islice, 0].scatter(ac_y, ac_x, color='red')
        axes[islice, 1].imshow(sbref_data_window[:, :, islice], cmap='gray')
        axes[islice, 1].scatter(fc_y, fc_x, color='red')
        cx = np.array(window_coords)[islice, :2].mean() - xmin
        cy = np.array(window_coords)[islice, 2:4].mean() - ymin
        axes[islice, 2].imshow(sbref_data[xmin:xmax, ymin:ymax, islice])
        axes[islice, 2].scatter(cy, cx, color='red')
        rect = patches.Rectangle((cy-args.span, cx-args.span), 2*args.span, 2*args.span, linewidth=0.5, edgecolor='red', facecolor='none')
        axes[islice, 2].add_patch(rect)
        for j in range(3):
            axes[islice, j].set_axis_off()
    column_titles = ['Aseg centered', 'SBRef centered', 'SBRef de-centered']
    for col, title in enumerate(column_titles):
        axes[0][col].set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()
    output_path = os.path.join(args.outdir, 'anat_and_func_windows.png')
    fig_windows.savefig(output_path, format='png')

    print("Computing metrics...")
    tmax_norm = metrics.compute_amp(func_data_window)
    sk_norm = metrics.compute_skew(func_data_window)
    dr_norm = metrics.compute_decay(func_data_window)
    sbref_norm = metrics.compute_sbref(sbref_data_window)
    print("Generating mask using method:", args.method)
    metrics = [tmax_norm, sk_norm, dr_norm, sbref_norm]
    weights = [args.w_amp, args.w_sk, args.w_dr, args.w_sbref]
    if args.method == 'simple':
        mask = masking.get_mask_simple(metrics, weights, args.thres)
    elif args.method == 'optim_all':
        mask = masking.get_mask_optim_all(metrics, func_data_window)
    elif args.method == 'hybrid':
        mask = masking.get_mask_hybrid(metrics, func_data_window)    
    
    print("Plotting metrics and saving to:", args.outdir)
    row_titles = ['Amp', 'Skew', 'Decay', 'SBRef', 'Mean', 'Mask']
    fig_metric, axes = plt.subplots(nrows=len(row_titles), ncols=args.nslice, figsize=(6, 7))
    def plot_metric_on_row(metric, axes, row=0):
        for islice, ax in enumerate(axes[row, :]):
            ax.imshow(metric[:, :, islice])
            ax.set_axis_off()
        fig_metric.text(0.5, 1.0 - (row - 0.05) / len(row_titles), row_titles[row], ha='center', va='top', fontsize=14)
    weights = np.array(weights, dtype=float).reshape(-1, 1, 1, 1)  # Reshape for broadcasting
    stacked_arrays = np.stack(metrics, axis=0)
    mean_metric_unnorm = np.sum(stacked_arrays * weights, axis=0)
    mean_metric = np.zeros_like(mean_metric_unnorm, dtype=float)
    for i in range(mean_metric.shape[2]):  # Loop over slices
        slice_data = mean_metric_unnorm[:, :, i]
        min_val, max_val = np.min(slice_data), np.max(slice_data)
        mean_metric[:, :, i] = (slice_data - min_val) / (max_val - min_val) if max_val > min_val else 0
    plot_metric_on_row(tmax_norm, axes, row=0)
    plot_metric_on_row(sk_norm, axes, row=1)
    plot_metric_on_row(dr_norm, axes, row=2)
    plot_metric_on_row(sbref_norm, axes, row=3)
    plot_metric_on_row(mean_metric, axes, row=4)
    plot_metric_on_row(mask, axes, row=5)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.05)
    plt.show()
    output_path = os.path.join(args.outdir, 'voxels.png')
    fig_metric.savefig(output_path, format='png')

    print("Extracting signal from mask...")
    s = np.zeros((func_data_window.shape[-1], args.nslice))
    for islice in range(args.nslice):    
        slice_mask = mask[:, :, islice].astype(bool)
        slice_data = func_data_window[:, :, islice, :][slice_mask, :]
        mean_ts = slice_data.mean(axis=0)
        s[:, islice] = mean_ts
    sraw = np.array(s)
    sproc = utils.scale_epi(s)
    fig_signal, axes = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].plot(sraw)
    axes[0, 1].plot(utils.smooth_timeseries(sraw))
    axes[1, 0].plot(sproc)
    axes[1, 1].plot(utils.smooth_timeseries(sproc))
    plt.tight_layout()
    plt.show()
    output_path = os.path.join(args.outdir, 'signal.txt')
    np.savetxt(output_path, sraw)
    output_path = os.path.join(args.outdir, 'signal.png')
    fig_signal.savefig(output_path, format='png')
    
    print("Computing decay-based metric...")
    da = utils.compute_deattenuation_matrix(sproc)
    da = da[:, :3, :3] # keep only first 3 slices
    num_upper = (da.shape[1] * (da.shape[2] - 1)) / 2
    upper_mask = np.triu(np.ones((da.shape[1], da.shape[2]), dtype=bool), k=1)
    scores = (da[:, upper_mask] > 0).sum(axis=1) / num_upper
    score = np.expand_dims(scores.sum()/scores.size, axis=0)
    output_path = os.path.join(args.outdir, 'score.txt')
    np.savetxt(output_path, score)
    fig, ax = plt.subplots()
    ax.imshow(da.mean(axis=0), cmap='RdBu')
    plt.show()
    output_path = os.path.join(args.outdir, 'diff_map.png')
    fig.savefig(output_path, format='png')
    
    print("Computing correlation-based metric...")
    corr_score = masking.compute_masked_correlation(mask, func_data_window, penalty_scale=0)
    output_path = os.path.join(args.outdir, 'score_corr.txt')
    np.savetxt(output_path, np.expand_dims(corr_score, axis=0))
    
    pred_full_mask = map_window_to_full(mask, window_coords, full_shape=func_data.shape[:3])
    Z = pred_full_mask.shape[2]
    pred_full_mask = pred_full_mask * np.arange(1, Z + 1)[np.newaxis, np.newaxis, :]
    new_img = nib.Nifti1Image(pred_full_mask, func_affine, header=func_header)
    output_path = os.path.join(args.outdir, 'csf_mask.nii.gz')
    nib.save(new_img, output_path)
    
    print("Segmentation complete. All outputs saved to:", args.outdir)
    
    
def load_data(func_path, sbref_path, aseg_path, reg_path):
    func_nifti = nib.load(func_path)
    sbref_nifti = nib.load(sbref_path)
    aseg_nifti = nib.load(aseg_path)
    func_data = func_nifti.get_fdata()
    sbref_data = sbref_nifti.get_fdata()
    if sbref_data.ndim > 3: # sometimes phase image is put in the 4th dimension
        sbref_data = sbref_data[:, :, :, 0]
    aseg_data = aseg_nifti.get_fdata()
    regmat = np.loadtxt(reg_path, skiprows=4, max_rows=4) 
    affine = func_nifti.affine
    header = func_nifti.header
    return func_data, sbref_data, aseg_data, regmat, affine, header


def window_data(func_data, sbref_data, aseg_data, regmat, span, nslice):
    aseg_data = aseg_data.copy()
    aseg_data[aseg_data != 15] = 0
    anatVOX2RAS = np.array([
        [-1.0, 0,    0,  0.5 * aseg_data.shape[0]],
        [0,    0,    1.0, -0.5 * aseg_data.shape[2]],
        [0,   -1.0,  0,  0.5 * aseg_data.shape[1]],
        [0,    0,    0,  1]
    ])
    funcVOX2RAS = np.array([
        [-2.5, 0,    0,  1.25 * func_data.shape[0]],
        [0,    0,    2.5, -1.25 * func_data.shape[2]],
        [0,   -2.5,  0,  1.25 * func_data.shape[1]],
        [0,    0,    0,  1]
    ])
    func_data_window = np.zeros((2*span, 2*span, nslice, func_data.shape[-1]))
    sbref_data_window = np.zeros((2*span, 2*span, nslice))
    aseg_windows = []
    centroids = []
    window_coords = []
    # find initial func coordinates using aseg 3D centroid
    centroid_3d = center_of_mass(aseg_data)
    centroid_3d_anatCRS = np.array([*centroid_3d, 1], ndmin=2)
    init_funcCRS = np.linalg.inv(funcVOX2RAS) @ regmat @ anatVOX2RAS @ centroid_3d_anatCRS.T
    cinit = (int(np.floor(init_funcCRS[0][0])), int(np.floor(init_funcCRS[1][0])))
    for islice in range(nslice):
        # find anat slice for current func slice
        funcCRS = np.array([cinit[0], cinit[1], islice, 1], ndmin=2)
        anatCRS = np.linalg.inv(anatVOX2RAS) @ np.linalg.inv(regmat) @ funcVOX2RAS @ funcCRS.T
        anat_slice_ind = int(np.floor(anatCRS[1][0]))
        # find centroi of anat slice
        aseg_slice = aseg_data[:, anat_slice_ind, :]
        centroid_2d = center_of_mass(aseg_slice)
        # map anat centroid to func centroid
        centroid_x = int(np.floor(centroid_2d[0]))
        centroid_y = int(np.floor(centroid_2d[1]))
        centroid_anatCRS = np.array([centroid_x, anat_slice_ind, centroid_y, 1], ndmin=2)
        centroid_funcCRS = np.linalg.inv(funcVOX2RAS) @ regmat @ anatVOX2RAS @ centroid_anatCRS.T
        centroid_x_func = int(np.floor(centroid_funcCRS[0][0]))
        centroid_y_func = int(np.floor(centroid_funcCRS[1][0]))
        cinit = (centroid_x_func, centroid_y_func)
        # define index for func window
        x_start = centroid_x_func - span
        x_end = centroid_x_func + span
        y_start = centroid_y_func - span
        y_end = centroid_y_func + span
        window_coords.append((x_start, x_end, y_start, y_end, islice))
        # window func and sbref data
        func_data_window[:, :, islice, :] = func_data[x_start:x_end, y_start:y_end, islice, :]
        sbref_data_window[:, :, islice] = sbref_data[x_start:x_end, y_start:y_end, islice]
        # define window for anat data, using corners of func window
        window_coords_func = np.array([[x_start, y_start, islice, 1], [x_end, y_end, islice, 1]]).T
        window_coords_anat = np.linalg.inv(anatVOX2RAS) @ np.linalg.inv(regmat) @ funcVOX2RAS @ window_coords_func
        window_coords_anat = np.floor(window_coords_anat[:3, :]).astype(int)
        if window_coords_anat[0, 0] < window_coords_anat[0, 1]:
            x_start_anat, x_end_anat = window_coords_anat[0, 0], window_coords_anat[0, 1]
        else:
            x_end_anat, x_start_anat = window_coords_anat[0, 0], window_coords_anat[0, 1]
        if window_coords_anat[2, 0] < window_coords_anat[2, 1]:
            y_start_anat, y_end_anat = window_coords_anat[2, 0], window_coords_anat[2, 1]
        else:
            y_end_anat, y_start_anat = window_coords_anat[2, 0], window_coords_anat[2, 1]
        # window anat data
        aseg_window = aseg_data[x_start_anat:x_end_anat, anat_slice_ind, y_start_anat:y_end_anat]
        aseg_windows.append(aseg_window)
        # save anat centroid
        ac_x, ac_y = center_of_mass(aseg_window)
        fc_x = centroid_x_func - x_start
        fc_y = centroid_y_func - y_start
        centroids.append(((fc_y, fc_x), (ac_y, ac_x)))  # func, aseg
    return func_data_window, sbref_data_window, window_coords, aseg_windows, centroids


def map_window_to_full(window_mask, window_coords, full_shape):
    """
    window_mask: shape (2*span, 2*span, nslice)
    window_coords: list of (x_start, x_end, y_start, y_end, z)
    full_shape: shape of full volume, e.g. (96, 96, 40)

    Returns:
        full_mask: shape full_shape with windowed mask placed at correct coords
    """
    full_mask = np.zeros(full_shape, dtype=window_mask.dtype)
    for islice, (x_start, x_end, y_start, y_end, z) in enumerate(window_coords):
        full_mask[x_start:x_end, y_start:y_end, z] = window_mask[:, :, islice]
    return full_mask


if __name__ == "__main__":
    main()