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


def main():
    
    parser = argparse.ArgumentParser(description='Automatic 4th ventricle CSF flow segmentation')
    parser.add_argument('--func', type=str, help='path to functional')
    parser.add_argument('--sbref', type=str, help='path to sbref')
    parser.add_argument('--aseg', type=str, help='path to freesurfer aseg')
    parser.add_argument('--reg', type=str, help='path to registration matrix)')
    parser.add_argument('--outdir', default='', type=str, help='path to output directory')
    parser.add_argument('--span', default=5, type=int, help='length of window')
    parser.add_argument('--nslice', default=5, type=int, help='number of slices in window')
    parser.add_argument('--w_amp', default=0.25, type=float, help='weight of amplitude metric')
    parser.add_argument('--w_sk', default=0.25, type=float, help='weight of skew metric')
    parser.add_argument('--w_dr', default=0.25, type=float, help='weight of decay rate metric')
    parser.add_argument('--w_sbref', default=0.25, type=float, help='weight of sbref metric')
    parser.add_argument('--thres', default=0.50, type=float, help='threshold for mask definition')
    parser.add_argument('--method', default='simple', type=str, help='algorithm to use')
    
    args = parser.parse_args()
    
    # args.func = '/om2/group/lewislab/aging/ag111/ses-02-night/mri/stcfsl/run01_rest_stc.nii.gz'
    # args.sbref = '/om2/group/lewislab/aging/ag111/ses-02-night/mri/sbref/run01_SBRef.nii.gz'
    # args.aseg = '/om2/group/lewislab/aging/ag111/ses-02-night/mri/fs_recon_biascorr/mri/aseg.mgz'
    # args.reg = '/om2/group/lewislab/aging/ag111/ses-02-night/mri/registration/reg01toref.dat'
    # args.outdir = '/om/user/bashen/repositories/autocsfmask/output/test'

    # args.func = '/om2/group/lewislab/aging/ag152/ses-01-day/mri/stcfsl/run06_rest_stc.nii.gz'
    # args.sbref = '/om2/group/lewislab/aging/ag152/ses-01-day/mri/sbref/run06_rest_SBRef.nii.gz'
    # args.aseg = '/om2/group/lewislab/aging/ag152/ses-01-day/mri/fs_recon_biascorr/mri/aseg.mgz'
    # args.reg = '/om2/group/lewislab/aging/ag152/ses-01-day/mri/registration/reg06toref.dat'
    # args.outdir = '/om/user/bashen/repositories/autocsfmask/output/test'
    # args.method = 'simple'
    
    # args.func = '/om2/group/lewislab/aging/ag152/ses-01-day/mri/stcfsl/run03_breath_stc.nii.gz'
    # args.sbref = '/om2/group/lewislab/aging/ag152/ses-01-day/mri/sbref/run03_breath_SBRef.nii.gz'
    # args.aseg = '/om2/group/lewislab/aging/ag152/ses-01-day/mri/fs_recon_biascorr/mri/aseg.mgz'
    # args.reg = '/om2/group/lewislab/aging/ag152/ses-01-day/mri/registration/reg03toref.dat'
    # args.outdir = '/om/user/bashen/repositories/autocsfmask/output/test'
    # args.method = 'simple'
    
    # args.func = '/om2/group/lewislab/aging/ag154/ses-01-day/mri/stcfsl/run03_breath_stc.nii.gz'
    # args.sbref = '/om2/group/lewislab/aging/ag154/ses-01-day/mri/sbref/run03_breath_SBRef.nii.gz'
    # args.aseg = '/om2/group/lewislab/aging/ag154/ses-01-day/mri/fs_recon_biascorr/mri/aseg.mgz'
    # args.reg = '/om2/group/lewislab/aging/ag154/ses-01-day/mri/registration/reg03toref.dat'
    # args.outdir = '/om/user/bashen/repositories/autocsfmask/output/test'
    # args.method = 'simple'
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load mri data
    print("Loading data...")
    func_data, sbref_data, aseg_data, regmat, func_affine, func_header = load_data(args.func, args.sbref, args.aseg, args.reg)
    print(f"Functional data shape: {func_data.shape}")
    print(f"SBRef shape: {sbref_data.shape}")
    print(f"Aseg shape: {aseg_data.shape}")
    print("Registration Matrix (regmat):")
    print(np.array2string(regmat, formatter={'float_kind':lambda x: f"{x:7.4f}"}))
    
    # window data around 4th ventricle for each functional slice
    print("Extracting windowed data...")
    func_data_window, sbref_data_window, window_coords, aseg_windows, centroids =  \
        window_data(func_data, sbref_data, aseg_data, regmat, args.span, args.nslice)
    print(f"Windowed data shape: {func_data_window.shape}")
    
    # plot windows
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

    # compute metrics
    print("Computing metrics...")
    tmax_norm = compute_amp_metric(func_data_window)
    print("Amplitude metric computed.")
    sk_norm = compute_skew_metric(func_data_window)
    print("Skewness metric computed.")
    dr_norm = compute_decay_metric(func_data_window)
    print("Decay rate metric computed.")
    sbref_norm = compute_sbref_metric(sbref_data_window)
    print("SBRef metric computed.")
    
    print("Generating mask using method:", args.method)
    
    # store metrics and weights in list
    metrics = [tmax_norm, sk_norm, dr_norm, sbref_norm]
    weights = [args.w_amp, args.w_sk, args.w_dr, args.w_sbref]
    
    if args.method == 'simple':
        mask = get_mask_simple(metrics, weights, args.thres)
    elif args.method == 'optim_all':
        mask = get_mask_optim_all(metrics, func_data_window)
    elif args.method == 'optim_thres':
        mask = get_mask_optim_thres(metrics, weights)
    elif args.method == 'hybrid':
        mask = get_mask_hybrid(metrics, func_data_window)    
    
    # plot all metrics
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
    # Normalize mean metric slice-wise
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
    
    # plot signal
    s = np.zeros((func_data_window.shape[-1], args.nslice))
    for islice in range(args.nslice):    
        slice_mask = mask[:, :, islice].astype(bool)
        slice_data = func_data_window[:, :, islice, :][slice_mask, :]
        mean_ts = slice_data.mean(axis=0)
        s[:, islice] = mean_ts
    
    sraw = np.array(s)
    sproc = scale_epi(s)
    fig_signal, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    ax1.plot(sraw)
    ax2.plot(sproc)
    plt.tight_layout()
    plt.show()
    
    output_path = os.path.join(args.outdir, 'signal.txt')
    np.savetxt(output_path, sraw)
    
    output_path = os.path.join(args.outdir, 'signal.png')
    fig_signal.savefig(output_path, format='png')
    
    print("Computing deattenuation matrix...")
    
    da = compute_deattenuation_matrix(sproc)
    num_upper = (da.shape[1] * (da.shape[2]-1))/2 # exlcuding diagonal
    score = []
    for itime in range(da.shape[0]):
        score.append((np.triu(da) > 0)[itime].sum() / num_upper)
    score = np.array(score)
    
    # da_sum = np.expand_dims((np.triu(da) < 0).sum(), axis=0)
    
    output_path = os.path.join(args.outdir, 'score.txt')
    np.savetxt(output_path, np.expand_dims(score.sum()/score.size, axis=0))
    
    # Create a heatmap using the 'RdBu' colormap
    fig, ax = plt.subplots()
    ax.imshow(da.mean(axis=0), cmap='RdBu')
    plt.show()
    output_path = os.path.join(args.outdir, 'diff_map.png')
    fig.savefig(output_path, format='png')
    
    # define and save full mask that can be used by fmri software
    pred_full_mask = map_window_to_full(mask, window_coords, full_shape=func_data.shape[:3])
    Z = pred_full_mask.shape[2]
    pred_full_mask = pred_full_mask * np.arange(1, Z + 1)[np.newaxis, np.newaxis, :]
    new_img = nib.Nifti1Image(pred_full_mask, func_affine, header=func_header)
    output_path = os.path.join(args.outdir, 'csf_mask.nii.gz')
    nib.save(new_img, output_path)
    
    # # plot each mask slice
    # fig, axes = plt.subplots(nrows=1, ncols=5)
    # for i, ax in enumerate(axes):
    #     ax.imshow(pred_full_mask[:, :, i])
    # plt.show()
    
    print("Segmentation complete. All outputs saved to:", args.outdir)
    
    
def load_data(func_path, sbref_path, aseg_path, reg_path):
        
    func_nifti = nib.load(func_path)
    sbref_nifti = nib.load(sbref_path)
    aseg_nifti = nib.load(aseg_path)
    func_data = func_nifti.get_fdata()
    sbref_data = sbref_nifti.get_fdata()
    aseg_data = aseg_nifti.get_fdata()
    
    regmat = np.loadtxt(reg_path, skiprows=4, max_rows=4) 
    
    affine = func_nifti.affine
    header = func_nifti.header
    
    return func_data, sbref_data, aseg_data, regmat, affine, header


def window_data(func_data, sbref_data, aseg_data, regmat, span, nslice):

    aseg_data = aseg_data.copy()
    aseg_data[aseg_data != 15] = 0

    # Construct VOX2RAS matrices
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
        window_coords_func = np.array([
            [x_start, y_start, islice, 1],
            [x_end, y_end, islice, 1]
        ]).T
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


def compute_amp_metric(func_data_window):
    tmax = np.max(func_data_window, axis=-1) / np.min(func_data_window, axis=-1)
    max_values = tmax.max(axis=(0, 1), keepdims=True)
    tmax_norm = tmax / max_values
    return tmax_norm


def compute_skew_metric(func_data_window):
    sk_arr = np.zeros((func_data_window.shape[0], func_data_window.shape[1], func_data_window.shape[2]))
    for islice in range(func_data_window.shape[2]):
        sk = skew(func_data_window[:, :, islice, :], axis=-1)
        sk_arr[:, :, islice] = np.abs(sk)
    max_values = sk_arr.max(axis=(0, 1), keepdims=True)
    sk_norm = sk_arr / max_values
    return sk_norm


def compute_decay_metric(func_data_window):
    
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


def compute_sbref_metric(sbref_data_window):
    
    if len(sbref_data_window.shape) > 3:
        sbref_data_window = sbref_data_window[:, :, :, 0]
    
    sbref_norm = sbref_data_window / sbref_data_window.max(axis=(0, 1), keepdims=True)
    return sbref_norm


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
    
    # Apply threshold and generate binary mask
    mask = (mean_metric > thres).astype(np.uint8)
    
    return mask


def compute_deattenuation_matrix(s):
    
    ntime = s.shape[0]
    nslice = s.shape[1]
    da = np.zeros((ntime, nslice, nslice))
    for itime in range(ntime):
        
        sslices = s[itime, :]
        for i in range(nslice):
            for j in range(nslice):
                da[itime, i, j] = sslices[i] - sslices[j]
    
    return da


def get_signal(func_data_window, mask):
    s = np.zeros((func_data_window.shape[-1], func_data_window.shape[-2]))
    for i in range(func_data_window.shape[-2]):
        masked_data = func_data_window[:, :, i, :] * mask[:, :, i][:, :, np.newaxis]
        s[:, i] = np.mean(masked_data, axis=(0, 1)) 
    return s


def get_mask_simple(metrics, weights, thres):
    # APPROACH 0 ===============================================

    assert np.sum(weights) == 1.0, 'Warning: Weights must sum to 1.0'
    
    # define mask based on metrics and weights
    mask = get_mask(metrics, weights, thres=thres)
    
    return mask
    # ==========================================================


def get_mask_optim_all(metrics, func_data_window):    
    # APPROACH 1 - vary everything =============================

    from scipy.optimize import minimize

    def objective(params):
        # Split weights and threshold from params
        weights = params[:4]
        thres = params[4]
        
        # Ensure weights sum to 1 (although SLSQP will enforce this)
        weights = weights / np.sum(weights)
        
        # Compute the mask based on metrics and threshold
        mask = get_mask(metrics, weights, thres=thres)
        
        # Compute da_sum
        s = get_signal(func_data_window, mask)
        sproc = scale_epi(s)
        da = compute_deattenuation_matrix(sproc)
        da_sum = np.float64((np.triu(da) < 0).sum())
        
        return da_sum  # We want to minimize this

    # Constraint: sum of weights must be 1
    def weight_constraint(params):
        return np.sum(params[:4]) - 1

    # Bounds: weights between 0 and 1, thres in a reasonable range
    bounds = [(0, 1)] * 4 + [(0.01, 1)]  # Assuming 0.01 ≤ thres ≤ 1

    results = []
    local_mins = []
    for _ in range(10):
        initial_guess = [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()]
        
        # Optimization
        result = minimize(objective, initial_guess, method='SLSQP',
                        constraints={'type': 'eq', 'fun': weight_constraint},
                        bounds=bounds)

        # Optimal weights and threshold
        optimal_weights = result.x[:4]
        optimal_thres = result.x[4]

        print("Optimal Weights:", optimal_weights)
        print("Optimal Threshold:", optimal_thres)
        print("Minimum da_sum:", result.fun)
        results.append(result)
        local_mins.append(result.fun)
        
    best_ind = np.argmin(local_mins)
    optimal_weights = results[best_ind].x[:4]
    optimal_thres = results[best_ind].x[4]
    
    mask = get_mask(metrics, optimal_weights, thres=optimal_thres)
    return mask
    # ==========================================================


def get_mask_optim_thres(metrics, weights):
    # APPROACH 2 - vary threshold ==============================
    
    from itertools import combinations
    
    assert np.sum(weights) == 1.0, 'Warning: Weights must sum to 1.0'
    
    def dice_score(mask1, mask2):
        """Compute the Dice similarity coefficient between two binary masks."""
        intersection = np.sum(mask1 & mask2)
        return 2 * intersection / (np.sum(mask1) + np.sum(mask2))
    
    thresholds = np.arange(0.1, 1, 0.1)
    dcsums = []
    for threshold in thresholds:

        # Normalize and threshold mask each array
        masked_metrics = [(metric > threshold).astype(int) for metric in metrics]
        masked_metrics.append(get_mask(metrics, weights, thres=threshold))

        # Compute pairwise Dice scores
        num_metrics = len(masked_metrics)
        dice_matrix = np.zeros((num_metrics, num_metrics))

        # Fill the upper triangle of the matrix with pairwise Dice scores
        for i, j in combinations(range(num_metrics), 2):
            dice_matrix[i, j] = dice_matrix[j, i] = dice_score(masked_metrics[i], masked_metrics[j])

        # Print the result
        print(dice_matrix)
        dcsums.append(dice_matrix.sum())
    
    # define mask based on metrics and weights
    best_ind = np.argmax(dcsums)
    optimal_thres = thresholds[best_ind]
    mask = get_mask(metrics, weights, thres=optimal_thres)
    return mask
    # ==========================================================


def get_mask_hybrid(metrics, func_data_window):
    # APPROACH 3 - hybrid ======================================
    from scipy.optimize import minimize
    from itertools import combinations

    def dice_score(mask1, mask2):
        intersection = np.sum(mask1 & mask2)
        return 2 * intersection / (np.sum(mask1) + np.sum(mask2))

    def weight_constraint(params):
        return np.sum(params[:4]) - 1

    # Approach 1: Optimize weights with fixed threshold
    def objective(params):
        weights = params[:4]
        thres = params[4]
        weights = weights / np.sum(weights)
        mask = get_mask(metrics, weights, thres=thres)
        s = get_signal(func_data_window, mask)
        sproc = scale_epi(s)
        da = compute_deattenuation_matrix(sproc)
        da_sum = np.float64((np.triu(da) < 0).sum())
        return da_sum  # We want to minimize this
        
    def hybrid_optimization(metrics, max_iter=5):

        final_dcs = []
        for iter in range(max_iter):
            print(f"ITER = {iter}")
            
            bounds = [(0, 1)] * 4 + [(0.01, 1)]  # Assuming 0.01 ≤ thres ≤ 1
            results = []
            local_mins = []

            for _ in range(5):
                initial_guess = np.random.rand(5)
                
                if iter > 1:
                    initial_guess[-1] = thres
                
                result = minimize(objective, initial_guess, method='SLSQP',
                                constraints={'type': 'eq', 'fun': weight_constraint},
                                bounds=bounds)
                results.append(result)
                local_mins.append(result.fun)
                
            # Optimal weights and threshold
            best_ind = np.argmin(local_mins)
            optimal_weights = results[best_ind].x[:4]
            optimal_thres = results[best_ind].x[4]
            print("Optimal Weights:", optimal_weights)
            print("Optimal Threshold:", optimal_thres)
            print("Minimum da_sum:", result.fun)
            mask_1 = get_mask(metrics, optimal_weights, thres=optimal_thres)
            
            # define thresholds around optimal
            thresholds = np.linspace(0.8*optimal_thres, 1.2*optimal_thres, 10)
            thresholds[thresholds > 1.0] = 1.0
            thresholds[thresholds < 0.0] = 0.0
            
            print(f"thres between {0.8*optimal_thres} and {1.2*optimal_thres}") 
            
            # Approach 2: Optimize threshold with fixed weights
            dcsums = []
            for threshold in thresholds:
                # Normalize and threshold mask each array
                masked_metrics = [(metric > threshold).astype(int) for metric in metrics]
                masked_metrics.append(get_mask(metrics, optimal_weights, thres=threshold))

                # Compute pairwise Dice scores
                num_metrics = len(masked_metrics)
                dice_matrix = np.zeros((num_metrics, num_metrics))
                
                for i, j in combinations(range(len(metrics)), 2):
                    dice_matrix[i, j] = dice_matrix[j, i] = dice_score(masked_metrics[i], masked_metrics[j])
                dcsums.append(dice_matrix.sum())
                print(f"dcsum = {dice_matrix.sum()}")
            
            
            best_ind = np.argmax(dcsums)
            thres = thresholds[best_ind]  # Update threshold
            mask_2 = get_mask(metrics, optimal_weights, thres=thres)
            
            # compute mask dice score
            final_mask_dice = dice_score(mask_1, mask_2)
            final_dcs.append(final_mask_dice)
        
        print(f"final_dcs = {final_dcs}")
        return mask_2

    mask = hybrid_optimization(metrics)
    return mask
# ==========================================================

    
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
        s -= mean_pct_portion(s, 5)
    slice1_maximum = mean_pct_portion(s[:, [0]], 5, fromtop=True)
    s /= slice1_maximum
    return s


if __name__ == "__main__":
    main()