# -*- coding: utf-8 -*-

import os
import json
import logging
import argparse
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import autocsfmask.utils as utils
import autocsfmask.metrics as metrics
import autocsfmask.masking as masking


def setup_logging(outdir):
    log_file = os.path.join(outdir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Automatic 4th ventricle CSF flow segmentation')
    parser.add_argument('--func', required=True, type=str, help='Path to functional data')
    parser.add_argument('--sbref', required=True, type=str, help='Path to sbref')
    parser.add_argument('--synthseg', required=True, type=str, help='Path to synthseg dilated mask')
    parser.add_argument('--outdir', default='outputs', type=str, help='Output directory')
    parser.add_argument('--nslice', default=5, type=int, help='Number of slices in window')
    # parser.add_argument('--metrics', default=['sbref'], nargs='+', type=str, help='Metrics to use')
    parser.add_argument('--metrics', default=['amp1', 'amp2', 'amp3', 'skew', 'sbref'], nargs='+', type=str, help='Metrics to use')
    parser.add_argument('--weights', default=4*[0.25], nargs='+', type=float, help='Metric weights for simple method')
    parser.add_argument('--thres', default=0.50, type=float, help='Threshold for mask definition')
    parser.add_argument('--method', default='optim', type=str, help='Algorithm to use (simple or optim)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(args.outdir)

    logging.info("Loading synthseg-masked data...")
    func_voxels, sbref_voxels, coords, func_affine, func_header, func_data, sbref_data = load_data(
        args.func, args.sbref, args.synthseg
    )

    logging.info("Computing metrics: %s", args.metrics)
    metrics_list, weights = get_metrics_and_weights(func_voxels, sbref_voxels, args.metrics, args.weights, args.method)

    logging.info("Generating mask using method: %s", args.method)
    mask, weights, thres, best_score = generate_mask(metrics_list, func_voxels, args.method, args.thres, weights, coords, func_data.shape[:3], N=1)

    # Convert mask list to full volume
    mask_vol = voxel_list_to_volume(mask, coords, func_data.shape[:3])

    print(f"func shape: {func_data.shape}")
    print(f"sbref shape: {sbref_data.shape}")
    print(f"mask shape: {mask_vol.shape}")
    
    # Plot overlay
    fig, _ = plot_mask_overlay(func_data, sbref_data, mask_vol, np.load(args.synthseg))
    fig.savefig(os.path.join(args.outdir, "mask_overlay.png"))

    # Extract signal and plot
    logging.info("Extracting signal from mask...")
    s = utils.get_signal(func_voxels, mask)
    np.savetxt(os.path.join(args.outdir, "signal.txt"), s)
    fig_signal, _ = plot_signal(s)
    fig_signal.savefig(os.path.join(args.outdir, "signal.png"))

    # Save mask volume
    mask_path = os.path.join(args.outdir, "mask.nii.gz")
    nib.save(nib.Nifti1Image(mask_vol, func_affine, header=func_header), mask_path)
    logging.info("Mask saved to: %s", mask_path)

    # Save optimization parameters if applicable
    if args.method == 'optim':
        optimization_results = {"weights": weights.tolist(), "thresholds": thres.tolist(), "best_score": best_score}
        with open(os.path.join(args.outdir, "optimal_params.json"), "w") as f:
            json.dump(optimization_results, f, indent=4)
        logging.info("Optimization parameters saved.")

    logging.info("Segmentation complete. All outputs saved to: %s", args.outdir)


def get_metrics_and_weights(func_voxels, sbref_voxels, mlist, wlist, method):
    weights = wlist.copy()
    if method == 'optim':
        weights = [0] * len(mlist)

    metrics_list = []
    for m in mlist:
        if m == 'amp1':
            metrics_list.append(metrics.compute_mean(func_voxels))
        elif m == 'amp2':
            metrics_list.append(metrics.compute_std(func_voxels))
        elif m == 'amp3':
            metrics_list.append(metrics.compute_bottom_mean(func_voxels))
        elif m == 'skew':
            metrics_list.append(metrics.compute_skew(func_voxels))
        elif m == 'sbref':
            metrics_list.append(metrics.compute_sbref(sbref_voxels))
        else:
            raise ValueError(f"Unknown metric: {m}")
    return metrics_list, weights


def generate_mask(metrics_list, func_voxels, method, thres, weights, coords, vol_shape, N=1):
    if method == 'simple':
        mask = masking.get_mask_simple(metrics_list, weights, thres)
        return mask, weights, thres, None

    elif method == 'optim':
        best_score = np.inf
        best_mask, best_weights, best_thres = None, None, None
        for i in range(N):
            logging.info("Optimization run %d/%d", i+1, N)
            mask, w, t, score = masking.get_mask_optim(metrics_list, func_voxels, coords, vol_shape)
            if score < best_score:
                best_score = score
                best_mask, best_weights, best_thres = mask, w, t
                logging.info("New best score: %.6f", best_score)
        logging.info("Best overall score: %.6f", best_score)
        return best_mask, best_weights, best_thres, best_score

    else:
        raise ValueError(f"Unknown method: {method}")


def load_data(func_path, sbref_path, synthseg_path):
    for path in [func_path, sbref_path, synthseg_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input file not found: {path}")

    func_nifti = nib.load(func_path)
    sbref_nifti = nib.load(sbref_path)
    func_data = func_nifti.get_fdata()
    sbref_data = sbref_nifti.get_fdata()
    synthseg_data = np.load(synthseg_path)

    # Heuristic to skip phase images if necessary
    if sbref_data.ndim > 3:
        sbref_data = sbref_data[:, :, :, 0] if sbref_data[:, :, :, 0].mean() > sbref_data[:, :, :, 1].mean() else sbref_data[:, :, :, 1]

    func_voxels, sbref_voxels, voxel_coords = [], [], []
    for islice in range(synthseg_data.shape[2]):
        slice_mask = synthseg_data[:, :, islice].astype(bool)
        coords = np.array(np.nonzero(slice_mask)).T
        coords = np.hstack([coords, np.full((coords.shape[0], 1), islice)])
        func_voxels.append(func_data[:, :, islice, :][slice_mask, :])
        sbref_voxels.append(sbref_data[:, :, islice][slice_mask])
        voxel_coords.append(coords)

    return func_voxels, sbref_voxels, voxel_coords, func_nifti.affine, func_nifti.header, func_data, sbref_data


def plot_metrics(nslice, metrics_list, weights, mask, metric_row_titles):
    row_titles = metric_row_titles + ['mean', 'mask']
    fig_metric, axes = plt.subplots(nrows=len(row_titles), ncols=nslice, figsize=(6, 7))
    def plot_metric_on_row(metric, axes, row=0):
        for islice, ax in enumerate(axes[row, :]):
            ax.imshow(metric[:, :, islice])
            ax.set_axis_off()
        fig_metric.text(0.5, 1.0-row / len(row_titles), row_titles[row], ha='center', va='top', fontsize=14)
    weights = np.array(weights, dtype=float).reshape(-1, 1, 1, 1)  # Reshape for broadcasting
    stacked_arrays = np.stack(metrics_list, axis=0)
    mean_metric_unnorm = np.sum(stacked_arrays * weights, axis=0)
    mean_metric = np.zeros_like(mean_metric_unnorm, dtype=float)
    for i in range(mean_metric.shape[2]):  # Loop over slices
        slice_data = mean_metric_unnorm[:, :, i]
        min_val, max_val = np.min(slice_data), np.max(slice_data)
        mean_metric[:, :, i] = (slice_data - min_val) / (max_val - min_val) if max_val > min_val else 0
    for i, m in enumerate(metrics_list):
        plot_metric_on_row(m, axes, row=i)
    plot_metric_on_row(mean_metric, axes, row=i+1)
    plot_metric_on_row(mask, axes, row=i+2)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.05)
    return fig_metric, axes


def crop_to_mask(sbref_vol, mask_vol, pad=10):
    nz = mask_vol.shape[2]
    
    # Find slice with largest mask
    areas = [mask_vol[:, :, i].sum() for i in range(nz)]
    max_slice = np.argmax(areas)
    
    if areas[max_slice] == 0:
        raise ValueError("Mask is empty — cannot compute center of mass.")
    
    # Center of mass in that slice
    cx, cy = center_of_mass(mask_vol[:, :, max_slice])
    cx, cy = int(round(cx)), int(round(cy))
    
    # Compute bounding box with padding
    nx, ny = sbref_vol.shape[:2]
    x0 = max(cx - pad, 0)
    x1 = min(cx + pad, nx)
    y0 = max(cy - pad, 0)
    y1 = min(cy + pad, ny)
    
    # Apply crop to all slices
    sbref_crop = sbref_vol[x0:x1, y0:y1, :nz]
    mask_crop  = mask_vol[x0:x1, y0:y1, :nz]
    
    return sbref_crop, mask_crop, (x0, x1, y0, y1)


def plot_mask_overlay(func_vol, sbref_vol, mask_vol, synthseg_mask, pad=10,
                      cmap_sbref='gray', cmap_func='gray',
                      cmap_mask='Reds', cmap_synthseg='Blues',
                      alpha=0.4):
    """
    Plot sbref and mean functional with mask overlay, with comparison to SynthSeg.
    """
    # Collapse time dimension
    func_vol_mn = func_vol.mean(axis=-1)

    # Crop around ROI
    sbref_crop, mask_crop, bounds = crop_to_mask(sbref_vol, mask_vol, pad=pad)
    func_crop, _, _ = crop_to_mask(func_vol_mn, mask_vol, pad=pad)
    synthseg_crop, _, _ = crop_to_mask(synthseg_mask, mask_vol, pad=pad)
    
    print(f"func_crop shape: {func_crop.shape}")
    print(f"sbref_crop shape: {sbref_crop.shape}")
    print(f"mask_crop shape: {mask_crop.shape}")
    print(f"synthseg_crop shape: {synthseg_crop.shape}")

    nz = synthseg_crop.shape[2]
    fig, axes = plt.subplots(3, nz, figsize=(4*nz, 12))

    if nz == 1:
        axes = np.array(axes).reshape(3, 1)

    for i in range(nz):
        # Row 0: sbref + output mask
        axes[0, i].imshow(sbref_crop[:, :, i], cmap=cmap_sbref)
        axes[0, i].imshow(mask_crop[:, :, i], cmap=cmap_mask, alpha=alpha)
        axes[0, i].set_title(f"Slice {i} (SBRef + Mask)")
        axes[0, i].axis("off")

        # Row 1: mean functional + mask
        axes[1, i].imshow(func_crop[:, :, i], cmap=cmap_func)
        axes[1, i].imshow(mask_crop[:, :, i], cmap=cmap_mask, alpha=alpha)
        axes[1, i].set_title(f"Slice {i} (Mean Func + Mask)")
        axes[1, i].axis("off")

        # Row 2: sbref + synthseg + mask overlay
        axes[2, i].imshow(sbref_crop[:, :, i], cmap=cmap_sbref)
        axes[2, i].imshow(synthseg_crop[:, :, i], cmap=cmap_synthseg, alpha=alpha)
        axes[2, i].imshow(mask_crop[:, :, i], cmap=cmap_mask, alpha=alpha*0.7)
        axes[2, i].set_title(f"Slice {i} (SBRef + SynthSeg + Mask)")
        axes[2, i].axis("off")

    plt.tight_layout()
    return fig, axes


def plot_signal(s, initial_trim=20, smoothing=None):
    """
    Plot extracted signal.
    - s: array of shape (timepoints, slices)
    - smoothing: optional integer, window size for moving average
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    splot = s.copy()
    splot = splot[initial_trim:, :]
    
    # Raw signals
    axes[0].plot(splot)
    axes[0].set_xlabel("Timepoint")
    axes[0].set_ylabel("Signal amplitude")
    axes[0].set_title(f"Raw signal per slice ({initial_trim} volume trimmed)")
    axes[0].grid(True)

    # Scaled signal
    scaled = utils.scale_data(splot)
    if smoothing is not None:
        import pandas as pd
        scaled = pd.DataFrame(scaled).rolling(smoothing, min_periods=1, center=True).mean().values

    axes[1].plot(scaled)
    axes[1].set_xlabel("Timepoint")
    axes[1].set_ylabel("Scaled signal")
    axes[1].set_title("Scaled signal per slice ({initial_trim} volume trimmed)")
    axes[1].grid(True)

    plt.tight_layout()
    return fig, axes


def map_window_to_full(window_mask, window_coords, full_shape):
    full_mask = np.zeros(full_shape, dtype=window_mask.dtype)
    for islice, (x_start, x_end, y_start, y_end, z) in enumerate(window_coords):
        full_mask[x_start:x_end, y_start:y_end, z] = window_mask[:, :, islice]
    return full_mask


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


if __name__ == "__main__":
    main()