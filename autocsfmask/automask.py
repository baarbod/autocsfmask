import os
import json
import logging
import argparse
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from scipy import signal
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

def main():
    parser = argparse.ArgumentParser(description='Automatic 4th ventricle CSF flow segmentation')
    parser.add_argument('--func', required=True, type=str, help='Path to functional data')
    parser.add_argument('--sbref', required=True, type=str, help='Path to sbref')
    parser.add_argument('--boundmask', required=True, type=str, help='Path to mask which covers the flow compartment')
    parser.add_argument('--outdir', default='outputs', type=str, help='Output directory')
    parser.add_argument('--metrics', default=['sbref'], nargs='+', type=str, help='Metrics to use')
    args = parser.parse_args()
    
    run_automask(
        func=args.func, 
        sbref=args.sbref, 
        boundmask=args.boundmask, 
        outdir=args.outdir, 
        metrics_list_names=args.metrics
    )

def run_automask(func, sbref, boundmask, outdir, metrics_list_names=['sd', 'sbref']):
    os.makedirs(outdir, exist_ok=True)
    setup_logging(outdir)

    logging.info("Loading data...")
    func_voxels, sbref_voxels, coords, func_affine, func_header, func_data, sbref_data = load_data(
        func, sbref, boundmask
    )

    logging.info("Computing metrics: %s", metrics_list_names)
    metrics_list = get_metrics(func_voxels, sbref_voxels, metrics_list_names)

    logging.info("Generating mask using optimization")
    mask, weights, thres, best_score = generate_mask(metrics_list, func_voxels)
    
    # Convert mask list to full volume
    mask_vol = voxel_list_to_volume(mask, coords, func_data.shape[:3])

    print(f"func shape: {func_data.shape}")
    print(f"sbref shape: {sbref_data.shape}")
    print(f"mask shape: {mask_vol.shape}")

    # Reconstruct metric volumes for the plot_metrics function
    metric_volumes = [voxel_list_to_volume(m, coords, func_data.shape[:3]) for m in metrics_list]

    # --- Plotting Metrics & Optimization ---
    logging.info("Saving metric optimization plots...")
    boundmask_arr = np.load(boundmask)
    fig_metrics, _ = plot_metrics(
        nslice=boundmask_arr.shape[2], 
        metrics_list=metric_volumes, 
        weights=weights, 
        mask=mask_vol, 
        metric_row_titles=metrics_list_names
    )
    fig_metrics.savefig(os.path.join(outdir, "metrics.png"))

    # Plot overlay
    fig_overlay, _ = plot_mask_overlay(func_data, sbref_data, mask_vol, boundmask_arr)
    fig_overlay.savefig(os.path.join(outdir, "mask_overlay.png"))

    # Extract signal and plot
    logging.info("Extracting signal from mask...")
    s = utils.get_signal(func_voxels, mask)
    np.savetxt(os.path.join(outdir, "signal.txt"), s)
    fig_signal, _ = plot_signal(s)
    fig_signal.savefig(os.path.join(outdir, "signal.png"))

    # Save mask volume
    mask_path = os.path.join(outdir, "mask.nii.gz")
    nib.save(nib.Nifti1Image(mask_vol, func_affine, header=func_header), mask_path)
    logging.info("Mask saved to: %s", mask_path)

    # Save optimization parameters if applicable
    optimization_results = {"weights": weights.tolist(), "thresholds": thres.tolist(), "best_score": best_score}
    with open(os.path.join(outdir, "optimal_params.json"), "w") as f:
        json.dump(optimization_results, f, indent=4)

    logging.info("Segmentation complete. All outputs saved to: %s", outdir)

def get_metrics(func_voxels, sbref_voxels, mlist):
    metrics_list = []
    for m in mlist:
        if m == 'mean':
            metrics_list.append(metrics.compute_mean(func_voxels))
        elif m == 'sd':
            metrics_list.append(metrics.compute_std(func_voxels))
        elif m == 'mean_bottom':
            metrics_list.append(metrics.compute_bottom_mean(func_voxels))
        elif m == 'skew':
            metrics_list.append(metrics.compute_skew(func_voxels))
        elif m == 'sbref':
            metrics_list.append(metrics.compute_sbref(sbref_voxels))
        else:
            raise ValueError(f"Unknown metric: {m}")
    return metrics_list

def generate_mask(metrics_list, func_voxels, N=1):
    best_score = np.inf
    best_mask, best_weights, best_thres = None, None, None
    for i in range(N):
        logging.info("Optimization run %d/%d", i+1, N)
        mask, w, t, score = masking.get_mask_optim(metrics_list, func_voxels)
        if score < best_score:
            best_score = score
            best_mask, best_weights, best_thres = mask, w, t
            logging.info("New best score: %.6f", best_score)
    logging.info("Best overall score: %.6f", best_score)
    return best_mask, best_weights, best_thres, best_score

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

def plot_metrics(nslice, metrics_list, weights, mask, metric_row_titles, pad=10):
    _, _, (x0, x1, y0, y1) = crop_to_mask(mask, mask, pad=pad)
    row_titles = [m.upper() for m in metric_row_titles] + ['COMBINED SCORE', 'FINAL MASK']
    nrows = len(row_titles)
    fig = plt.figure(figsize=(3 * nslice + 1, 3 * nrows))
    gs = fig.add_gridspec(nrows, nslice, hspace=0.3, wspace=0.1)
    weights_arr = np.array(weights).reshape(-1, 1, 1, 1)
    mean_metric = np.sum(np.stack(metrics_list, axis=0) * weights_arr, axis=0)
    rows_data = metrics_list + [mean_metric, mask]
    for r in range(nrows):
        vol_crop = rows_data[r][x0:x1, y0:y1, :]
        if r == nrows - 1: # Mask
            cmap = 'Reds'
        elif r == nrows - 2: # Combined Score
            cmap = 'viridis' 
        else:
            cmap = 'magma'
        for c in range(nslice):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(vol_crop[:, :, c], cmap=cmap)
            if r == 0:
                ax.set_title(f"Slice {c}", fontsize=12, fontweight='bold')
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=11, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            if r == nrows - 2 and c == nslice - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, gs

def crop_to_mask(sbref_vol, mask_vol, pad=10):
    nz = mask_vol.shape[2]
    areas = [mask_vol[:, :, i].sum() for i in range(nz)]
    max_slice = np.argmax(areas)
    if areas[max_slice] == 0:
        raise ValueError("Mask is empty — cannot compute center of mass.")
    cx, cy = center_of_mass(mask_vol[:, :, max_slice])
    cx, cy = int(round(cx)), int(round(cy))
    nx, ny = sbref_vol.shape[:2]
    x0 = max(cx - pad, 0)
    x1 = min(cx + pad, nx)
    y0 = max(cy - pad, 0)
    y1 = min(cy + pad, ny)
    sbref_crop = sbref_vol[x0:x1, y0:y1, :nz]
    mask_crop  = mask_vol[x0:x1, y0:y1, :nz]
    return sbref_crop, mask_crop, (x0, x1, y0, y1)

def plot_mask_overlay(func_vol, sbref_vol, mask_vol, boundmask_arr, pad=10,
                      cmap_sbref='gray', cmap_func='gray',
                      cmap_mask='Reds', cmap_synthseg='Blues',
                      alpha=0.4):
    func_vol_mn = func_vol.mean(axis=-1)
    sbref_crop, mask_crop, bounds = crop_to_mask(sbref_vol, mask_vol, pad=pad)
    func_crop, _, _ = crop_to_mask(func_vol_mn, mask_vol, pad=pad)
    synthseg_crop, _, _ = crop_to_mask(boundmask_arr, mask_vol, pad=pad)
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
        # Row 2: sbref + boundmask + mask overlay
        axes[2, i].imshow(sbref_crop[:, :, i], cmap=cmap_sbref)
        axes[2, i].imshow(synthseg_crop[:, :, i], cmap=cmap_synthseg, alpha=alpha)
        axes[2, i].imshow(mask_crop[:, :, i], cmap=cmap_mask, alpha=alpha*0.7)
        axes[2, i].set_title(f"Slice {i} (SBRef + SynthSeg + Mask)")
        axes[2, i].axis("off")
    plt.tight_layout()
    return fig, axes

def plot_signal(s, initial_trim=20, smoothing=25, ref_val_mean=1):
    
    def scale_data(s, pct=2.5):
        raw_mean = np.mean(s, axis=0)
        detrended = signal.detrend(s, axis=0)
        return detrended / raw_mean

    splot = s[initial_trim:, :]
    scaled = scale_data(splot)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle(f"CSF Signal Extraction Analysis (Trimmed: {initial_trim} vols)", 
                 fontsize=16, fontweight='bold', y=1.05)
    titles = ["Raw Signal", "Normalized", f"Smoothed (Win={smoothing})"]
    data_to_plot = [splot, scaled]
    scaled_smoothed = pd.DataFrame(scaled).rolling(smoothing, min_periods=1, center=True).mean().values
    data_to_plot.append(scaled_smoothed)
    for i, ax in enumerate(axes):
        ax.plot(data_to_plot[i], linewidth=1.5, alpha=0.8)
        ax.set_title(titles[i], fontsize=13, pad=10)
        ax.set_xlabel("Timepoint", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.set_ylabel("Amplitude (A.U.)", fontsize=10)
        else:
            ax.set_ylabel("Normalized Units", fontsize=10)
    return fig, axes

def voxel_list_to_volume(voxel_list, coords_list, full_shape):
    example = voxel_list[0]
    if example.ndim == 1:
        volume = np.zeros(full_shape, dtype=example.dtype)
        for voxels, coords in zip(voxel_list, coords_list):
            volume[coords[:, 0], coords[:, 1], coords[:, 2]] = voxels
    elif example.ndim == 2:
        volume = np.zeros(full_shape + (example.shape[1],), dtype=example.dtype)
        for voxels, coords in zip(voxel_list, coords_list):
            volume[coords[:, 0], coords[:, 1], coords[:, 2], :] = voxels
    return volume

if __name__ == "__main__":
    main()