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
    return s[:, :, np.newaxis] - s[:, np.newaxis, :]

def get_mask(metrics, weights, thres=1.0):
    mean_metric_voxels = []
    nslice = len(metrics[0])
    weights = np.array(weights)

    for islice in range(nslice):
        slice_metrics = [metrics[i][islice] for i in range(len(metrics))]
        stacked = np.stack(slice_metrics, axis=0)
        mean_data = np.average(stacked, axis=0, weights=weights)
        mean_metric_voxels.append(mean_data)
    if np.isscalar(thres) or len(thres) == 1:
        mask = [(mean_metric_voxels[i] > thres).astype(np.uint8) for i in range(nslice)]
    else:
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

def mean_pct_portion(x, pct, fromtop=False):
    x_sorted = np.sort(x)
    if fromtop:
        x_sorted = x_sorted[::-1]
    n = max(1, round(len(x_sorted) * pct / 100))
    return np.mean(x_sorted[:n])

def scale_data(s, bottom_pct=2.5):
    s_copy = s.copy()
    for ch in range(s_copy.shape[1]):
        baseline = mean_pct_portion(s_copy[:, ch], bottom_pct)
        s_copy[:, ch] -= baseline
        s_copy[:, ch] /= baseline
    return s_copy