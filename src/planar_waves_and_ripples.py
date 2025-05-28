#!/usr/bin/env python3
"""
Planar Waves and Sharp Wave-Ripples Analysis (Optimized)
=======================================================
Analysis of coherent local field potential structures in the hippocampus 
and their relation to sharp waves and ripples.
Optimized version with single PGD computation.
"""

# Imports
from braingeneers import analysis
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy import stats
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from cmcrameri import cm as cmc
from loaders import *
from new_lfp_processor_class import *
import os
import pywt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import concurrent.futures
import time
import psutil
from datetime import datetime
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from scipy.signal import butter, filtfilt
from scipy import stats as scipy_stats
from matplotlib.patches import FancyArrowPatch
import pathlib
import math

# Configuration dictionary to replace duplicate Config classes
WAVE_CONFIG = {
    'MAX_NEIGHBOR_DIST': 50.0,      # µm
    'SPATIAL_SIGMA': 45.0,           # µm for Gaussian kernel
    'RIDGE_LAMBDA': 1e-5,            # regularization
    'COVERAGE_ANGLE_GAP': 120.0,     # degrees
    'MIN_GRADIENT': 1e-5,            # rad/µm (default, can be overridden)
    'V_MIN': 0.0,                    # minimum wave speed
    'V_MAX': 300_000.0,              # maximum wave speed (μm/s)
}

# Single RegularizedGradient class to replace duplicates
class RegularizedGradient:
    """
    Compute regularized phase gradients for wave analysis.
    This single class replaces all duplicate implementations.
    """
    def __init__(self, locations, config=None, use_gpu=False, xp=None):
        self.locations = locations
        self.cfg = config if config is not None else WAVE_CONFIG.copy()
        self.use_gpu = use_gpu
        self.xp = xp if xp is not None else np
        
        # Build KD-tree
        self.tree = cKDTree(locations)
        # Precompute neighbors within MAX_NEIGHBOR_DIST
        dists, idxs = self.tree.query(locations,
                                      k=min(len(locations), 20),
                                      distance_upper_bound=self.cfg['MAX_NEIGHBOR_DIST'])
        # Trim infinite distances
        self.neighbor_indices = [idxs[i][dists[i] < np.inf] for i in range(len(locations))]
        self.neighbor_dists = [dists[i][dists[i] < np.inf] for i in range(len(locations))]
        
        # Precompute neighbor relationships
        self._precompute_neighbors()

    def compute(self, phase_data):
        n = len(phase_data)
        grad_x = np.zeros(n, dtype=np.float64)
        grad_y = np.zeros(n, dtype=np.float64)

        for i in range(n):
            nbrs = self.neighbor_indices[i]
            dists = self.neighbor_dists[i]
            if len(nbrs) < 3:
                continue  # insufficient neighbors

            # Exclude self
            mask = nbrs != i
            nbrs = nbrs[mask]
            dists = dists[mask]

            if len(nbrs) < 3:
                continue  # insufficient neighbors after excluding self
            
            dx = self.locations[nbrs, 0] - self.locations[i, 0]
            dy = self.locations[nbrs, 1] - self.locations[i, 1]

            # Phase diff with wrapping
            phase_diff = phase_data[nbrs] - phase_data[i]
            diffs = ((phase_diff + np.pi) % (2*np.pi)) - np.pi

            # Gaussian weights
            w = np.exp(-0.5 * (dists / self.cfg['SPATIAL_SIGMA'])**2)

            # Coverage check
            angles = np.degrees(np.mod(np.arctan2(dy, dx), 2*np.pi))
            angles_sorted = np.sort(angles)
            gaps = np.diff(np.concatenate([angles_sorted, angles_sorted[:1] + 360]))
            if np.max(gaps) > self.cfg['COVERAGE_ANGLE_GAP']:
                continue  # poor coverage

            # Weighted design matrix
            A = np.stack([dx, dy], axis=1)
            
            # Ensure weights are properly used
            w_reshape = w.reshape(-1, 1)
            
            # Regularized least squares
            ATA = A.T @ (w_reshape * A) + self.cfg['RIDGE_LAMBDA'] * np.eye(2)
            ATb = A.T @ (w * diffs)
            
            try:
                g = np.linalg.solve(ATA, ATb)
                grad_x[i], grad_y[i] = g[0], g[1]
            except np.linalg.LinAlgError:
                # skip if singular
                continue

        return grad_x, grad_y
    
    def _precompute_neighbors(self):
        """Precompute neighbor relationships for all electrodes"""
        # Store neighbor relationships
        self.dx_dy_arrays = []
        self.weight_arrays = []
        self.coverage_masks = []
        
        for i in range(len(self.locations)):
            nbrs = self.neighbor_indices[i]
            dist = self.neighbor_dists[i]
            
            # Exclude self
            mask = nbrs != i
            nbrs = nbrs[mask]
            dist = dist[mask]
            
            # Store if we have enough neighbors
            if len(nbrs) >= 3:
                # Calculate dx, dy for each neighbor
                dx = self.locations[nbrs, 0] - self.locations[i, 0]
                dy = self.locations[nbrs, 1] - self.locations[i, 1]
                
                # Compute Gaussian weights
                weights = np.exp(-0.5 * (dist / self.cfg['SPATIAL_SIGMA'])**2)
                
                # Check angular coverage
                angles = np.degrees(np.mod(np.arctan2(dy, dx), 2*np.pi))
                angles_sorted = np.sort(angles)
                gaps = np.diff(np.concatenate([angles_sorted, angles_sorted[:1] + 360]))
                has_coverage = np.max(gaps) <= self.cfg['COVERAGE_ANGLE_GAP']
                
                self.dx_dy_arrays.append(np.column_stack([dx, dy]))
                self.weight_arrays.append(weights)
                self.coverage_masks.append(has_coverage)
            else:
                self.dx_dy_arrays.append(np.empty((0, 2)))
                self.weight_arrays.append(np.array([]))
                self.coverage_masks.append(False)
        
        # Mark valid electrodes for efficiency
        self.has_valid_neighbors = np.array([len(nbrs) >= 3 for nbrs in self.neighbor_indices])
        self.has_coverage = np.array(self.coverage_masks)
        self.valid_electrodes = self.has_valid_neighbors & self.has_coverage
    
    def compute_pgd_for_timepoint(self, phase_data):
        """
        Compute PGD for a single timepoint.
        
        Parameters:
        -----------
        phase_data : array
            Phase values for all electrodes at a specific time point
            
        Returns:
        --------
        float
            PGD value
        """
        # Compute gradients
        grad_x, grad_y = self.compute(phase_data)
        
        # Calculate gradient magnitudes
        gradients = np.column_stack((grad_x, grad_y))
        grad_magnitudes = np.sqrt(np.sum(gradients**2, axis=1))
        
        # Mask out invalid gradients
        valid_mask = grad_magnitudes >= self.cfg['MIN_GRADIENT']
        valid_gradients = gradients[valid_mask]
        valid_magnitudes = grad_magnitudes[valid_mask]
        
        if len(valid_gradients) > 0:
            # Calculate PGD: ||∇φ|| / ||∇φ||
            mean_gradient = np.mean(valid_gradients, axis=0)
            mean_gradient_magnitude = np.linalg.norm(mean_gradient)
            mean_magnitude = np.mean(valid_magnitudes)
            
            if mean_magnitude > 0:
                return mean_gradient_magnitude / mean_magnitude
        
        return 0.0
    
    def compute_pgd_batch(self, phase_data_batch):
        """
        Compute PGD values for a batch of time points.
        
        Parameters:
        -----------
        phase_data_batch : array, shape (n_electrodes, n_timepoints)
            Phase data for multiple time points
        
        Returns:
        --------
        array, shape (n_timepoints,)
            PGD values for each time point
        """
        n_timepoints = phase_data_batch.shape[1]
        pgd_values = np.zeros(n_timepoints, dtype=np.float32)
        
        for t in range(n_timepoints):
            pgd_values[t] = self.compute_pgd_for_timepoint(phase_data_batch[:, t])
        
        return pgd_values


# New function to compute PGD once for entire dataset
def compute_pgd_for_window(lfp_processor, data_type='sharpWave', window_start=0, window_length=None,
                         downsample_factor=1, smoothing_sigma=15, min_gradient=1e-5,
                         use_gpu=True, batch_size=100, verbose=True):
    """
    Compute PGD values once for an entire window and return the results.
    
    Parameters:
    -----------
    lfp_processor : LFPDataProcessor
        Instance of LFPDataProcessor containing wave data
    data_type : str
        Type of oscillation to analyze
    window_start : int
        Start frame for analysis window
    window_length : int or None
        Length of analysis window in frames
    downsample_factor : int
        Factor by which to downsample the time series
    smoothing_sigma : float
        Standard deviation for Gaussian smoothing kernel
    min_gradient : float
        Minimum gradient magnitude for calculations
    use_gpu : bool
        Whether to use GPU acceleration if available
    batch_size : int
        Number of timepoints to process simultaneously
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'time_points': Time points (s)
        - 'pgd_raw': Raw PGD values
        - 'pgd_smooth': Smoothed PGD values
        - 'grad_calc': Gradient calculator instance
        - 'window_start': Start frame
        - 'window_end': End frame
    """
    # Custom logger
    def log(message):
        if verbose:
            current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{current_time}] {message}")
    
    log(f"Computing PGD for {data_type} band...")
    
    # Validate inputs
    if data_type not in lfp_processor.waves:
        raise ValueError(f"Data type {data_type} not found in waves dictionary")
    
    if window_length is None:
        window_length = lfp_processor.waves[data_type].shape[1] - window_start
    
    window_end = window_start + window_length
    if window_end > lfp_processor.waves[data_type].shape[1]:
        raise ValueError("Analysis window exceeds data length")
    
    # Get phase data
    log("Getting analytical data...")
    analytical_data = lfp_processor._get_analytical_data(data_type, window_start, window_end)
    phase_data = analytical_data['phase']
    
    # Generate time points
    time_points = np.arange(window_start, window_end, downsample_factor) / lfp_processor.fs
    
    # Initialize gradient calculator
    config = WAVE_CONFIG.copy()
    config['MIN_GRADIENT'] = min_gradient
    grad_calc = RegularizedGradient(lfp_processor.locations, config, use_gpu)
    
    # Process data in batches
    n_times = len(range(0, window_length, downsample_factor))
    pgd_values = np.zeros(n_times, dtype=np.float32)
    
    log(f"Computing PGD for {n_times} timepoints...")
    
    # Create downsampled indices
    downsampled_indices = list(range(0, window_length, downsample_factor))
    if len(downsampled_indices) > n_times:
        downsampled_indices = downsampled_indices[:n_times]
    
    # Process in batches
    n_batches = (n_times + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(downsampled_indices))
        
        if end_idx <= start_idx:
            continue
        
        batch_indices = downsampled_indices[start_idx:end_idx]
        phase_data_batch = phase_data[:, batch_indices]
        
        # Compute PGD for this batch
        pgd_batch = grad_calc.compute_pgd_batch(phase_data_batch)
        pgd_values[start_idx:end_idx] = pgd_batch
        
        # Print progress
        if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
            progress = (batch_idx+1)/n_batches*100
            log(f"Processed batch {batch_idx+1}/{n_batches} ({progress:.1f}%)")
    
    # Apply smoothing
    log("Applying Gaussian smoothing...")
    smoothed_pgd = gaussian_filter1d(pgd_values, sigma=smoothing_sigma)
    
    log("PGD computation completed")
    
    return {
        'time_points': time_points,
        'pgd_raw': pgd_values,
        'pgd_smooth': smoothed_pgd,
        'grad_calc': grad_calc,
        'window_start': window_start,
        'window_end': window_end,
        'downsample_factor': downsample_factor,
        'phase_data': phase_data,  # Store phase data for later use
        'analytical_data': analytical_data
    }


# Modified detect_pgd_peaks function to use precomputed PGD
def detect_pgd_peaks_from_precomputed(pgd_data, threshold=0.5, min_duration=0.1, min_interval=0.2,
                                    return_raw=False, plot_results=False, figsize=(15, 6),
                                    color='green', highlight_color='lightgreen', highlight_alpha=0.3,
                                    show_stats=True, title=None, save_path=None):
    """
    Detect PGD peaks from precomputed PGD data.
    
    Parameters:
    -----------
    pgd_data : dict
        Output from compute_pgd_for_window function
    threshold : float
        Threshold for peak detection as a multiple of the root mean square
    min_duration : float
        Minimum duration of a peak in seconds
    min_interval : float
        Minimum interval between peaks in seconds
    return_raw : bool
        Whether to return raw PGD data
    plot_results : bool
        Whether to plot the results
    Other parameters same as original detect_pgd_peaks_accelerated
        
    Returns:
    --------
    dict
        Dictionary containing peak information
    """
    # Extract data from precomputed results
    time_points = pgd_data['time_points']
    smoothed_pgd = pgd_data['pgd_smooth']
    
    # Calculate threshold based on RMS
    rms = np.sqrt(np.mean(smoothed_pgd**2))
    peak_threshold = rms * threshold
    
    # Initialize peaks dictionary
    peaks = {
        'peak_times': [],
        'peak_heights': [],
        'intervals': [],
        'durations': [],
        'threshold': peak_threshold
    }
    
    # Find peaks above threshold
    fs = 1 / (time_points[1] - time_points[0]) if len(time_points) > 1 else 1000  # Approximate fs
    peak_indices, peak_properties = find_peaks(
        smoothed_pgd, 
        height=peak_threshold, 
        distance=int(min_interval * fs / pgd_data.get('downsample_factor', 1))
    )
    
    if len(peak_indices) > 0:
        # Calculate widths for each peak
        widths, width_heights, left_ips, right_ips = peak_widths(
            smoothed_pgd, peak_indices, rel_height=0.3)
        
        # Convert indices to times
        peak_times = time_points[peak_indices]
        peak_heights = smoothed_pgd[peak_indices]
        
        # Create time interpolator
        time_interpolator = interp1d(
            np.arange(len(time_points)), 
            time_points, 
            bounds_error=False, 
            fill_value="extrapolate"
        )
        
        # Convert left and right indices to times
        left_times = time_interpolator(left_ips)
        right_times = time_interpolator(right_ips)
        
        # Filter peaks based on minimum duration
        valid_peaks = []
        for i, (left, right) in enumerate(zip(left_times, right_times)):
            duration = right - left
            if duration >= min_duration:
                valid_peaks.append(i)
        
        # Store peak information for valid peaks
        if valid_peaks:
            peaks['peak_times'] = peak_times[valid_peaks]
            peaks['peak_heights'] = peak_heights[valid_peaks]
            peaks['intervals'] = list(zip(left_times[valid_peaks], right_times[valid_peaks]))
            peaks['durations'] = right_times[valid_peaks] - left_times[valid_peaks]
    
    # Add raw data if requested
    if return_raw:
        peaks['time_bins'] = time_points
        peaks['pgd'] = smoothed_pgd
    
    # Plot results if requested
    if plot_results:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot PGD
        ax.plot(time_points, smoothed_pgd, color=color, linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Phase Gradient Directionality', fontsize=12, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        
        # Add fill under curve
        ax.fill_between(time_points, 0, smoothed_pgd, color=color, alpha=0.2)
        
        # Plot threshold line
        ax.axhline(y=peak_threshold, color='gray', linestyle='--', alpha=0.7, 
                  label=f'Threshold ({threshold}×RMS)')
        
        # Highlight detected peaks
        for i, (start, end) in enumerate(peaks['intervals']):
            ax.axvspan(start, end, color=highlight_color, alpha=highlight_alpha)
            
            # Add peak marker
            peak_time = peaks['peak_times'][i]
            peak_value = peaks['peak_heights'][i]
            ax.plot(peak_time, peak_value, 'o', color=highlight_color, 
                   markersize=8, alpha=0.8)
        
        # Set title
        if title is None:
            title = "Phase Gradient Directionality Peaks"
        ax.set_title(title, fontsize=14)
        
        # Add statistics if requested
        if show_stats and len(peaks['peak_times']) > 0:
            stats_text = (
                f"Detected peaks: {len(peaks['peak_times'])}\n"
                f"Threshold: {peak_threshold:.4f} ({threshold}×RMS)\n"
                f"Mean duration: {np.mean(peaks['durations']):.3f}s\n"
                f"Mean peak value: {np.mean(peaks['peak_heights']):.4f}"
            )
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            # Save both PNG and SVG
            base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
            fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{base_path}.svg", format='svg', bbox_inches='tight')
            print(f"Saved figures to {base_path}.png and {base_path}.svg")
        
        # Store figure and axis in the result
        peaks['fig'] = fig
        peaks['ax'] = ax
    
    return peaks


# Helper function to save figures in both formats
def save_figure_both_formats(fig, base_path, dpi=300):
    """Save a figure in both PNG and SVG formats."""
    if base_path:
        # Remove extension if present
        base = base_path.rsplit('.', 1)[0] if '.' in base_path else base_path
        fig.savefig(f"{base}.png", dpi=dpi, bbox_inches='tight')
        fig.savefig(f"{base}.svg", format='svg', bbox_inches='tight')
        print(f"Saved: {base}.png and {base}.svg")


# Simplified detect_energy_peaks (keeping as is, no PGD computation here)
def detect_energy_peaks(lfp_processor, data_type='lfp', window_start=0, window_length=None, 
                       downsample_factor=1, smoothing_sigma=15, threshold=2.0, 
                       min_duration=0.1, min_interval=0.2, return_raw=False, 
                       plot_results=False, figsize=(15, 6), 
                       color='blue', highlight_color='skyblue', highlight_alpha=0.3,
                       show_stats=True, title=None, optogenetic_intervals=None, save_path=None):
    """
    Detect peaks of high energy in LFP data with option to omit optogenetic intervals.
    """
    # Validate inputs
    if data_type not in lfp_processor.waves:
        raise ValueError(f"Data type {data_type} not found in waves dictionary")
    
    # Validate optogenetic intervals
    if optogenetic_intervals is not None:
        optogenetic_intervals = sorted(optogenetic_intervals, key=lambda x: x[0])
        print(f"Using {len(optogenetic_intervals)} optogenetic intervals for filtering")
    
    # Get window parameters
    if window_length is None:
        window_length = lfp_processor.waves[data_type].shape[1] - window_start
    
    window_end = window_start + window_length
    if window_end > lfp_processor.waves[data_type].shape[1]:
        raise ValueError("Analysis window exceeds data length")
    
    # Compute instantaneous energy
    time_points, mean_energy, std_energy = lfp_processor.compute_instantaneous_energy(
        data_type, window_start, window_length, downsample_factor)
    
    # Apply Gaussian smoothing
    smoothed_energy = gaussian_filter1d(mean_energy, sigma=smoothing_sigma)
    
    # Initialize energy peaks dictionary
    peaks = {
        'peak_times': [],
        'peak_heights': [],
        'intervals': [],
        'durations': [],
        'threshold': 0,
        'n_peaks_before_filtering': 0,
        'n_peaks_filtered_out': 0,
        'optogenetic_intervals_used': optogenetic_intervals
    }
    
    # Calculate threshold based on RMS
    rms = np.sqrt(np.mean(smoothed_energy**2))
    peak_threshold = rms * threshold
    peaks['threshold'] = peak_threshold
    
    # Find peaks above threshold
    peak_indices, peak_properties = find_peaks(
        smoothed_energy, 
        height=peak_threshold, 
        distance=int(min_interval * lfp_processor.fs / downsample_factor)
    )
    
    if len(peak_indices) > 0:
        # Calculate widths for each peak
        widths, width_heights, left_ips, right_ips = peak_widths(
            smoothed_energy, peak_indices, rel_height=0.9)
        
        # Convert indices to times
        peak_times = time_points[peak_indices]
        peak_heights = smoothed_energy[peak_indices]
        
        # Create time interpolator
        time_interpolator = interp1d(
            np.arange(len(time_points)), 
            time_points, 
            bounds_error=False, 
            fill_value="extrapolate"
        )
        
        # Convert left and right indices to times
        left_times = time_interpolator(left_ips)
        right_times = time_interpolator(right_ips)
        
        # Filter peaks based on minimum duration
        valid_peaks = []
        for i, (left, right) in enumerate(zip(left_times, right_times)):
            duration = right - left
            if duration >= min_duration:
                valid_peaks.append(i)
        
        # Store count before optogenetic filtering
        peaks['n_peaks_before_filtering'] = len(valid_peaks)
        
        # Apply optogenetic interval filtering if specified
        if optogenetic_intervals is not None and len(valid_peaks) > 0:
            def is_in_optogenetic_interval(peak_time, peak_start, peak_end, intervals):
                for opto_start, opto_end in intervals:
                    if opto_start <= peak_time <= opto_end:
                        return True
                    if peak_start < opto_end and peak_end > opto_start:
                        return True
                return False
            
            # Filter out peaks that overlap with optogenetic intervals
            filtered_valid_peaks = []
            for i in valid_peaks:
                peak_time = peak_times[i]
                peak_start = left_times[i]
                peak_end = right_times[i]
                
                if not is_in_optogenetic_interval(peak_time, peak_start, peak_end, optogenetic_intervals):
                    filtered_valid_peaks.append(i)
            
            # Update valid_peaks list
            n_filtered_out = len(valid_peaks) - len(filtered_valid_peaks)
            peaks['n_peaks_filtered_out'] = n_filtered_out
            valid_peaks = filtered_valid_peaks
            
            if n_filtered_out > 0:
                print(f"Filtered out {n_filtered_out} peaks that overlapped with optogenetic intervals")
        
        # Store peak information for valid peaks
        if valid_peaks:
            peaks['peak_times'] = peak_times[valid_peaks]
            peaks['peak_heights'] = peak_heights[valid_peaks]
            peaks['intervals'] = list(zip(left_times[valid_peaks], right_times[valid_peaks]))
            peaks['durations'] = right_times[valid_peaks] - left_times[valid_peaks]
    
    # Add raw data if requested
    if return_raw:
        peaks['time_bins'] = time_points
        peaks['energy'] = smoothed_energy
    
    # Plot results if requested
    if plot_results:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot energy
        ax.plot(time_points, smoothed_energy, color=color, linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        
        # Add fill under curve
        ax.fill_between(time_points, 0, smoothed_energy, color=color, alpha=0.2)
        
        # Plot threshold line
        ax.axhline(y=peak_threshold, color='gray', linestyle='--', alpha=0.7, 
                  label=f'Threshold ({threshold}×RMS)')
        
        # Highlight optogenetic intervals if provided
        if optogenetic_intervals is not None:
            for opto_start, opto_end in optogenetic_intervals:
                if opto_end >= time_points[0] and opto_start <= time_points[-1]:
                    ax.axvspan(opto_start, opto_end, color='red', alpha=0.15, 
                              label='Optogenetic intervals' if opto_start == optogenetic_intervals[0][0] else "")
        
        # Highlight detected peaks
        for i, (start, end) in enumerate(peaks['intervals']):
            ax.axvspan(start, end, color=highlight_color, alpha=highlight_alpha)
            
            # Add peak marker
            peak_time = peaks['peak_times'][i]
            peak_value = peaks['peak_heights'][i]
            ax.plot(peak_time, peak_value, 'o', color=highlight_color, 
                   markersize=8, alpha=0.8)
        
        # Set title
        if title is None:
            title = f"{data_type.upper()} Instantaneous Energy Peaks"
        ax.set_title(title, fontsize=14)
        
        # Add statistics if requested
        if show_stats:
            stats_text = f"Detected peaks: {len(peaks['peak_times'])}"
            
            if optogenetic_intervals is not None:
                stats_text += f" (after filtering)\nPeaks before filtering: {peaks['n_peaks_before_filtering']}"
                stats_text += f"\nFiltered out: {peaks['n_peaks_filtered_out']}"
            
            stats_text += f"\nThreshold: {peak_threshold:.4f} ({threshold}×RMS)"
            
            if len(peaks['peak_times']) > 0:
                stats_text += f"\nMean duration: {np.mean(peaks['durations']):.3f}s"
                stats_text += f"\nMean peak value: {np.mean(peaks['peak_heights']):.4f}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            save_figure_both_formats(fig, save_path)
        
        # Store figure and axis in the result
        peaks['fig'] = fig
        peaks['ax'] = ax
    
    return peaks


# Modified analyze_planar_waves_and_ripples to use precomputed PGD
def analyze_planar_waves_and_ripples_optimized(
    lfp_processor,
    sw_pgd_peaks,
    ripple_events,
    pgd_data_dict,  # New parameter: dictionary of precomputed PGD data by band
    bands=['sharpWave', 'narrowRipples'],
    window_size=0.5,
    smoothing_sigma=15,
    plot_individual=True,
    max_individual_plots=100,
    generate_summary=True,
    save_dir=None,
    dpi=300,
    verbose=True,
    figure_kwargs=None,
    swr_marker='line',
    swr_waveform_window=0.05,
    swr_waveform_height_scale=0.15,
    swr_waveform_smoothing=1.5,
    horizontal_scale_factor=0.5,
    waveform_colormap='cmc.hawaii'
):
    """
    Optimized version of analyze_planar_waves_and_ripples that uses precomputed PGD data.
    
    Additional Parameters:
    ---------------------
    pgd_data_dict : dict
        Dictionary mapping band names to precomputed PGD data from compute_pgd_for_window
    """
    # Validate swr_marker parameter
    if swr_marker not in ['line', 'waveform']:
        raise ValueError("swr_marker must be either 'line' or 'waveform'")
    
    # Color definitions
    colors = {
        'sharpWave': 'gray',
        'narrowRipples': 'black',
        'background': '#F5F5F5',
        'gridlines': '#CCCCCC',
        'text': '#333333',
        'ripple_line': '#000000',
        'ripple_waveform': '#000000',
        'highlight': '#FFFF00'
    }
    
    # Set default figure parameters
    default_fig_kwargs = {
        'figsize': (10, 6),
        'facecolor': colors['background'],
        'edgecolor': 'none',
        'dpi': 100
    }
    
    if figure_kwargs is not None:
        default_fig_kwargs.update(figure_kwargs)
    
    # Helper function for logging
    def log(message):
        if verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    # Helper function to extract ripple waveforms
    def extract_ripple_waveform(ripple_time, channel):
        try:
            half_window = swr_waveform_window / 2
            start_time = ripple_time - half_window
            end_time = ripple_time + half_window
            
            fs = lfp_processor.fs
            start_sample = max(0, int(start_time * fs))
            end_sample = min(lfp_processor.waves['lfp'].shape[1], int(end_time * fs))
            
            lfp_signal = lfp_processor.waves['lfp'][channel, start_sample:end_sample]
            
            if swr_waveform_smoothing > 0:
                smoothed_waveform = gaussian_filter1d(lfp_signal, sigma=swr_waveform_smoothing)
            else:
                smoothed_waveform = lfp_signal.copy()
            
            time_vector = np.arange(start_sample, end_sample) / fs - ripple_time
            
            return time_vector, smoothed_waveform
            
        except Exception as e:
            if verbose:
                log(f"Error extracting waveform for ripple at {ripple_time:.3f}s: {str(e)}")
            return None, None
    
    # Start timing and setup
    start_time = time.time()
    log(f"Starting comprehensive analysis of planar waves and ripples (SWR marker: {swr_marker})")
    
    # Create save directory if specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        log(f"Using save directory: {save_dir}")
    
    # Validate and extract ripple events
    ripple_times = []
    channel_ripples = {}
    ripple_channels = {}
    
    for ch in ripple_events:
        if isinstance(ch, (int, np.integer)):
            ch_ripples = ripple_events[ch]
            if len(ch_ripples) > 0:
                channel_ripples[ch] = ch_ripples['peak_time'].values
                ripple_times.extend(ch_ripples['peak_time'].values)
                
                for ripple_time in ch_ripples['peak_time'].values:
                    ripple_channels[ripple_time] = ch
    
    # Sort and remove duplicates
    ripple_times = np.sort(np.array(ripple_times))
    if len(ripple_times) > 1:
        unique_mask = np.concatenate(([True], np.diff(ripple_times) > 0.001))
        ripple_times = ripple_times[unique_mask]
        
        unique_ripple_channels = {}
        for ripple_time in ripple_times:
            for orig_time, channel in ripple_channels.items():
                if abs(orig_time - ripple_time) < 0.001:
                    unique_ripple_channels[ripple_time] = channel
                    break
        ripple_channels = unique_ripple_channels
    
    log(f"Found {len(ripple_times)} unique ripple events across {len(channel_ripples)} channels")
    
    # Extract PGD peak times
    pgd_peak_times = sw_pgd_peaks['peak_times']
    
    if len(pgd_peak_times) == 0:
        log("No PGD peaks found in input data")
        return None
        
    log(f"Analyzing {len(pgd_peak_times)} PGD peaks")
    
    # Limit the number of events to analyze if necessary
    if plot_individual and len(pgd_peak_times) > max_individual_plots:
        log(f"Limiting individual plot analysis to {max_individual_plots} events")
        indices = np.linspace(0, len(pgd_peak_times)-1, max_individual_plots).astype(int)
        events_to_plot = pgd_peak_times[indices]
    else:
        events_to_plot = pgd_peak_times
    
    # Initialize result containers
    results = {
        'pgd_data': {},
        'ripple_analysis': {
            'time_lags': [],
            'closest_ripple_per_event': [],
            'events_with_ripples': 0,
            'events_without_ripples': 0
        },
        'statistics': {},
        'figures': {},
        'report': '',
        'waveforms': {} if swr_marker == 'waveform' else None
    }
    
    # Calculate PGD values around each event using precomputed data
    log(f"Extracting PGD values from precomputed data for all bands around each event")
    fs = lfp_processor.fs
    window_frames = int(window_size * fs)
    time_axis = np.linspace(-window_size, window_size, 2 * window_frames + 1)
    
    # Initialize data structures for PGD values
    for band in bands:
        results['pgd_data'][band] = {
            'time': time_axis,
            'values': np.zeros((len(events_to_plot), len(time_axis))),
            'mean': None,
            'std': None
        }
    
    # Process each event using precomputed PGD data
    for e_idx, event_time in enumerate(events_to_plot):
        if verbose and (e_idx % 5 == 0 or e_idx == len(events_to_plot) - 1):
            log(f"Processing event {e_idx+1}/{len(events_to_plot)} at {event_time:.2f}s")
        
        # Find ripples within this window
        window_ripples = ripple_times[(ripple_times >= event_time - window_size) & 
                                       (ripple_times <= event_time + window_size)]
        
        # Record ripple statistics
        if len(window_ripples) > 0:
            results['ripple_analysis']['events_with_ripples'] += 1
            closest_ripple = window_ripples[np.argmin(np.abs(window_ripples - event_time))]
            results['ripple_analysis']['closest_ripple_per_event'].append(closest_ripple - event_time)
            
            for ripple_time in window_ripples:
                results['ripple_analysis']['time_lags'].append(ripple_time - event_time)
        else:
            results['ripple_analysis']['events_without_ripples'] += 1
        
        # Extract PGD values from precomputed data for each band
        for band in bands:
            if band not in pgd_data_dict:
                log(f"Warning: No precomputed PGD data for band {band}")
                continue
                
            pgd_data = pgd_data_dict[band]
            
            # Find the time indices in the precomputed data that correspond to our window
            precomputed_times = pgd_data['time_points']
            
            # Find indices for the window around the event
            window_start_time = event_time - window_size
            window_end_time = event_time + window_size
            
            # Find closest indices in precomputed data
            start_idx = np.argmin(np.abs(precomputed_times - window_start_time))
            end_idx = np.argmin(np.abs(precomputed_times - window_end_time))
            
            # Extract the PGD values
            if end_idx > start_idx:
                # Interpolate to match our desired time axis
                f_interp = interp1d(precomputed_times[start_idx:end_idx+1], 
                                  pgd_data['pgd_smooth'][start_idx:end_idx+1],
                                  bounds_error=False, fill_value=0)
                
                # Create time points relative to event
                absolute_times = time_axis + event_time
                
                # Only interpolate for times within the precomputed range
                valid_mask = (absolute_times >= precomputed_times[0]) & (absolute_times <= precomputed_times[-1])
                
                if np.any(valid_mask):
                    results['pgd_data'][band]['values'][e_idx, valid_mask] = f_interp(absolute_times[valid_mask])
    
    # Calculate statistics for each band
    for band in bands:
        results['pgd_data'][band]['mean'] = np.nanmean(results['pgd_data'][band]['values'], axis=0)
        results['pgd_data'][band]['std'] = np.nanstd(results['pgd_data'][band]['values'], axis=0)
    
    # Generate individual event plots
    if plot_individual:
        log(f"Generating individual event plots for {len(events_to_plot)} events")
        results['figures']['individual'] = []
        
        for e_idx, event_time in enumerate(events_to_plot):
            if verbose and (e_idx % 10 == 0 or e_idx == len(events_to_plot) - 1):
                log(f"Plotting event {e_idx+1}/{len(events_to_plot)}")
                
            # Find ripples within window
            window_ripples = ripple_times[(ripple_times >= event_time - window_size) & 
                                         (ripple_times <= event_time + window_size)]
            
            # Create figure
            fig = plt.figure(**default_fig_kwargs)
            ax = plt.gca()
            
            # Plot PGD curves for each band
            for band in bands:
                plt.plot(time_axis, results['pgd_data'][band]['values'][e_idx], 
                         color=colors[band], linewidth=2.5, 
                         label=f"{band.replace('narrow', '').replace('sharp', 'Sharp ')} Band")
            
            # Add ripple markers based on swr_marker parameter
            if swr_marker == 'line':
                # Original line markers
                for ripple_time in window_ripples:
                    time_lag = ripple_time - event_time
                    plt.axvline(x=time_lag, color=colors['ripple_line'], 
                               linestyle='-', alpha=0.6, linewidth=1.2)
            
            elif swr_marker == 'waveform':
                # Extract and plot waveforms
                y_min, y_max = ax.get_ylim()
                plot_range = y_max - y_min
                
                # Get all unique channels in this event
                event_channels = [ripple_channels[rt] for rt in window_ripples if rt in ripple_channels]
                unique_channels = sorted(set(event_channels))
                
                # Set up colormap
                try:
                    if waveform_colormap.startswith('cmc.'):
                        cmap = eval(waveform_colormap)
                    else:
                        cmap = plt.get_cmap(waveform_colormap)
                except:
                    cmap = plt.get_cmap('viridis')
                
                # Create normalization
                if len(unique_channels) > 1:
                    norm = plt.Normalize(vmin=min(unique_channels), vmax=max(unique_channels))
                else:
                    norm = plt.Normalize(vmin=0, vmax=1)
                
                # Create channel-to-y-position mapping
                if len(unique_channels) > 1:
                    y_start = y_min + 0.05 * plot_range
                    y_end = y_min + 0.70 * plot_range
                    channel_y_positions = {}
                    for i, channel in enumerate(unique_channels):
                        y_pos = y_start + (y_end - y_start) * i / max(1, len(unique_channels) - 1)
                        channel_y_positions[channel] = y_pos
                else:
                    channel_y_positions = {unique_channels[0]: y_min + 0.25 * plot_range}
                
                # Scale waveforms
                waveform_amplitude = plot_range * swr_waveform_height_scale
                
                event_waveforms = []
                
                for ripple_time in window_ripples:
                    time_lag = ripple_time - event_time
                    
                    if ripple_time in ripple_channels:
                        channel = ripple_channels[ripple_time]
                        
                        # Extract waveform
                        wf_time, wf_data = extract_ripple_waveform(ripple_time, channel)
                        
                        if wf_time is not None and wf_data is not None:
                            # Normalize and scale waveform
                            wf_data_norm = wf_data - np.mean(wf_data)
                            if np.std(wf_data_norm) > 0:
                                wf_data_norm = wf_data_norm / np.std(wf_data_norm)
                            
                            wf_data_scaled = wf_data_norm * waveform_amplitude
                            
                            channel_baseline = channel_y_positions[channel]
                            wf_data_positioned = wf_data_scaled + channel_baseline
                            
                            wf_time_scaled = wf_time * horizontal_scale_factor
                            wf_time_shifted = wf_time_scaled + time_lag
                            
                            waveform_color = cmap(norm(channel))
                            
                            plt.plot(wf_time_shifted, wf_data_positioned, 
                                    color=waveform_color, linewidth=1.2, alpha=0.9)
                            
                            event_waveforms.append({
                                'time': wf_time_shifted,
                                'data': wf_data_positioned,
                                'channel': channel,
                                'ripple_time': ripple_time,
                                'time_lag': time_lag,
                                'y_position': channel_baseline
                            })
                
                if swr_marker == 'waveform':
                    results['waveforms'][f'event_{e_idx}'] = event_waveforms
            
            # Add styling
            plt.grid(True, alpha=0.3, color=colors['gridlines'])
            plt.xlabel('Time from PGD Peak (s)', fontsize=12, color=colors['text'])
            plt.ylabel('Phase Gradient Directionality', fontsize=12, color=colors['text'])
            plt.title(f'Planar Wave Event at {event_time:.2f}s', fontsize=14, color=colors['text'])
            
            # Add legend
            plt.legend(loc='upper right', framealpha=0.9)
            
            # Add annotation about ripples
            ripple_info = f"Ripples: {len(window_ripples)}"
            if len(window_ripples) > 0:
                closest_ripple = window_ripples[np.argmin(np.abs(window_ripples - event_time))]
                closest_lag = closest_ripple - event_time
                ripple_info += f"\nClosest ripple: {closest_lag:.3f}s"
            
            plt.text(0.02, 0.98, ripple_info, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', color=colors['text'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save figure if requested
            if save_dir is not None:
                fig_path = os.path.join(save_dir, f'event_{e_idx+1:04d}_t{event_time:.2f}')
                save_figure_both_formats(fig, fig_path, dpi)
                plt.close(fig)
            
            results['figures']['individual'].append(fig)
    
    # Generate summary statistics
    if generate_summary:
        log("Generating summary statistics and plots")
        
        # Calculate overall statistics
        results['statistics'] = {
            'pgd_peaks_analyzed': len(pgd_peak_times),
            'ripple_events_total': len(ripple_times),
            'pgd_peaks_with_ripples': results['ripple_analysis']['events_with_ripples'],
            'pgd_peaks_with_ripples_pct': (results['ripple_analysis']['events_with_ripples'] / 
                                          max(1, len(events_to_plot)) * 100),
            'mean_pgd_values': {band: np.nanmax(results['pgd_data'][band]['mean']) 
                               for band in bands},
            'time_lag_mean': np.mean(results['ripple_analysis']['time_lags']) if 
                             len(results['ripple_analysis']['time_lags']) > 0 else None,
            'time_lag_median': np.median(results['ripple_analysis']['time_lags']) if 
                               len(results['ripple_analysis']['time_lags']) > 0 else None,
            'time_lag_std': np.std(results['ripple_analysis']['time_lags']) if 
                            len(results['ripple_analysis']['time_lags']) > 0 else None
        }
        
        # Calculate percentage of ripples occurring during PGD peaks
        ripple_in_planar_count = 0
        for ripple_time in ripple_times:
            for interval in sw_pgd_peaks['intervals']:
                if interval[0] <= ripple_time <= interval[1]:
                    ripple_in_planar_count += 1
                    break
        
        results['statistics']['ripples_during_planar_waves'] = ripple_in_planar_count
        results['statistics']['ripples_during_planar_waves_pct'] = (ripple_in_planar_count / 
                                                                  max(1, len(ripple_times)) * 100)
        
        # Create summary figure
        fig = plt.figure(figsize=(18, 10), dpi=100, facecolor=colors['background'])
        gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1])
        
        # Panel 1: Average PGD profiles
        ax1 = fig.add_subplot(gs[0, 0:2])
        for band in bands:
            ax1.plot(time_axis, results['pgd_data'][band]['mean'], 
                   color=colors[band], linewidth=3.0, 
                   label=f"{band.replace('narrow', '').replace('sharp', 'Sharp ')} Band")
            
            # Add shaded error region
            ax1.fill_between(time_axis, 
                           results['pgd_data'][band]['mean'] - results['pgd_data'][band]['std'],
                           results['pgd_data'][band]['mean'] + results['pgd_data'][band]['std'],
                           color=colors[band], alpha=0.2)
        
        # Add vertical line at t=0
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.7)
        
        # Add styling
        ax1.grid(True, alpha=0.3, color=colors['gridlines'])
        ax1.set_xlabel('Time from PGD Peak (s)', fontsize=12, color=colors['text'])
        ax1.set_ylabel('Phase Gradient Directionality', fontsize=12, color=colors['text'])
        ax1.set_title('Average PGD Profiles Around Planar Wave Events', fontsize=14, color=colors['text'])
        ax1.legend(loc='upper right', framealpha=0.9)
        
        # Create histogram of ripple event distribution
        if len(results['ripple_analysis']['time_lags']) > 0:
            # Panel 2: Histogram of time lags
            ax2 = fig.add_subplot(gs[0, 2])
            time_lags = np.array(results['ripple_analysis']['time_lags'])
            
            # Create KDE for smoothed distribution
            kde = gaussian_kde(time_lags)
            x = np.linspace(-window_size, window_size, 1000)
            y = kde(x)
            
            # Plot both histogram and KDE
            ax2.hist(time_lags, bins=25, color='#4682B4', alpha=0.5, density=True)
            ax2.plot(x, y, color='navy', linewidth=2)
            
            # Add vertical line at t=0
            ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            
            # Add styling
            ax2.grid(True, alpha=0.3, color=colors['gridlines'])
            ax2.set_xlabel('Time Lag (s)', fontsize=12, color=colors['text'])
            ax2.set_ylabel('Density', fontsize=12, color=colors['text'])
            ax2.set_title('Distribution of Time Lags Between\nPlanar Waves and Ripples', 
                        fontsize=14, color=colors['text'])
            
            # Add text with statistics
            stats_text = (
                f"Mean lag: {results['statistics']['time_lag_mean']:.3f}s\n"
                f"Median lag: {results['statistics']['time_lag_median']:.3f}s\n"
                f"Std dev: {results['statistics']['time_lag_std']:.3f}s"
            )
            ax2.text(0.03, 0.97, stats_text, transform=ax2.transAxes,
                   fontsize=10, verticalalignment='top', color=colors['text'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 3: Scatter plot of PGD values between bands
        ax3 = fig.add_subplot(gs[1, 0])
        if len(bands) >= 2:
            band1, band2 = bands[0], bands[1]
            # Extract max PGD value around each event for each band
            max_indices = np.argmax(results['pgd_data'][band1]['values'], axis=1)
            band1_max = np.array([results['pgd_data'][band1]['values'][i, max_indices[i]] 
                                 for i in range(len(events_to_plot))])
            band2_max = np.array([results['pgd_data'][band2]['values'][i, max_indices[i]] 
                                 for i in range(len(events_to_plot))])
            
            # Create scatter plot with color based on whether ripples were present
            has_ripple = np.zeros(len(events_to_plot), dtype=bool)
            for i, event_time in enumerate(events_to_plot):
                window_ripples = ripple_times[(ripple_times >= event_time - window_size) & 
                                             (ripple_times <= event_time + window_size)]
                has_ripple[i] = len(window_ripples) > 0
            
            ax3.scatter(band1_max[~has_ripple], band2_max[~has_ripple], 
                       alpha=0.7, color='gray', label='Without Ripples')
            ax3.scatter(band1_max[has_ripple], band2_max[has_ripple], 
                       alpha=0.7, color='red', label='With Ripples')
            
            # Add styling
            ax3.grid(True, alpha=0.3, color=colors['gridlines'])
            ax3.set_xlabel(f'{band1.replace("narrow", "").replace("sharp", "Sharp ")} PGD', 
                          fontsize=12, color=colors['text'])
            ax3.set_ylabel(f'{band2.replace("narrow", "").replace("sharp", "Sharp ")} PGD', 
                          fontsize=12, color=colors['text'])
            ax3.set_title('Correlation Between Band PGD Values', 
                         fontsize=14, color=colors['text'])
            ax3.legend(loc='upper left', framealpha=0.9)
            
            # Add correlation coefficient
            from scipy.stats import pearsonr
            r, p = pearsonr(band1_max, band2_max)
            ax3.text(0.03, 0.97, f"r = {r:.3f} (p = {p:.3f})", 
                    transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 4: Summary statistics as text
        ax4 = fig.add_subplot(gs[1, 1:])
        summary_text = (
            f"PLANAR WAVE AND RIPPLE ANALYSIS SUMMARY\n"
            f"SWR Marker Mode: {swr_marker.upper()}\n\n"
            f"Total planar wave events: {results['statistics']['pgd_peaks_analyzed']}\n"
            f"Total ripple events: {results['statistics']['ripple_events_total']}\n\n"
            f"Planar waves with ripples: {results['statistics']['pgd_peaks_with_ripples']} "
            f"({results['statistics']['pgd_peaks_with_ripples_pct']:.1f}%)\n"
            f"Ripples during planar waves: {results['statistics']['ripples_during_planar_waves']} "
            f"({results['statistics']['ripples_during_planar_waves_pct']:.1f}%)\n\n"
            f"Mean ripple time lag: {results['statistics']['time_lag_mean']:.3f}s\n"
            f"Median ripple time lag: {results['statistics']['time_lag_median']:.3f}s\n\n"
            f"Peak PGD values:\n" + 
            '\n'.join([f"  - {band.replace('narrow', '').replace('sharp', 'Sharp ')}: "
                       f"{results['statistics']['mean_pgd_values'][band]:.3f}" 
                       for band in bands])
        )
        
        ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
               transform=ax4.transAxes, color=colors['text'],
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax4.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save summary figure if requested
        if save_dir is not None:
            summary_path = os.path.join(save_dir, f'summary_analysis_{swr_marker}')
            save_figure_both_formats(fig, summary_path, dpi)
            log(f"Saved summary figure")
        
        results['figures']['summary'] = fig
        
        # Generate a text report
        report = f"""
        ==========================================================
        PLANAR WAVE AND SHARP WAVE-RIPPLE ANALYSIS REPORT
        ==========================================================
        
        Analysis Parameters:
        - Window size: {window_size:.2f} seconds
        - Smoothing sigma: {smoothing_sigma}
        - SWR marker type: {swr_marker}
        """
        
        if swr_marker == 'waveform':
            report += f"""
        - Waveform window: {swr_waveform_window:.3f} seconds
        - Waveform height scale: {swr_waveform_height_scale:.2f}
        - Waveform smoothing: {swr_waveform_smoothing:.1f}
            """
        
        report += f"""
        
        Dataset Information:
        - Total planar wave events detected: {results['statistics']['pgd_peaks_analyzed']}
        - Total ripple events detected: {results['statistics']['ripple_events_total']}
        - Ripple events per channel: {len(ripple_times) / max(1, len(channel_ripples)):.1f} (avg)
        
        Key Findings:
        1. Temporal Relationship:
           - {results['statistics']['pgd_peaks_with_ripples_pct']:.1f}% of planar waves coincide with ripple events
           - {results['statistics']['ripples_during_planar_waves_pct']:.1f}% of ripples occur during planar waves
           - Mean time lag: {results['statistics']['time_lag_mean']:.3f} seconds
           - Median time lag: {results['statistics']['time_lag_median']:.3f} seconds
           
        2. Band Relationships:
           - Peak PGD values (mean):
        """
        
        for band in bands:
            nice_name = band.replace('narrow', '').replace('sharp', 'Sharp ')
            report += f"     - {nice_name}: {results['statistics']['mean_pgd_values'][band]:.3f}\n"
        
        if len(bands) >= 2 and 'r' in locals():
            report += f"""
           - Correlation between {bands[0]} and {bands[1]}: r = {r:.3f} (p = {p:.3f})
            """
        
        report += """
        Interpretation:
        """
        
        # Add interpretations based on results
        if results['statistics']['pgd_peaks_with_ripples_pct'] > 75:
            report += """
        - Most planar waves are associated with ripple events, suggesting
          they may be part of the same network phenomenon.
            """
        elif results['statistics']['pgd_peaks_with_ripples_pct'] > 50:
            report += """
        - Many planar waves are associated with ripple events, suggesting
          a possible causal relationship that warrants further investigation.
            """
        else:
            report += """
        - Only some planar waves are associated with ripple events, suggesting
          these may be distinct network phenomena with occasional interactions.
            """
        
        # Add timing interpretation
        if results['statistics']['time_lag_mean'] is not None:
            if abs(results['statistics']['time_lag_mean']) < 0.05:
                report += """
        - The near-zero time lag suggests that planar waves and ripples occur
          simultaneously, possibly as part of the same neural event.
                """
            elif results['statistics']['time_lag_mean'] > 0:
                report += f"""
        - Ripples tend to follow planar waves by {results['statistics']['time_lag_mean']:.3f}s,
          suggesting that planar waves may trigger or facilitate ripple generation.
                """
            else:
                report += f"""
        - Ripples tend to precede planar waves by {-results['statistics']['time_lag_mean']:.3f}s,
          suggesting that ripples may trigger or facilitate planar wave formation.
                """
        
        if swr_marker == 'waveform':
            report += f"""
        
        3. Waveform Analysis:
           - {len([w for w in results['waveforms'].values() if w])} events had extractable waveforms
           - Waveforms provide morphological context for ripple-planar wave relationships
           - Individual waveform shapes can be analyzed for amplitude and duration patterns
            """
        
        report += """
        ==========================================================
        """
        
        results['report'] = report
        log("Summary report generated")
        
        # Print report if verbose
        if verbose:
            print(report)
    
    # Report total execution time
    execution_time = time.time() - start_time
    log(f"Analysis completed in {execution_time:.2f} seconds")
    
    return results


# Simplified visualize_all_swr_components (no changes needed)
def visualize_all_swr_components(lfp_processor, ripples, channel_subset=None, 
                              time_window=0.3, n_per_page=9, figsize=(15, 16), 
                              sort_by='peak_normalized_power', low_pass_cutoff=30,
                              dpi=100, save_path=None, waveform_smoothing_sigma=5.0,
                              include_waveform=True):
    """
    Visualize all sharp wave ripples with separate plots for sharp wave, ripple band components,
    and the combined waveform.
    """
    # Convert single channel to list if needed
    if channel_subset is not None and not isinstance(channel_subset, (list, tuple)):
        channel_subset = [channel_subset]
    
    # Collect all ripples from specified channels
    all_ripples = []
    
    for ch in ripples:
        if not isinstance(ch, (int, np.integer)) or ch == 'metadata':
            continue
            
        # Skip if we're filtering by channel and this channel isn't in the subset
        if channel_subset is not None and ch not in channel_subset:
            continue
            
        ch_ripples = ripples[ch]
        if len(ch_ripples) == 0:
            continue
            
        # Get channel info for ripples
        for idx, ripple in ch_ripples.iterrows():
            all_ripples.append({
                'channel': ch,
                'index': idx,
                'start_time': ripple['start_time'],
                'peak_time': ripple['peak_time'],
                'end_time': ripple['end_time'],
                'peak_normalized_power': ripple.get('peak_normalized_power', 1.0),
                'duration': ripple['end_time'] - ripple['start_time']
            })
    
    if not all_ripples:
        if channel_subset:
            channels_str = ", ".join(str(ch) for ch in channel_subset)
            print(f"No ripples found for channel(s) {channels_str}")
        else:
            print("No ripples found!")
        return {'figures': [], 'waveforms': []}
    
    # Sort ripples by the specified field
    if sort_by in all_ripples[0]:
        all_ripples.sort(key=lambda r: r.get(sort_by, 0), reverse=True)
    else:
        print(f"Warning: sort_by field '{sort_by}' not found. Using default order.")
    
    # Calculate number of pages needed
    total_ripples = len(all_ripples)
    pages = math.ceil(total_ripples / n_per_page)
    
    print(f"Visualizing {total_ripples} ripples across {pages} pages")
    
    # Design filter for sharp wave component
    fs = lfp_processor.fs
    nyquist = fs / 2
    b, a = butter(3, low_pass_cutoff / nyquist, btype='low')
    
    # Storage for waveform data
    waveform_data = []
    
    # Generate figures for each page
    figures = []
    
    for page in range(pages):
        start_idx = page * n_per_page
        end_idx = min(start_idx + n_per_page, total_ripples)
        page_ripples = all_ripples[start_idx:end_idx]
        
        # Create figure with 3 rows per ripple if including waveform, 2 otherwise
        n_ripples = len(page_ripples)
        rows_per_ripple = 3 if include_waveform else 2
        
        # Calculate grid dimensions
        n_cols = min(3, n_ripples)
        n_rows = math.ceil(n_ripples / n_cols) * rows_per_ripple
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, 
                               sharex='col', squeeze=False)
        
        # Process each ripple on this page
        for i, ripple in enumerate(page_ripples):
            # Calculate subplot positions
            col = i % n_cols
            row_base = (i // n_cols) * rows_per_ripple
            
            # Get channel info
            ch = ripple['channel']
            peak_time = ripple['peak_time']
            
            # Calculate window boundaries
            half_window = time_window / 2
            start_time = peak_time - half_window
            end_time = peak_time + half_window
            
            # Get indices in the signal
            start_sample = max(0, int(start_time * fs))
            end_sample = min(lfp_processor.waves['lfp'].shape[1], int(end_time * fs))
            
            # Extract signals
            lfp_signal = lfp_processor.waves['lfp'][ch, start_sample:end_sample]
            
            # Get ripple band signal
            if 'narrowRipples' in lfp_processor.waves:
                ripple_band = lfp_processor.waves['narrowRipples'][ch, start_sample:end_sample]
            elif 'broadRipples' in lfp_processor.waves:
                print(f"Warning: 'narrowRipples' not found. Using 'broadRipples' instead.")
                ripple_band = lfp_processor.waves['broadRipples'][ch, start_sample:end_sample]
            else:
                print(f"Warning: No ripple band data found. Using zeros.")
                ripple_band = np.zeros_like(lfp_signal)
            
            # Get sharp wave component by low-pass filtering LFP
            sharp_wave = filtfilt(b, a, lfp_signal)
            
            # Generate clean waveform from full LFP signal with smoothing
            combined_waveform = lfp_signal.copy()  # Use full LFP signal
            
            # Apply smoothing for clean, schematic-like appearance
            if waveform_smoothing_sigma > 0:
                combined_waveform_smooth = gaussian_filter1d(combined_waveform, 
                                                           sigma=waveform_smoothing_sigma)
            else:
                combined_waveform_smooth = combined_waveform.copy()
            
            # Store waveform data
            time_vector = np.arange(start_sample, end_sample) / fs - peak_time
            waveform_info = {
                'channel': ch,
                'peak_time': peak_time,
                'start_time': ripple['start_time'],
                'end_time': ripple['end_time'],
                'time_vector': time_vector,
                'sharp_wave': sharp_wave,
                'ripple_band': ripple_band,
                'combined_waveform': combined_waveform,
                'combined_waveform_smooth': combined_waveform_smooth,
                'sampling_rate': fs,
                'peak_power': ripple['peak_normalized_power'],
                'duration': ripple['duration']
            }
            waveform_data.append(waveform_info)
            
            # Create time vector for plotting (centered at ripple peak)
            time = time_vector
            
            # Define time markers for ripple boundaries
            start_marker = ripple['start_time'] - peak_time
            end_marker = ripple['end_time'] - peak_time
            
            # Plot sharp wave component (top subplot)
            ax_sw = axes[row_base, col]
            ax_sw.plot(time, sharp_wave, 'b-', linewidth=1.5, label='Sharp wave (<30 Hz)')
            
            # Mark ripple boundaries
            ax_sw.axvline(start_marker, color='g', linestyle='--', alpha=0.7, label='Start')
            ax_sw.axvline(0, color='r', linestyle='-', linewidth=2, alpha=0.8, label='Peak')
            ax_sw.axvline(end_marker, color='m', linestyle='--', alpha=0.7, label='End')
            
            # Configure top plot
            if i == 0:
                ax_sw.legend(fontsize=7, loc='upper right')
            ax_sw.set_title(f"Ch {ch}: Sharp Wave Component\nPower: {ripple['peak_normalized_power']:.2f}", 
                           fontsize=9)
            ax_sw.set_ylabel('Amplitude (μV)', fontsize=8)
            ax_sw.grid(True, alpha=0.3)
            ax_sw.tick_params(labelsize=7)
            
            # Plot ripple band component (middle subplot)
            ax_ripple = axes[row_base + 1, col]
            ax_ripple.plot(time, ripple_band, 'r-', linewidth=1, label='Ripple (150-250 Hz)')
            
            # Mark ripple boundaries
            ax_ripple.axvline(start_marker, color='g', linestyle='--', alpha=0.7)
            ax_ripple.axvline(0, color='r', linestyle='-', linewidth=2, alpha=0.8)
            ax_ripple.axvline(end_marker, color='m', linestyle='--', alpha=0.7)
            
            # Configure middle plot
            ax_ripple.set_ylabel('Amplitude (μV)', fontsize=8)
            ax_ripple.grid(True, alpha=0.3)
            ax_ripple.tick_params(labelsize=7)
            
            # Add duration information
            duration_ms = ripple['duration'] * 1000
            ax_ripple.set_title(f"Ripple Band Component\nDuration: {duration_ms:.1f} ms", fontsize=9)
            
            # Plot combined waveform (bottom subplot) if requested
            if include_waveform:
                ax_combined = axes[row_base + 2, col]
                
                # Plot the clean, smoothed LFP waveform
                ax_combined.plot(time, combined_waveform_smooth, 'k-', linewidth=1.8, 
                               label='Clean waveform')
                
                # Mark only ripple boundaries (no peak line for cleaner look)
                ax_combined.axvline(start_marker, color='g', linestyle='--', alpha=0.7)
                ax_combined.axvline(end_marker, color='m', linestyle='--', alpha=0.7)
                
                # Highlight the ripple period with subtle background
                ripple_mask = (time >= start_marker) & (time <= end_marker)
                if np.any(ripple_mask):
                    y_min, y_max = ax_combined.get_ylim() if ax_combined.get_ylim() != (0, 1) else (np.min(combined_waveform_smooth), np.max(combined_waveform_smooth))
                    ax_combined.fill_between(time, y_min, y_max,
                                           where=ripple_mask, alpha=0.1, color='yellow',
                                           label='Ripple period')
                
                # Configure bottom plot for clean appearance
                ax_combined.set_xlabel('Time from peak (s)', fontsize=8)
                ax_combined.set_ylabel('Amplitude (μV)', fontsize=8)
                ax_combined.grid(True, alpha=0.3)
                ax_combined.tick_params(labelsize=7)
                
                # Add waveform statistics
                max_amp = np.max(np.abs(combined_waveform_smooth))
                ax_combined.set_title(f"Clean Waveform\nMax amplitude: {max_amp:.1f} μV", fontsize=9)
                
                if i == 0:
                    ax_combined.legend(fontsize=7, loc='upper right')
            else:
                # If not including waveform, make the ripple plot the bottom one with xlabel
                ax_ripple.set_xlabel('Time from peak (s)', fontsize=8)
        
        # Remove empty subplots
        total_subplots = n_rows * n_cols
        used_subplots = len(page_ripples) * rows_per_ripple
        for i in range(used_subplots, total_subplots):
            row = i // n_cols
            col = i % n_cols
            if row < n_rows:  # Safety check
                fig.delaxes(axes[row, col])
        
        # Add page information
        waveform_text = " with Waveforms" if include_waveform else ""
        fig.suptitle(f'Sharp Wave Ripple Components{waveform_text} - Page {page+1}/{pages}\n' + 
                   f'Ripples {start_idx+1}-{end_idx} of {total_ripples}', 
                   fontsize=12, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        
        # Save figure if requested
        if save_path:
            base = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
            page_path = f"{base}_page{page+1}"
            save_figure_both_formats(fig, page_path, dpi)
            print(f"Saved page {page+1}")
        
        figures.append(fig)
    
    print(f"Generated {len(waveform_data)} waveforms from {total_ripples} ripples")
    
    return {
        'figures': figures, 
        'waveforms': waveform_data
    }


# Modified plot_pgd_wave_analysis to use precomputed data
def plot_pgd_wave_analysis_optimized(
    lfp_processor,
    sharp_wave_pgd,
    pgd_data,  # New parameter: precomputed PGD data
    data_type='sharpWave',
    speed_range=None,
    n_bins_speed=100,
    n_bins_direction=36,
    colormap='cmc.lapaz',
    fig_width=8,
    fig_height=8,
    compute_kde=True,
    title=None,
    save_path_base=None,
    dpi=300,
    show_stats=True,
    use_joypy=False
):
    """
    Optimized version of plot_pgd_wave_analysis using precomputed PGD data.
    
    Additional Parameters:
    ---------------------
    pgd_data : dict
        Precomputed PGD data from compute_pgd_for_window
    """
    # Try to import crameri colormaps
    try:
        if colormap.startswith('cmc.'):
            cmap = eval(colormap)
        else:
            cmap = plt.get_cmap(colormap)
    except (ImportError, NameError):
        print("cmcrameri package not found or invalid colormap. Using viridis.")
        cmap = plt.get_cmap('viridis')
    
    # Check if joypy is available for ridgeline plots
    if use_joypy:
        try:
            import joypy
            have_joypy = True
        except ImportError:
            print("joypy package not found. Creating custom ridgeline implementation.")
            have_joypy = False
    else:
        have_joypy = False
    
    # Helper function to create custom ridgeline plots if joypy is not available
    def custom_ridgeline(data_list, labels=None, colormap=None, ax=None, 
                       figsize=(10, 6), x_range=None, overlap=0.8, alpha=0.8, 
                       kde=True, bw_method=0.2, fill=True, grid=True,
                       linewidth=1.0):
        """Create custom ridgeline plot (a.k.a. joy plot) WITHOUT mean lines"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
            
        # Determine x_range if not provided
        if x_range is None:
            x_min = min([np.min(d) for d in data_list if len(d) > 0])
            x_max = max([np.max(d) for d in data_list if len(d) > 0])
            margin = (x_max - x_min) * 0.05
            x_range = (x_min - margin, x_max + margin)
        
        n_items = len(data_list)
        
        # Create a colormap if not provided
        if colormap is None:
            colormap = plt.cm.viridis
            
        if isinstance(colormap, str):
            colormap = plt.get_cmap(colormap)
        
        # Calculate heights and distributions
        max_heights = []
        distributions = []
        
        # Generate distributions and find maximum heights
        for i, data in enumerate(data_list):
            if len(data) > 0:
                # Create density distribution
                if kde:
                    # Create kernel density estimate
                    kernel = scipy_stats.gaussian_kde(data, bw_method=bw_method)
                    x = np.linspace(x_range[0], x_range[1], 500)
                    y = kernel(x)
                else:
                    # Create histogram
                    hist, bin_edges = np.histogram(data, bins=100, range=x_range, density=True)
                    x = (bin_edges[:-1] + bin_edges[1:]) / 2
                    y = hist
                
                # Store distribution
                distributions.append((x, y))
                max_heights.append(np.max(y))
            else:
                distributions.append(([], []))
                max_heights.append(0)
        
        # Normalize heights
        if max_heights:
            max_height = max(max_heights)
            height_scale = 1.0 / max_height if max_height > 0 else 1.0
        else:
            height_scale = 1.0
            
        # Scale factor for vertical spacing
        spacing = (1.0 - overlap) / (n_items - 1) if n_items > 1 else 0
        
        # Plot each distribution
        for i, (x, y) in enumerate(distributions):
            if len(x) > 0:
                # Calculate vertical position
                y_pos = i * spacing
                
                # Scale y values
                y_scaled = y * height_scale * overlap
                
                # Get color
                color = colormap(i / max(1, n_items - 1))
                
                # Plot the distribution
                ax.plot(x, y_scaled + y_pos, color=color, linewidth=linewidth, zorder=n_items - i + 1)
                
                if fill:
                    ax.fill_between(x, y_pos, y_scaled + y_pos, color=color, alpha=alpha, zorder=n_items - i)
        
        # Set labels if provided
        if labels is not None:
            # Set y-ticks at the baseline of each distribution
            ax.set_yticks([i * spacing for i in range(n_items)])
            ax.set_yticklabels(labels)
        else:
            # Hide y-axis
            ax.set_yticks([])
        
        # Set x-axis range
        ax.set_xlim(x_range)
        
        # Set y-axis range
        ax.set_ylim(-0.02, 1.0 if n_items <= 1 else (n_items - 1) * spacing + overlap)
        
        # Add grid if requested
        if grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        return fig, ax
    
    # Validate input data
    if not isinstance(sharp_wave_pgd, dict) or 'peak_times' not in sharp_wave_pgd:
        raise ValueError("Invalid sharp_wave_pgd dictionary or no PGD peak times found")
    
    # Initialize results structure
    results = {
        'events': [],
        'speeds': [],
        'directions': [],
        'statistics': []
    }
    
    # Get sampling frequency
    fs = lfp_processor.fs
    
    # Use gradient calculator from precomputed data
    grad_calc = pgd_data['grad_calc']
    config = grad_calc.cfg
    
    # Calculate speeds and directions for each PGD event
    n_events = len(sharp_wave_pgd['peak_times'])
    
    for i, peak_time in enumerate(sharp_wave_pgd['peak_times']):
        # Define window around peak
        if 'intervals' in sharp_wave_pgd and i < len(sharp_wave_pgd['intervals']):
            start_time, end_time = sharp_wave_pgd['intervals'][i]
        else:
            # Default window of 100ms around peak
            start_time = peak_time - 0.05
            end_time = peak_time + 0.05
        
        # Convert times to frames relative to precomputed data window
        window_start_frame = pgd_data['window_start']
        start_frame = max(0, int(start_time * fs) - window_start_frame)
        end_frame = min(pgd_data['phase_data'].shape[1], int(end_time * fs) - window_start_frame)
        peak_frame = int(peak_time * fs) - window_start_frame
        
        # Get phase data from precomputed results
        phases = pgd_data['phase_data'][:, start_frame:end_frame]
        
        # Get target frame (corresponding to peak)
        target_frame = min(peak_frame - start_frame, phases.shape[1] - 1)
        target_frame = max(0, target_frame)
        
        # Extract phase at target time
        phase_at_time = phases[:, target_frame]
        
        # Compute spatial gradients
        grad_x, grad_y = grad_calc.compute(phase_at_time)
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        
        # Compute temporal derivatives
        if phases.shape[1] > 2 and target_frame > 0 and target_frame < phases.shape[1] - 1:
            dt = 1.0 / fs
            dphase_dt = (phases[:, min(target_frame + 1, phases.shape[1] - 1)] - 
                        phases[:, max(target_frame - 1, 0)]) / (2 * dt)
        else:
            # For edge cases
            dt = 1.0 / fs
            if target_frame == 0 and phases.shape[1] > 1:
                dphase_dt = (phases[:, 1] - phases[:, 0]) / dt
            elif target_frame == phases.shape[1] - 1 and phases.shape[1] > 1:
                dphase_dt = (phases[:, target_frame] - phases[:, target_frame - 1]) / dt
            else:
                # Single time point
                dphase_dt = np.ones_like(grad_norm) * (2 * np.pi * 10)  # Assume ~10 Hz
        
        # Compute wave speeds
        speeds = np.abs(dphase_dt / np.maximum(grad_norm, config['MIN_GRADIENT']))
        
        # Apply masks for valid speeds
        mask = (speeds >= config['V_MIN']) & (speeds <= config['V_MAX']) & (grad_norm >= config['MIN_GRADIENT'])
        valid_speeds = speeds[mask]
        
        # Compute wave propagation directions
        wave_directions = np.arctan2(-grad_y, -grad_x)
        valid_directions = wave_directions[mask]
        
        # Convert to range [0, 2*pi]
        valid_directions = np.mod(valid_directions, 2*np.pi)
        
        # Store valid measurements
        results['speeds'].append(valid_speeds)
        results['directions'].append(valid_directions)
        
        # Store event info
        results['events'].append({
            'peak_time': peak_time,
            'start_frame': start_frame + window_start_frame,
            'end_frame': end_frame + window_start_frame,
            'target_frame': target_frame
        })
        
        # Calculate statistics
        event_stats = {
            'peak_time': peak_time,
            'n_valid_speeds': len(valid_speeds),
            'mean_speed': np.mean(valid_speeds) if len(valid_speeds) > 0 else None,
            'median_speed': np.median(valid_speeds) if len(valid_speeds) > 0 else None,
            'std_speed': np.std(valid_speeds) if len(valid_speeds) > 0 else None,
            'n_valid_directions': len(valid_directions),
            'circular_mean': None,
            'circular_std': None
        }
        
        # Calculate circular statistics if we have valid directions
        if len(valid_directions) > 0:
            sin_sum = np.sum(np.sin(valid_directions))
            cos_sum = np.sum(np.cos(valid_directions))
            mean_angle = np.arctan2(sin_sum, cos_sum)
            R = np.sqrt(sin_sum**2 + cos_sum**2) / len(valid_directions)
            circular_std = np.sqrt(-2 * np.log(R))
            
            event_stats['circular_mean'] = mean_angle
            event_stats['circular_std'] = circular_std
        
        results['statistics'].append(event_stats)

    # Create separate figures for each plot
    figs = {}
    axes = {}
    
    # Determine speed range if not provided
    if speed_range is None and any(len(s) > 0 for s in results['speeds']):
        speeds_all = np.concatenate([s for s in results['speeds'] if len(s) > 0])
        min_speed = max(0, np.percentile(speeds_all, 1))
        max_speed = min(np.percentile(speeds_all, 99), config['V_MAX'])
        speed_range = (min_speed, max_speed)
    elif speed_range is None:
        speed_range = (0, 100_000)

    # 1. RIDGELINE PLOT FOR SPEEDS
    fig_speed = plt.figure(figsize=(fig_width, fig_height))
    ax_speed = fig_speed.add_subplot(111)
    figs['speed'] = fig_speed
    axes['speed'] = ax_speed
    
    # Create labels for ridgeline plots
    event_labels = [f"Event {i+1} (t={results['events'][i]['peak_time']:.2f}s)" 
                   for i in range(n_events)]
    
    if have_joypy:
        # Convert to DataFrame for joypy
        import pandas as pd
        speed_data = pd.DataFrame()
        for i, speeds in enumerate(results['speeds']):
            if len(speeds) > 0:
                speed_data[f"Event {i+1}"] = pd.Series(speeds)
        
        if not speed_data.empty:
            # Close the current figure to prevent joypy from clearing it
            plt.close(fig_speed)
            
            # Let joypy create its own figure
            joypy.joyplot(
                speed_data,
                figsize=(fig_width, fig_height),
                colormap=cmap,
                overlap=0.05,
                alpha=0.6,
                range_style='own',
                x_range=speed_range,
                bins=n_bins_speed
            )
            
            # Get the current figure that joypy created
            figs['speed'] = plt.gcf()
            axes['speed'] = plt.gca()
            
            # Set title
            plt.title(f'{data_type.upper()} Wave Speed Distributions', fontsize=14)
            plt.xlabel('Wave Speed (μm/s)', fontsize=12)
    else:
        # Use custom implementation
        custom_ridgeline(
            results['speeds'],
            labels=event_labels,
            colormap=cmap,
            ax=ax_speed,
            x_range=speed_range,
            overlap=0.8,
            alpha=0.6,
            kde=compute_kde
        )
        
        # Set speed axis labels
        ax_speed.set_xlabel('Wave Speed (μm/s)', fontsize=12)
        ax_speed.set_title(f'{data_type.upper()} Wave Speed Distributions', fontsize=14)

    # 2. POLAR PLOT FOR DIRECTIONS
    fig_direction = plt.figure(figsize=(fig_width, fig_height))
    ax_direction = fig_direction.add_subplot(111, projection='polar')
    figs['direction'] = fig_direction
    axes['direction'] = ax_direction
    
    # Configure polar plot for directions
    ax_direction.set_theta_zero_location('E')
    ax_direction.set_theta_direction(-1)
    
    # Create radial labels
    ax_direction.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_direction.tick_params(axis='both', which='major', labelsize=9)
    
    # Set up radian labels
    ax_direction.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
    ax_direction.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'])
    
    # Store all histogram data to normalize across all events
    all_hist_data = []
    mean_angles = []
    
    # First collect all histogram data for normalization
    for i, directions in enumerate(results['directions']):
        if len(directions) > 0:
            # Define bin edges for the histogram (in radians)
            bin_edges = np.linspace(0, 2*np.pi, n_bins_direction+1)
            
            # Create histogram
            hist, _ = np.histogram(directions, bins=bin_edges, density=True)
            all_hist_data.append(hist)
            
            # Store mean angle if available
            if results['statistics'][i]['circular_mean'] is not None:
                mean_angles.append((i, results['statistics'][i]['circular_mean']))
    
    # Find global max for normalization
    global_max = max([np.max(h) for h in all_hist_data]) if all_hist_data else 1.0
    
    # Now plot the normalized histograms
    for i, directions in enumerate(results['directions']):
        if len(directions) > 0:
            # Define bin edges for the histogram (in radians)
            bin_edges = np.linspace(0, 2*np.pi, n_bins_direction+1)
            
            # Get bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Calculate bar width
            width = 2 * np.pi / n_bins_direction
            
            # Create histogram
            hist, _ = np.histogram(directions, bins=bin_edges, density=True)
            
            # Normalize by global max
            hist_norm = hist / global_max
            
            # Get color
            color = cmap(i / max(1, n_events - 1))
            
            # Plot bars
            ax_direction.bar(bin_centers, hist_norm, width=width, alpha=0.6, color=color,
                           edgecolor='none', label=f"Event {i+1}")
    
    # Add mean direction arrows with the same color as their event
    for event_idx, mean_angle in mean_angles:
        # Get event color
        color = cmap(event_idx / max(1, n_events - 1))
        
        # Add arrow
        ax_direction.annotate('',
                            xy=(mean_angle, 1.0),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', 
                                          color=color,
                                          lw=2,
                                          alpha=0.9,
                                          zorder=10),
                            xycoords='data')
    
    ax_direction.set_title(f'{data_type.upper()} Wave Direction Distributions', fontsize=14)
    
    # Add direction legend with custom positioning
    if n_events > 0:
        # Limit to last 5 events for legend
        legend_count = min(5, n_events)
        legend_indices = range(n_events - legend_count, n_events)
        legend_labels = [f"Event {i+1}" for i in legend_indices]
        legend_handles = [plt.Rectangle((0,0), 1, 1, color=cmap(i / max(1, n_events - 1))) 
                         for i in legend_indices]
        ax_direction.legend(legend_handles, legend_labels, 
                         loc='upper right', bbox_to_anchor=(1.3, 1.0),
                         fontsize=8)

    # 3. BOXPLOT FOR SPEEDS
    fig_boxplot = plt.figure(figsize=(fig_width, fig_height))
    ax_boxplot = fig_boxplot.add_subplot(111)
    figs['boxplot'] = fig_boxplot
    axes['boxplot'] = ax_boxplot
    
    if any(len(s) > 0 for s in results['speeds']):
        # Prepare data for boxplot
        boxplot_data = [s for s in results['speeds'] if len(s) > 0]
        boxplot_labels = [f"E{i+1}" for i, s in enumerate(results['speeds']) if len(s) > 0]
        
        # Create positions for colored boxes
        positions = np.arange(1, len(boxplot_data) + 1)
        
        # Create boxplot
        bp = ax_boxplot.boxplot(boxplot_data, positions=positions, patch_artist=True, 
                               widths=0.6, showfliers=False)
        
        # Customize boxplot colors to match the colormap
        for i, box in enumerate(bp['boxes']):
            color = cmap(i / max(1, len(boxplot_data) - 1))
            box.set(facecolor=color, alpha=0.6)
            bp['medians'][i].set(color='black', linewidth=1.5)
        
        # Set labels and title
        ax_boxplot.set_xlabel('Event Number', fontsize=12)
        ax_boxplot.set_ylabel('Wave Speed (μm/s)', fontsize=12)
        ax_boxplot.set_title(f'{data_type.upper()} Wave Speeds', fontsize=14)
        
        # Set x-tick labels
        ax_boxplot.set_xticks(positions)
        ax_boxplot.set_xticklabels(boxplot_labels)
        
        # Set y-limits to match ridgeline plot
        ax_boxplot.set_ylim(speed_range)
        
        # Add grid
        ax_boxplot.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add summary statistics
        if show_stats:
            all_valid_speeds = [s for speeds in results['speeds'] for s in speeds if np.isfinite(s)]
            if all_valid_speeds:
                mean_speed = np.mean(all_valid_speeds)
                median_speed = np.median(all_valid_speeds)
                std_speed = np.std(all_valid_speeds)
                
                stats_text = (
                    f"All Events:\n"
                    f"Mean: {mean_speed:.0f} μm/s\n"
                    f"Median: {median_speed:.0f} μm/s\n"
                    f"Std: {std_speed:.0f} μm/s"
                )
                
                ax_boxplot.text(0.95, 0.05, stats_text, transform=ax_boxplot.transAxes,
                               fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # No valid data for boxplot
        ax_boxplot.text(0.5, 0.5, "Insufficient data for boxplot", 
                      ha='center', va='center', fontsize=12)
        ax_boxplot.set_xlabel('Event Number', fontsize=12)
        ax_boxplot.set_ylabel('Wave Speed (μm/s)', fontsize=12)
        ax_boxplot.set_title(f'{data_type.upper()} Wave Speeds', fontsize=14)
    
    # 4. MEAN DIRECTION PLOT ACROSS ALL EVENTS
    # Calculate mean direction across all events using circular statistics
    all_mean_directions = [angle for _, angle in mean_angles]
    
    # Only proceed if we have mean directions
    if all_mean_directions:
        # Calculate circular mean of means
        sin_sum = np.sum(np.sin(all_mean_directions))
        cos_sum = np.sum(np.cos(all_mean_directions))
        global_mean_angle = np.arctan2(sin_sum, cos_sum)
        
        # Calculate circular R value
        R = np.sqrt(sin_sum**2 + cos_sum**2) / len(all_mean_directions)
        circular_std = np.sqrt(-2 * np.log(R))
        
        # Add global mean and std to results
        results['global_mean_direction_rad'] = global_mean_angle
        results['global_mean_direction_deg'] = np.degrees(global_mean_angle)
        results['global_direction_concentration'] = R
        results['global_direction_std_rad'] = circular_std
        
        # Print the mean direction
        print(f"\nGlobal Mean Direction: {global_mean_angle:.4f} radians = {np.degrees(global_mean_angle):.2f}°")
        print(f"Direction Concentration: {R:.4f}")
        print(f"Circular Standard Deviation: {circular_std:.4f} radians = {np.degrees(circular_std):.2f}°")
        
        # Create figure for mean direction plot
        fig_mean_dir = plt.figure(figsize=(fig_width, fig_height))
        ax_mean_dir = fig_mean_dir.add_subplot(111, projection='polar')
        figs['mean_direction'] = fig_mean_dir
        axes['mean_direction'] = ax_mean_dir
        
        # Configure polar plot for directions
        ax_mean_dir.set_theta_zero_location('E')
        ax_mean_dir.set_theta_direction(-1)
        
        # Create radial labels
        ax_mean_dir.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_mean_dir.tick_params(axis='both', which='major', labelsize=9)
        
        # Set up radian labels
        ax_mean_dir.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        ax_mean_dir.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'])
        
        # Plot the same histograms as in the individual direction plot
        for i, directions in enumerate(results['directions']):
            if len(directions) > 0:
                bin_edges = np.linspace(0, 2*np.pi, n_bins_direction+1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                width = 2 * np.pi / n_bins_direction
                
                hist, _ = np.histogram(directions, bins=bin_edges, density=True)
                hist_norm = hist / global_max  # Use same normalization as before
                
                color = cmap(i / max(1, n_events - 1))
                ax_mean_dir.bar(bin_centers, hist_norm, width=width, alpha=0.6, color=color,
                               edgecolor='none')
        
        # Add single black arrow for global mean direction
        ax_mean_dir.annotate('',
                           xy=(global_mean_angle, 0.8),
                           xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->',
                                         color='black',
                                         lw=3,
                                         alpha=1.0,
                                         zorder=20),
                           xycoords='data')
        
        # Add text for mean direction
        text_angle = global_mean_angle
        text_radius = 0.9
        
        # Format the angle in terms of π for the label
        angle_in_pi = global_mean_angle / np.pi
        
        # Format nicely as a fraction or multiple of π
        if np.isclose(angle_in_pi, 0) or np.isclose(angle_in_pi, 2):
            angle_text = "0"
        elif np.isclose(angle_in_pi, 1):
            angle_text = "π"
        elif np.isclose(angle_in_pi, 0.5):
            angle_text = "π/2"
        elif np.isclose(angle_in_pi, 1.5):
            angle_text = "3π/2"
        elif np.isclose(angle_in_pi % 1, 0):
            angle_text = f"{int(angle_in_pi)}π"
        elif np.isclose(angle_in_pi % 0.25, 0):
            numerator = int(angle_in_pi * 4)
            if numerator % 4 == 0:
                angle_text = f"{numerator // 4}π"
            else:
                from math import gcd
                denominator = 4
                divisor = gcd(numerator, denominator)
                numerator //= divisor
                denominator //= divisor
                angle_text = f"{numerator}π/{denominator}"
        else:
            angle_text = f"{global_mean_angle:.2f}"
        
        # Add text annotation with mean direction in radians
        ax_mean_dir.text(text_angle, text_radius, 
                        angle_text,
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_mean_dir.set_title(f'{data_type.upper()} Global Mean Direction', fontsize=14)
        
        # Add legend on the right side
        if n_events > 0:
            legend_count = min(5, n_events)
            legend_indices = range(n_events - legend_count, n_events)
            legend_labels = [f"Event {i+1}" for i in legend_indices]
            legend_handles = [plt.Rectangle((0,0), 1, 1, color=cmap(i / max(1, n_events - 1))) 
                            for i in legend_indices]
            ax_mean_dir.legend(legend_handles, legend_labels, 
                            loc='upper right', bbox_to_anchor=(1.3, 1.0),
                            fontsize=8)

    # Set titles for all plots if provided
    if title is not None:
        for fig in figs.values():
            fig.suptitle(title, fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figures if requested
    if save_path_base is not None:
        # Create directory if it doesn't exist
        if os.path.dirname(save_path_base) and not os.path.exists(os.path.dirname(save_path_base)):
            os.makedirs(os.path.dirname(save_path_base))
        
        # Save each figure with appropriate suffix
        save_figure_both_formats(figs['speed'], f"{save_path_base}_speed", dpi)
        save_figure_both_formats(figs['direction'], f"{save_path_base}_direction", dpi)
        save_figure_both_formats(figs['boxplot'], f"{save_path_base}_boxplot", dpi)
        if 'mean_direction' in figs:
            save_figure_both_formats(figs['mean_direction'], f"{save_path_base}_mean_direction", dpi)
        print(f"Figures saved to {save_path_base}_*.png and *.svg")
    
    return figs, axes, results


# Main script execution
if __name__ == "__main__":
    # Paths
    folder_dir = "D:\\dvorak\\" 
    raw_path = f"{folder_dir}dvorak_3.raw.h5"
    spike_path = f"{folder_dir}dvorak_3_acqm.zip" 
    lfp_path = "C:\\Users\\visio\\OneDrive\\Desktop\\special_bandsdvorak_3_special_bands_20_[0, 120].npz"
    
    # Curation_data
    train, neuron_data, config, fs = load_curation(spike_path)
    train = [np.array(t)*1000 for t in train]
    spike_data = analysis.SpikeData(train, neuron_data={0: neuron_data})
    
    # Raw data
    version, time_stamp, config_df, raster_df = load_info_maxwell(raw_path)
    
    # lfp data
    waves = np.load(lfp_path)
    
    # Initialize your LFPDataProcessor object
    x_mod = np.load("D:\\testing_new_lfp_class\\wave_prop\\ladybugs\\ai_examples\\figures_code\\scripts\\x_mod.npy")
    y_mod = np.load("D:\\testing_new_lfp_class\\wave_prop\\ladybugs\\ai_examples\\figures_code\\scripts\\y_mod.npy")
    processor = LFPDataProcessor(waves, x_mod, y_mod, config_df)
    processor.add_frequency_band(1, 30, band_name="sharpWave", use_gpu=True, store_analytical=True)
    
    # opto_intervals
    opto_intervals = [(26.5, 27), (30, 40), (60, 70), (90, 100.5)]
    
    # Set window parameters
    window_start = 14000
    window_length = 44000
    
    # COMPUTE PGD ONCE FOR EACH BAND
    print("Computing PGD for Sharp Wave band...")
    sharp_wave_pgd_data = compute_pgd_for_window(
        processor,
        data_type='sharpWave',
        window_start=window_start,
        window_length=window_length,
        smoothing_sigma=20,
        min_gradient=1e-5,
        use_gpu=True,
        batch_size=100,
        verbose=True
    )
    
    print("\nComputing PGD for Narrow Ripple band...")
    narrow_ripple_pgd_data = compute_pgd_for_window(
        processor,
        data_type='narrowRipples',
        window_start=window_start,
        window_length=window_length,
        smoothing_sigma=15,
        min_gradient=1e-5,
        use_gpu=True,
        batch_size=100,
        verbose=True
    )
    
    # Store precomputed PGD data
    pgd_data_dict = {
        'sharpWave': sharp_wave_pgd_data,
        'narrowRipples': narrow_ripple_pgd_data
    }
    
    # Detect PGD peaks using precomputed data
    print("\nDetecting planar waves in Sharp Wave band...")
    sharp_wave_pgd = detect_pgd_peaks_from_precomputed(
        sharp_wave_pgd_data,
        threshold=1.4,
        min_duration=0.1,
        plot_results=True,
        color='blue',
        highlight_color='lightskyblue',
        save_path='sharp_wave_pgd_peaks'
    )

    print("\nDetecting planar waves in Narrow Ripple band...")
    narrow_ripple_pgd = detect_pgd_peaks_from_precomputed(
        narrow_ripple_pgd_data,
        threshold=1.5,
        min_duration=0.05,
        plot_results=True,
        color='red',
        highlight_color='salmon',
        save_path='narrow_ripple_pgd_peaks'
    )
    
    # Detect ripple events
    ripple_events = processor.detect_ripples(
        narrowband_key='narrowRipples',
        wideband_key='broadRipples',
        low_threshold=3.5,
        high_threshold=5,
        min_duration=20,
        max_duration=200,
        min_interval=20,
        sharp_wave_threshold=3,
        sharp_wave_band=(0.1, 30),
        require_sharp_wave=True
    )
    
    # Run the comprehensive analysis with precomputed data
    results = analyze_planar_waves_and_ripples_optimized(
        processor,
        sharp_wave_pgd,
        ripple_events,
        pgd_data_dict,  # Pass precomputed PGD data
        bands=['sharpWave', 'narrowRipples'],
        window_size=0.5,
        save_dir='planar_wave_analysis_alt',
        smoothing_sigma=15,
        verbose=True,
        swr_marker='waveform',
        swr_waveform_window=0.1,
        swr_waveform_height_scale=0.013,
        horizontal_scale_factor=0.2,
    )

    # Access the summary report
    print(results['report'])

    # The key statistics are available in:
    print(f"Percentage of planar waves with ripples: {results['statistics']['pgd_peaks_with_ripples_pct']:.1f}%")
    print(f"Percentage of ripples during planar waves: {results['statistics']['ripples_during_planar_waves_pct']:.1f}%")
    
    # Visualize SWR components
    result = visualize_all_swr_components(
        processor, 
        ripple_events,
        time_window=0.3,
        waveform_smoothing_sigma=0.01,
        include_waveform=True,
        save_path="swr_analysis_with_waveforms",
    )

    # Access the waveform data
    figures = result['figures']
    waveforms = result['waveforms']

    # Each waveform contains:
    for waveform in waveforms[:3]:  # Print first 3
        print(f"Channel {waveform['channel']}: Peak at {waveform['peak_time']:.3f}s")
    
    # Plot wave analysis using precomputed data
    figs, axes, wave_results = plot_pgd_wave_analysis_optimized(
        processor,
        sharp_wave_pgd,
        sharp_wave_pgd_data,  # Pass precomputed PGD data
        data_type='sharpWave',
        colormap='cmc.lapaz',
        save_path_base='theta_pgd_analysis',
        use_joypy=False,
        fig_width=8,
        fig_height=8
    )

    # Display the polar plot specifically
    plt.figure(figs['direction'].number)
    plt.show()

    # Print summary statistics
    avg_speeds = [stats['mean_speed'] for stats in wave_results['statistics'] if stats['mean_speed'] is not None]
    if avg_speeds:
        print(f"Average wave speed across all events: {np.mean(avg_speeds):.2f} μm/s")
        
        # Find the fastest and slowest events
        fastest_idx = np.argmax(avg_speeds)
        slowest_idx = np.argmin(avg_speeds)
        
        print(f"Fastest wave event: Event {fastest_idx+1} at {avg_speeds[fastest_idx]:.2f} μm/s")
        print(f"Slowest wave event: Event {slowest_idx+1} at {avg_speeds[slowest_idx]:.2f} μm/s")