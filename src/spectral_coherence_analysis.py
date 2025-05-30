#!/usr/bin/env python3
"""
Spectral Coherence Analysis for Hippocampal LFP Data

This script analyzes coherent local field potential structures in hippocampal recordings,
focusing on sharp waves and ripples using phase coherence, power spectral density, and
coherence spectra analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, welch, butter, filtfilt
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from pywt import cwt
import os
import time
import psutil
import concurrent.futures
from datetime import datetime
import pathlib

# Import required libraries
from braingeneers import analysis
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy import stats
from matplotlib.animation import FuncAnimation
from cmcrameri import cm as cmc
from matplotlib.gridspec import GridSpec

# Import custom modules
from loaders import *
from new_lfp_processor_class import *

# Configuration dictionary for gradient calculations
GRADIENT_CONFIG = {
    'MAX_NEIGHBOR_DIST': 50.0,      # µm
    'SPATIAL_SIGMA': 45.0,           # µm for Gaussian kernel
    'RIDGE_LAMBDA': 1e-5,            # regularization
    'COVERAGE_ANGLE_GAP': 120.0,     # degrees
    'MIN_GRADIENT': 1e-5             # rad/µm
}


def detect_energy_peaks(lfp_processor, data_type='lfp', window_start=0, window_length=None, 
                       downsample_factor=1, smoothing_sigma=15, threshold=2.0, 
                       min_duration=0.1, min_interval=0.2, return_raw=False, 
                       plot_results=False, figsize=(15, 6), 
                       color='blue', highlight_color='skyblue', highlight_alpha=0.3,
                       show_stats=True, title=None, optogenetic_intervals=None):
    """
    Detect peaks of high energy in LFP data with option to omit optogenetic intervals.
    
    [Full docstring remains the same as in notebook]
    """
    # Validate inputs
    if data_type not in lfp_processor.waves:
        raise ValueError(f"Data type {data_type} not found in waves dictionary. Available types: {list(lfp_processor.waves.keys())}")
    
    # Validate optogenetic intervals
    if optogenetic_intervals is not None:
        if not isinstance(optogenetic_intervals, (list, tuple)):
            raise ValueError("optogenetic_intervals must be a list or tuple of (start_time, end_time) pairs")
        
        # Validate each interval
        for i, interval in enumerate(optogenetic_intervals):
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(f"Each optogenetic interval must be a (start_time, end_time) pair. Invalid interval at index {i}: {interval}")
            
            start_time, end_time = interval
            if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
                raise ValueError(f"Interval times must be numeric. Invalid interval at index {i}: {interval}")
            
            if start_time >= end_time:
                raise ValueError(f"Start time must be less than end time. Invalid interval at index {i}: {interval}")
        
        # Sort intervals by start time for easier processing
        optogenetic_intervals = sorted(optogenetic_intervals, key=lambda x: x[0])
        print(f"Using {len(optogenetic_intervals)} optogenetic intervals for filtering")
    
    # Get window parameters
    if window_length is None:
        window_length = lfp_processor.waves[data_type].shape[1] - window_start
    
    window_end = window_start + window_length
    if window_end > lfp_processor.waves[data_type].shape[1]:
        raise ValueError("Analysis window exceeds data length")
    
    # Compute instantaneous energy using the processor's method
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
        
        # Create time interpolator for accurate conversion between indices and times
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
                """
                Check if a peak overlaps with any optogenetic interval.
                Returns True if the peak should be filtered out.
                """
                for opto_start, opto_end in intervals:
                    # Check if peak time is within optogenetic interval
                    if opto_start <= peak_time <= opto_end:
                        return True
                    
                    # Check if peak interval overlaps with optogenetic interval
                    # Overlap occurs if: peak_start < opto_end AND peak_end > opto_start
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
        
        # Store peak information for valid peaks (after all filtering)
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
        
        # Add fill under curve for better visibility
        ax.fill_between(time_points, 0, smoothed_energy, color=color, alpha=0.2)
        
        # Plot threshold line
        ax.axhline(y=peak_threshold, color='gray', linestyle='--', alpha=0.7, 
                  label=f'Threshold ({threshold}×RMS)')
        
        # Highlight optogenetic intervals if provided
        if optogenetic_intervals is not None:
            for opto_start, opto_end in optogenetic_intervals:
                # Only highlight intervals that overlap with the time window
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
        
        # Store figure and axis in the result
        peaks['fig'] = fig
        peaks['ax'] = ax
    
    return peaks


def plot_coherence_spectral_analysis(
    lfp_processor,
    burst_intervals,
    freq_range=(1, 80),
    n_bins=100,
    coherence_method='l2_norm',
    smoothing_sigma=3,
    figure_size=(20, 15),
    colormap='cmc.lapaz',
    save_path=None,
    show_fband_markers=True,
    min_gradient=1e-4,
    dpi=300,
    n_wavelets=30,
    wavelet_width=7.0,
    amp_thresh_factor=0.1,
    time_decim=10,
    nfft=2048,
    lpf_cutoff=80,
    detrend_method='linear',
    window_type='hann',
    max_nperseg=1024,
    overlap_fraction=0.75,
    scaling_factor=10,
    x_left_lim=None):
    """
    Spectral analysis of LFP burst intervals using both Morlet wavelets and Welch method,
    calculating per‐timepoint coherence (MI) & PGD, and precomputed neighbor maps
    for fast complex‐gradient computation.
    
    [Full docstring remains the same as in notebook]
    """
    #— prepare colormap
    try:
        from cmcrameri import cm as cmc
        cmap = eval(colormap) if colormap.startswith('cmc.') else plt.get_cmap(colormap)
    except (ImportError, NameError):
        cmap = plt.get_cmap('viridis')

    #— frequency bands for annotation (updated for new range)
    frequency_bands = [
        (1, 4, 'Delta'),
        (4, 8, 'Theta'),
        (8, 13, 'Alpha'),
        (13, 30, 'Beta'),
        (30, 60, 'Gamma'),
        (60, 80, 'High Gamma')
    ]

    #— results container
    results = {'burst_intervals': burst_intervals, 'psd': [], 'coherence': [], 'pgd': [], 'welch_psd': []}

    #— build figure & axes with expanded grid including Coherence vs Power plots
    fig = plt.figure(figsize=figure_size)
    gs = GridSpec(3, 4, figure=fig, wspace=0.4, hspace=0.4)

    # Create all axes
    ax_psd   = fig.add_subplot(gs[0,0])  # CWT PSD (Individual)
    ax_coh   = fig.add_subplot(gs[0,1])  # Coherence (Individual)
    ax_pgd   = fig.add_subplot(gs[0,2])  # PGD (Individual)
    ax_wpsd  = fig.add_subplot(gs[0,3])  # Welch PSD (Individual)

    ax_psd_m = fig.add_subplot(gs[1,0])  # CWT PSD (Median)
    ax_coh_m = fig.add_subplot(gs[1,1])  # Coherence (Median)
    ax_pgd_m = fig.add_subplot(gs[1,2])  # PGD (Median)
    ax_wpsd_m = fig.add_subplot(gs[1,3]) # Welch PSD (Median)

    ax_coh_vs_pow = fig.add_subplot(gs[2,0])    # NEW: Coherence vs Power (Individual)
    ax_coh_vs_pow_m = fig.add_subplot(gs[2,1])  # NEW: Coherence vs Power (Median)
    ax_pgd_vs_pow = fig.add_subplot(gs[2,2])    # NEW: PGD vs Power (Individual)
    ax_pgd_vs_pow_m = fig.add_subplot(gs[2,3])  # NEW: PGD vs Power (Median)

    # Set titles for all subplots with improved naming
    for ax, title in zip(
        [ax_psd, ax_coh, ax_pgd, ax_wpsd, ax_psd_m, ax_coh_m, ax_pgd_m, ax_wpsd_m,
         ax_coh_vs_pow, ax_coh_vs_pow_m, ax_pgd_vs_pow, ax_pgd_vs_pow_m],
        ['Power Spectral Density per event', 'Phase Coherence Index per event', 'PGD per event', 'Power Spectral Density per event',
         'Power Spectral Density (Median)', 'Phase Coherence Index (Median)', 'PGD (Median)', 'Power Spectral Density (Median)',
         'Coherence vs Power', 'Coherence vs Power (Med)', 'PGD vs Power', 'PGD vs Power (Med)']
    ):
        ax.set_title(title, fontsize=14)
        
        # Set x-limits based on plot type
        if 'vs Power' in title:
            # Power plots will have their own x-axis limits set dynamically
            pass
        else:
            # Use dynamic x-limits: user override or original freq_range
            left_lim = x_left_lim if x_left_lim is not None else freq_range[0]
            ax.set_xlim((left_lim, freq_range[1]))
        ax.grid(True, alpha=0.3)

    # Set log scales for PSD plots only (not the vs Power plots)
    for ax in [ax_psd, ax_psd_m, ax_wpsd, ax_wpsd_m]:
        ax.set_xscale('log')
        ax.set_yscale('log')

    # Set axis labels with proper units
    ax_psd.set_xlabel('Freq (Hz)')
    ax_psd.set_ylabel('Power (µV²)')  # CWT PSD units
    ax_coh.set_xlabel('Freq (Hz)')
    ax_coh.set_ylabel('Coherence (MI)')
    ax_pgd.set_xlabel('Freq (Hz)')
    ax_pgd.set_ylabel('PGD')
    ax_wpsd.set_xlabel('Freq (Hz)')
    ax_wpsd.set_ylabel('Power Density (µV²/Hz)')  # Welch PSD units
    
    ax_psd_m.set_xlabel('Freq (Hz)')
    ax_psd_m.set_ylabel('Power (µV²)')  # CWT PSD units
    ax_coh_m.set_xlabel('Freq (Hz)')
    ax_coh_m.set_ylabel('Coherence (MI)')
    ax_pgd_m.set_xlabel('Freq (Hz)')
    ax_pgd_m.set_ylabel('PGD')
    ax_wpsd_m.set_xlabel('Freq (Hz)')
    ax_wpsd_m.set_ylabel('Power Density (µV²/Hz)')  # Welch PSD units

    # Set axis labels for new plots
    ax_coh_vs_pow.set_xlabel('Power Density (µV²/Hz)')
    ax_coh_vs_pow.set_ylabel('Coherence (MI)')
    ax_coh_vs_pow_m.set_xlabel('Power Density (µV²/Hz)')
    ax_coh_vs_pow_m.set_ylabel('Coherence (MI)')
    ax_pgd_vs_pow.set_xlabel('Power Density (µV²/Hz)')
    ax_pgd_vs_pow.set_ylabel('PGD')
    ax_pgd_vs_pow_m.set_xlabel('Power Density (µV²/Hz)')
    ax_pgd_vs_pow_m.set_ylabel('PGD')

    # Set log scale for power axes
    for ax in [ax_coh_vs_pow, ax_coh_vs_pow_m, ax_pgd_vs_pow, ax_pgd_vs_pow_m]:
        ax.set_xscale('log')

    #— electrode locations & neighbor precompute
    locs = np.array(lfp_processor.locations)  # shape (n_ch, 2)
    n_ch = locs.shape[0]
    tree = cKDTree(locs)
    neighbor_idx, inv_ATA = [], []
    for i in range(n_ch):
        d, idx = tree.query(locs[i], k=min(n_ch, 9), distance_upper_bound=200.0)
        valid = (d > 0) & np.isfinite(d)
        nbrs = idx[valid]
        if len(nbrs) < 3:
            neighbor_idx.append(np.array([],dtype=int))
            inv_ATA.append(None)
        else:
            rel = locs[nbrs] - locs[i]                   # (n_nbrs, 2)
            ATA = rel.T @ rel + 1e-2 * np.eye(2)         # ridge
            inv_ATA.append(np.linalg.inv(ATA))
            neighbor_idx.append(nbrs)

    eps = 1e-12  # for stability

    #— helpers
    def compute_coherence(phases, method, n_bins):
        """
        Compute phase coherence using specified method with normalization.
        """
        if len(phases) < 3:
            return 0.0
        hist, _ = np.histogram(phases, bins=np.linspace(-np.pi, np.pi, n_bins+1), density=True)
        hist = hist/np.sum(hist)
        uni = np.ones_like(hist)/hist.size
        
        if method == 'l2_norm':
            return np.sqrt(np.sum((hist - uni)**2))
        
        h = hist + 1e-10
        h /= np.sum(h)
        
        if method == 'kl_divergence':
            # Compute KL divergence
            kl_div = np.sum(h * np.log(h/uni))
            # Normalize to [0,1] range to obtain Modulation Index
            # Using normalization: MI = kl_div / log(n_bins)
            return kl_div / np.log(n_bins)
        
        if method == 'entropy':
            ent = -np.sum(h * np.log(h))
            max_ent = -np.sum(uni * np.log(uni))
            return 1 - ent/max_ent
        
        raise ValueError(f"Unknown coherence method: {method}")

    def compute_complex_gradient_timepoint(phs):
        """
        Compute complex gradient at a single timepoint.
        """
        gx = np.zeros(n_ch, dtype=complex)
        gy = np.zeros(n_ch, dtype=complex)
        amps = np.abs(phs)
        thr = amp_thresh_factor * np.median(amps)
        for i in range(n_ch):
            nbrs = neighbor_idx[i]
            if nbrs.size < 3:
                continue
            # mask low‐amp neighbors
            good = nbrs[amps[nbrs] >= thr]
            if good.size < 3 or amps[i] < thr:
                continue
            diffs = phs[good] / (phs[i] + eps)        # (n_good,)
            ATb = (locs[good] - locs[i]).T @ diffs    # (2,)
            g = inv_ATA[i] @ ATb                      # (2,)
            gx[i], gy[i] = g[0], g[1]
        return gx, gy

    def compute_pgd(gx, gy, min_g):
        """
        Compute Phase Gradient Directionality from gradient components.
        """
        mag = np.sqrt(np.abs(gx)**2 + np.abs(gy)**2)
        valid = mag > min_g
        if np.sum(valid) < 3:
            return 0.0
        vm = mag[valid]
        mvx = np.mean(gx[valid])
        mvy = np.mean(gy[valid])
        return np.sqrt(np.abs(mvx)**2 + np.abs(mvy)**2) / np.mean(vm)

    # Design low-pass filter for artifact reduction
    nyq = lfp_processor.fs / 2
    if lpf_cutoff < nyq:
        b_lpf, a_lpf = butter(4, lpf_cutoff / nyq, btype='low')
        use_filter = True
    else:
        use_filter = False
        print(f"Warning: lpf_cutoff ({lpf_cutoff}) >= Nyquist frequency ({nyq}). Skipping filtering.")

    #— frequency bins & PyWavelets scales
    min_f, max_f = freq_range
    freq_bins = np.linspace(min_f, max_f, n_wavelets)
    scales = wavelet_width * lfp_processor.fs / (2 * np.pi * freq_bins)

    all_vals = {'psd': [], 'coherence': [], 'pgd': [], 'welch_psd': []}

    #— process each burst
    for i, (t0, t1) in enumerate(burst_intervals):
        print(f"Processing burst {i+1}/{len(burst_intervals)}: {t0:.2f}-{t1:.2f}s")
        color = cmap(i / max(1, len(burst_intervals)-1))
        pad = 0.5
        s0, s1 = max(0, t0-pad), t1+pad
        i0, i1 = int(s0 * lfp_processor.fs), int(s1 * lfp_processor.fs)
        lfp = lfp_processor.waves['lfp'][:, i0:i1]
        n_ch_lfp, n_samp = lfp.shape
        if n_samp < 10:
            print("  window too short, skipping")
            continue

        #— compute CWT for each channel
        tfr = np.zeros((n_ch, n_wavelets, n_samp), dtype=complex)
        for ch in range(n_ch):
            coefs, _ = cwt(lfp[ch], scales, 'morl', sampling_period=1/lfp_processor.fs)
            tfr[ch] = coefs

        #— PSD (time & channel averaged) using Morlet wavelets
        psd = np.mean(np.abs(tfr)**2, axis=2)       # (n_ch × n_freqs)
        psd_mean = np.mean(psd, axis=0)             # (n_freqs,)
        results['psd'].append({
            'burst_idx': i, 'start_time': t0, 'end_time': t1,
            'freqs': freq_bins, 'values': psd_mean
        })
        all_vals['psd'].append(psd_mean)
        
        #— IMPROVED: Compute Welch PSD with artifact reduction and better low-frequency resolution
        welch_psd_all = []
        for ch in range(n_ch):
            # Pre-filter the signal to remove high-frequency artifacts
            if use_filter:
                try:
                    lfp_filtered = filtfilt(b_lpf, a_lpf, lfp[ch])
                except Exception as e:
                    print(f"Warning: Filtering failed for channel {ch}: {e}")
                    lfp_filtered = lfp[ch]
            else:
                lfp_filtered = lfp[ch]
            
            # Calculate nperseg to ensure we can reach the minimum frequency
            # Frequency resolution = fs / nperseg, so nperseg = fs / desired_freq_resolution
            min_freq_resolution = min_f / 2  # Allow resolution of half the minimum frequency
            min_nperseg = int(lfp_processor.fs / min_freq_resolution)
            
            # Use larger segments for better low-frequency resolution
            nperseg = min(max_nperseg, n_samp//2)  # Allow larger segments
            nperseg = max(nperseg, min_nperseg)    # Ensure minimum for low frequencies
            nperseg = min(nperseg, n_samp)         # Don't exceed signal length
            
            noverlap = int(nperseg * overlap_fraction)  # Controlled overlap
            
            # Ensure minimum segment size
            if nperseg < 64:  # Increased minimum
                nperseg = min(64, n_samp//2)
                noverlap = int(nperseg * 0.5)
            
            try:
                f, Pxx = welch(
                    lfp_filtered, 
                    fs=lfp_processor.fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    nfft=nfft,
                    window=window_type,
                    detrend=detrend_method,
                    scaling='density'  # Use density scaling
                )
                
                # Apply reduced scaling factor to avoid amplifying noise
                Pxx = Pxx * scaling_factor
                
                # Limit to frequency range
                mask = (f >= min_f) & (f <= max_f)
                welch_psd_all.append(Pxx[mask])
                welch_freqs = f[mask]
                
            except Exception as e:
                print(f"Warning: Welch computation failed for channel {ch}: {e}")
                # Create dummy data if Welch fails
                welch_freqs = np.linspace(min_f, max_f, 50)
                welch_psd_all.append(np.ones_like(welch_freqs) * 1e-6)
        
        # Average across channels
        if welch_psd_all:
            welch_psd_mean = np.mean(np.array(welch_psd_all), axis=0)
            results['welch_psd'].append({
                'burst_idx': i, 'start_time': t0, 'end_time': t1,
                'freqs': welch_freqs, 'values': welch_psd_mean
            })
            all_vals['welch_psd'].append(welch_psd_mean)
            
            # Plot Welch PSD
            ax_wpsd.plot(welch_freqs, welch_psd_mean, color=color, linewidth=2.5)

        #— per-frequency coherence & PGD (over decimated time)
        coh_vals = np.zeros(n_wavelets)
        pgd_vals = np.zeros(n_wavelets)
        t_idx = np.arange(0, n_samp, time_decim)

        for fi in range(n_wavelets):
            coh_t = np.zeros_like(t_idx, dtype=float)
            pgd_t = np.zeros_like(t_idx, dtype=float)
            for j, tt in enumerate(t_idx):
                inst = tfr[:, fi, tt]
                coh_t[j] = compute_coherence(np.angle(inst), coherence_method, n_bins)
                gx, gy = compute_complex_gradient_timepoint(inst)
                pgd_t[j] = compute_pgd(gx, gy, min_gradient)
            coh_vals[fi] = np.nanmean(coh_t)
            pgd_vals[fi] = np.nanmean(pgd_t)

        #— smooth across frequency
        coh_vals = gaussian_filter1d(coh_vals, sigma=smoothing_sigma)
        pgd_vals = gaussian_filter1d(pgd_vals, sigma=smoothing_sigma)

        #— store & plot
        results['coherence'].append({
            'burst_idx': i, 'start_time': t0, 'end_time': t1,
            'freqs': freq_bins, 'values': coh_vals
        })
        results['pgd'].append({
            'burst_idx': i, 'start_time': t0, 'end_time': t1,
            'freqs': freq_bins, 'values': pgd_vals
        })
        all_vals['coherence'].append(coh_vals)
        all_vals['pgd'].append(pgd_vals)

        ax_psd.plot(freq_bins, psd_mean, color=color, linewidth=2.5)
        ax_coh.plot(freq_bins, coh_vals, color=color, linewidth=2.5)
        ax_pgd.plot(freq_bins, pgd_vals, color=color, linewidth=2.5)
        
        # NEW: Plot Coherence vs Power and PGD vs Power
        if welch_psd_all:
            # Create interpolation functions
            coh_interp = interp1d(freq_bins, coh_vals, bounds_error=False, fill_value=np.nan)
            pgd_interp = interp1d(freq_bins, pgd_vals, bounds_error=False, fill_value=np.nan)
            
            # Get coherence and PGD values at Welch frequencies
            coh_at_welch_freqs = coh_interp(welch_freqs)
            pgd_at_welch_freqs = pgd_interp(welch_freqs)
            
            # Remove NaN values
            valid_mask = ~(np.isnan(coh_at_welch_freqs) | np.isnan(pgd_at_welch_freqs) | np.isnan(welch_psd_mean))
            
            if np.sum(valid_mask) > 0:
                ax_coh_vs_pow.scatter(welch_psd_mean[valid_mask], coh_at_welch_freqs[valid_mask], 
                                    color=color, alpha=0.7, s=30)
                ax_pgd_vs_pow.scatter(welch_psd_mean[valid_mask], pgd_at_welch_freqs[valid_mask], 
                                    color=color, alpha=0.7, s=30)

    #— median panels
    if all(len(all_vals[k]) for k in all_vals):
        for arr, ax in [('psd', ax_psd_m), ('coherence', ax_coh_m), ('pgd', ax_pgd_m)]:
            vals = np.vstack(all_vals[arr])
            med = np.median(vals, axis=0)
            q25 = np.percentile(vals, 25, axis=0)
            q75 = np.percentile(vals, 75, axis=0)
            ax.plot(freq_bins, med, color='black', linewidth=3, label='Median')
            ax.fill_between(freq_bins, q25, q75, color='gray', alpha=0.3)
        
        # Add Welch PSD median plot if available
        if len(all_vals['welch_psd']) > 0:
            # Interpolate to common frequency grid for combination
            ref_freqs = results['welch_psd'][0]['freqs']
            interp_psd = []
            
            for idx, wpsd in enumerate(all_vals['welch_psd']):
                if isinstance(wpsd, list):
                    # Convert list to numpy array if needed
                    wpsd = np.array(wpsd)
                
                if len(wpsd) != len(ref_freqs):
                    # Interpolate to match reference frequencies
                    curr_freqs = results['welch_psd'][min(idx, len(results['welch_psd'])-1)]['freqs']
                    if len(curr_freqs) == len(wpsd):
                        f = interp1d(curr_freqs, wpsd, bounds_error=False, fill_value='extrapolate')
                        interp_psd.append(f(ref_freqs))
                    else:
                        interp_psd.append(wpsd[:len(ref_freqs)] if len(wpsd) > len(ref_freqs) else 
                                        np.pad(wpsd, (0, len(ref_freqs)-len(wpsd)), 'constant'))
                else:
                    interp_psd.append(wpsd)
            
            if interp_psd:
                welch_vals = np.vstack(interp_psd)
                med_welch = np.median(welch_vals, axis=0)
                q25_welch = np.percentile(welch_vals, 25, axis=0)
                q75_welch = np.percentile(welch_vals, 75, axis=0)
                
                ax_wpsd_m.plot(ref_freqs, med_welch, color='black', linewidth=3, label='Median')
                ax_wpsd_m.fill_between(ref_freqs, q25_welch, q75_welch, color='gray', alpha=0.3)
        
        # Add median Coherence vs Power plots
        if len(all_vals['welch_psd']) > 0 and len(all_vals['coherence']) > 0:
            # Collect all coherence vs power pairs
            all_powers = []
            all_cohs = []
            all_pgds = []
            
            for idx in range(len(all_vals['welch_psd'])):
                wpsd = all_vals['welch_psd'][idx]
                coh = all_vals['coherence'][idx]
                pgd = all_vals['pgd'][idx]
                
                # Interpolate to common frequency grid
                coh_interp = interp1d(freq_bins, coh, bounds_error=False, fill_value=np.nan)
                pgd_interp = interp1d(freq_bins, pgd, bounds_error=False, fill_value=np.nan)
                
                welch_freqs_curr = results['welch_psd'][min(idx, len(results['welch_psd'])-1)]['freqs']
                coh_at_welch = coh_interp(welch_freqs_curr)
                pgd_at_welch = pgd_interp(welch_freqs_curr)
                
                valid_mask = ~(np.isnan(coh_at_welch) | np.isnan(pgd_at_welch) | np.isnan(wpsd))
                
                if np.sum(valid_mask) > 0:
                    all_powers.extend(wpsd[valid_mask])
                    all_cohs.extend(coh_at_welch[valid_mask])
                    all_pgds.extend(pgd_at_welch[valid_mask])
            
            # Plot median relationships
            if len(all_powers) > 0:
                # Sort by power for clean line plotting
                sort_idx = np.argsort(all_powers)
                sorted_powers = np.array(all_powers)[sort_idx]
                sorted_cohs = np.array(all_cohs)[sort_idx]
                sorted_pgds = np.array(all_pgds)[sort_idx]
                
                ax_coh_vs_pow_m.scatter(sorted_powers, sorted_cohs, color='black', alpha=0.5, s=20)
                ax_pgd_vs_pow_m.scatter(sorted_powers, sorted_pgds, color='black', alpha=0.5, s=20)
                
    else:
        print("Not enough data for median plots")

    #— frequency‐band markers - simplified for better display
    if show_fband_markers:
        # Function to add band markers to specific axes
        def add_band_markers(ax):
            # Get y-axis limits for proper scaling (important for log scale)
            y0, y1 = ax.get_ylim()
            
            # Calculate appropriate height for the markers based on plot type
            if ax in [ax_psd, ax_psd_m, ax_wpsd, ax_wpsd_m]:  # Log scale
                # Place at the very bottom of the plot
                band_height = 0.01 * y0
                rect_height = 0.02 * y0
            else:  # Linear scale
                band_height = 0.01 * (y1 - y0)
                rect_height = 0.02 * (y1 - y0)
            
            # Reset y-limits to ensure rectangles don't change the scale
            ax.set_ylim(y0, y1)
        
        # Add markers to frequency axes only (not power vs coherence plots)
        for ax in [ax_psd, ax_coh, ax_pgd, ax_psd_m, ax_coh_m, ax_pgd_m, ax_wpsd, ax_wpsd_m]:
            add_band_markers(ax)

    #— legends & title
    ax_psd.legend(fontsize=8, loc='upper right')
    ax_psd_m.legend(fontsize=8, loc='upper right')
    ax_coh_m.legend(fontsize=8, loc='upper right')
    ax_pgd_m.legend(fontsize=8, loc='upper right')
    ax_wpsd_m.legend(fontsize=8, loc='upper right')
    
    fig.suptitle(f'Spectral Analysis (CWT + Welch + MI + PGD + Power Relationships) - Filtered at {lpf_cutoff}Hz', fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)

    #— save if requested (both PNG and SVG) with robust error handling
    if save_path:
        try:
            # Convert to absolute path and ensure parent directory exists
            path = pathlib.Path(save_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save PNG version
            png_path = str(path)
            fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved PNG figure to {png_path}")
            
            # Create SVG path more carefully
            svg_filename = path.stem + '.svg'
            svg_path = path.parent / svg_filename
            
            # Try to save SVG version
            try:
                fig.savefig(str(svg_path), format='svg', bbox_inches='tight')
                print(f"Saved SVG figure to {svg_path}")
            except (OSError, PermissionError) as svg_error:
                print(f"Could not save SVG to {svg_path}: {svg_error}")
                
                # Try saving SVG to a simpler location/name
                try:
                    simple_svg_path = pathlib.Path.cwd() / f"figure_{path.stem}.svg"
                    fig.savefig(str(simple_svg_path), format='svg', bbox_inches='tight')
                    print(f"Saved SVG figure to {simple_svg_path}")
                except Exception as e2:
                    print(f"Could not save SVG at all: {e2}")
            
        except Exception as e:
            print(f"Error in save process: {e}")
            # Fallback: try to save just PNG to current directory
            try:
                fallback_path = pathlib.Path.cwd() / "fallback_figure.png"
                fig.savefig(str(fallback_path), dpi=dpi, bbox_inches='tight')
                print(f"Saved fallback PNG to {fallback_path}")
            except Exception as e2:
                print(f"Complete save failure: {e2}")

    return fig, (ax_psd, ax_coh, ax_pgd, ax_wpsd, ax_psd_m, ax_coh_m, ax_pgd_m, ax_wpsd_m, 
                 ax_coh_vs_pow, ax_coh_vs_pow_m, ax_pgd_vs_pow, ax_pgd_vs_pow_m), results


def _pgd_process_batch(batch_idx, batch_size, phase_data, downsampled_indices, grad_calc, verbose=True, debug=False):
    """
    Process a batch of timepoints for PGD calculation.
    This function works with both threading and multiprocessing.
    
    [Full docstring remains the same as in notebook]
    """
    from datetime import datetime
    
    # Helper function for logging
    def log(message, is_debug=False):
        if not verbose and not is_debug:
            return
        if is_debug and not debug:
            return
        
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefix = f"[{current_time}][Batch {batch_idx}][DEBUG]" if is_debug else f"[{current_time}][Batch {batch_idx}]"
        print(f"{prefix} {message}")
    
    batch_start = time.time()
    log(f"Processing batch with batch size {batch_size}...", is_debug=True)
    
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(downsampled_indices))
    
    if end_idx <= start_idx:
        log(f"Batch is empty (start_idx={start_idx}, end_idx={end_idx})", is_debug=True)
        return [], []
    
    batch_indices = downsampled_indices[start_idx:end_idx]
    extract_start = time.time()
    phase_data_batch = phase_data[:, batch_indices]
    log(f"Data extraction time: {time.time() - extract_start:.4f} seconds", is_debug=True)
    
    # Compute PGD for this batch
    pgd_start = time.time()
    pgd_batch = grad_calc.compute_pgd_batch(phase_data_batch)
    log(f"PGD computation time: {time.time() - pgd_start:.4f} seconds", is_debug=True)
    
    log(f"Total processing time: {time.time() - batch_start:.4f} seconds")
    return list(range(start_idx, end_idx)), pgd_batch


def detect_pgd_peaks_accelerated(lfp_processor, data_type='theta', window_start=0, window_length=None, 
                       downsample_factor=1, smoothing_sigma=15, threshold=0.5, 
                       min_duration=0.1, min_interval=0.2, return_raw=False, 
                       plot_results=False, figsize=(15, 6), 
                       color='green', highlight_color='lightgreen', highlight_alpha=0.3,
                       show_stats=True, title=None, min_gradient=1e-5,
                       batch_size=100, use_gpu=True, n_processes=None, 
                       verbose=True, debug=False):
    """
    Accelerated version of detect_pgd_peaks using GPU computation and threading.
    
    [Full docstring remains the same as in notebook]
    """
    # Function to get current memory usage
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # in MB
    
    # Custom logger with timestamps that respects verbosity settings
    def log(message, is_debug=False):
        if not verbose and not is_debug:
            return
        if is_debug and not debug:
            return
        
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefix = f"[{current_time}][DEBUG]" if is_debug else f"[{current_time}]"
        print(f"{prefix} {message}")
    
    # Start timing and initial memory usage
    start_time_total = time.time()
    initial_memory = get_memory_usage()
    log(f"Starting PGD peak detection. Initial memory usage: {initial_memory:.2f} MB")
    
    # GPU detection and setup
    if use_gpu and lfp_processor.cuda_available:
        xp = lfp_processor.xp  # This should be cupy if CUDA is available
        log(f"Using GPU acceleration for PGD computation")
        
        # Get GPU info if available
        try:
            import cupy as cp
            device = cp.cuda.Device()
            device_properties = cp.cuda.runtime.getDeviceProperties(0)
            log(f"GPU: {device_properties['name'].decode()}")
            log(f"Total GPU Memory: {device.mem_info[1] / 1024**3:.2f} GB")
            log(f"Free GPU Memory: {device.mem_info[0] / 1024**3:.2f} GB")
        except (ImportError, AttributeError) as e:
            log(f"Could not get detailed GPU info: {str(e)}", is_debug=True)
    else:
        xp = np
        use_gpu = False
        log(f"Using CPU for PGD computation")
        log(f"Number of CPU cores available: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    
    # Standard validation and setup code
    validation_start = time.time()
    log("Validating inputs...")
    
    if data_type not in lfp_processor.waves:
        raise ValueError(f"Data type {data_type} not found in waves dictionary. Available types: {list(lfp_processor.waves.keys())}")
    
    if window_length is None:
        window_length = lfp_processor.waves[data_type].shape[1] - window_start
    
    window_end = window_start + window_length
    if window_end > lfp_processor.waves[data_type].shape[1]:
        raise ValueError("Analysis window exceeds data length")
    
    log(f"Input validation completed in {time.time() - validation_start:.4f} seconds")
    
    # Get phase data for the entire window
    phase_data_start = time.time()
    log(f"Getting analytical data for window [{window_start}:{window_end}] (length {window_length} frames)...")
    analytical_data = lfp_processor._get_analytical_data(data_type, window_start, window_end)
    phase_data = analytical_data['phase']
    phase_data_time = time.time() - phase_data_start
    log(f"Got analytical data in {phase_data_time:.4f} seconds. Shape: {phase_data.shape}")
    log(f"Memory usage after loading phase data: {get_memory_usage():.2f} MB (+{get_memory_usage() - initial_memory:.2f} MB)")
    
    # Generate time points for the window
    time_points = np.arange(window_start, window_end, downsample_factor) / lfp_processor.fs
    
    # Define GPU-accelerated RegularizedGradient class
    class GPURegularizedGradient:
        def __init__(self, locations, min_gradient=1e-5, max_neighbor_dist=50.0, 
                     spatial_sigma=45.0, ridge_lambda=1e-5, coverage_angle_gap=120.0, use_gpu=True):
            self.locations = locations
            self.min_gradient = min_gradient
            self.max_neighbor_dist = max_neighbor_dist
            self.spatial_sigma = spatial_sigma
            self.ridge_lambda = ridge_lambda
            self.coverage_angle_gap = coverage_angle_gap
            self.use_gpu = use_gpu and lfp_processor.cuda_available
            self.xp = xp
            
            # Precompute neighbor relationships
            init_start = time.time()
            log("Precomputing electrode neighbor relationships...", is_debug=True)
            self._precompute_neighbors()
            log(f"Neighbor precomputation completed in {time.time() - init_start:.4f} seconds", is_debug=True)
            
        def _precompute_neighbors(self):
            """Precompute neighbor relationships for all electrodes"""
            
            precompute_start = time.time()
            # Build KD-tree
            log(f"Building KD-tree for {len(self.locations)} electrodes...", is_debug=True)
            kdtree_start = time.time()
            self.tree = cKDTree(self.locations)
            log(f"KD-tree built in {time.time() - kdtree_start:.4f} seconds", is_debug=True)
            
            # Query for neighbors within max_neighbor_dist
            query_start = time.time()
            log(f"Querying for electrode neighbors within {self.max_neighbor_dist} units...", is_debug=True)
            dists, idxs = self.tree.query(
                self.locations, 
                k=min(len(self.locations), 20),
                distance_upper_bound=self.max_neighbor_dist
            )
            log(f"Neighbor query completed in {time.time() - query_start:.4f} seconds", is_debug=True)
            
            # Store neighbor indices and distances
            process_start = time.time()
            log("Processing neighbor relationships for each electrode...", is_debug=True)
            
            self.neighbor_indices = []
            self.neighbor_dists = []
            self.electrode_masks = []
            self.dx_dy_arrays = []
            self.weight_arrays = []
            self.coverage_masks = []
            
            for i in range(len(self.locations)):
                # Get valid neighbors
                valid = dists[i] < np.inf
                nbrs = idxs[i][valid]
                dist = dists[i][valid]
                
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
                    weights = np.exp(-0.5 * (dist / self.spatial_sigma)**2)
                    
                    # Check angular coverage
                    angles = np.degrees(np.mod(np.arctan2(dy, dx), 2*np.pi))
                    angles_sorted = np.sort(angles)
                    gaps = np.diff(np.concatenate([angles_sorted, angles_sorted[:1] + 360]))
                    has_coverage = np.max(gaps) <= self.coverage_angle_gap
                    
                    self.neighbor_indices.append(nbrs)
                    self.neighbor_dists.append(dist)
                    self.dx_dy_arrays.append(np.column_stack([dx, dy]))
                    self.weight_arrays.append(weights)
                    self.coverage_masks.append(has_coverage)
                else:
                    self.neighbor_indices.append(np.array([], dtype=int))
                    self.neighbor_dists.append(np.array([]))
                    self.dx_dy_arrays.append(np.empty((0, 2)))
                    self.weight_arrays.append(np.array([]))
                    self.coverage_masks.append(False)
            
            # Convert to arrays for faster indexing
            self.has_valid_neighbors = np.array([len(nbrs) >= 3 for nbrs in self.neighbor_indices])
            self.has_coverage = np.array(self.coverage_masks)
            self.valid_electrodes = self.has_valid_neighbors & self.has_coverage
            
            log(f"Neighbor processing completed in {time.time() - process_start:.4f} seconds", is_debug=True)
            log(f"Total precomputation time: {time.time() - precompute_start:.4f} seconds", is_debug=True)
            log(f"Precomputed neighbors for {sum(self.valid_electrodes)} electrodes out of {len(self.locations)}")
            log(f"Excluded {len(self.locations) - sum(self.valid_electrodes)} electrodes due to insufficient neighbors or poor coverage")
            
            # Report memory usage
            approx_mem = (
                sum(len(x) for x in self.neighbor_indices) * 4 +  # 4 bytes per int for neighbor indices
                sum(len(x) for x in self.neighbor_dists) * 8 +     # 8 bytes per float for distances
                sum(x.size * 8 for x in self.dx_dy_arrays) +      # 8 bytes per float for dx_dy arrays
                sum(len(x) * 8 for x in self.weight_arrays)       # 8 bytes per float for weight arrays
            ) / (1024 * 1024)  # Convert to MB
            
            log(f"Approximate memory used for neighbor data: {approx_mem:.2f} MB")
            log(f"Current total memory usage: {get_memory_usage():.2f} MB", is_debug=True)
        
        def compute_batch(self, phase_data_batch):
            """
            Compute phase gradients for a batch of time points.
            
            [Method docstring remains the same as in notebook]
            """
            compute_start = time.time()
            n_electrodes, n_timepoints = phase_data_batch.shape
            
            log(f"Computing gradients for batch with {n_timepoints} timepoints across {n_electrodes} electrodes...", is_debug=True)
            
            # Initialize output arrays
            grad_x_batch = self.xp.zeros((n_timepoints, n_electrodes), dtype=self.xp.float32)
            grad_y_batch = self.xp.zeros((n_timepoints, n_electrodes), dtype=self.xp.float32)
            
            # Move phase data to GPU if needed
            if self.use_gpu:
                gpu_transfer_start = time.time()
                log(f"Moving phase data to GPU...", is_debug=True)
                phase_data_gpu = self.xp.asarray(phase_data_batch)
                log(f"GPU transfer completed in {time.time() - gpu_transfer_start:.4f} seconds", is_debug=True)
            else:
                phase_data_gpu = phase_data_batch
            
            # Process each electrode with valid neighbors
            electrode_time_total = 0
            electrode_count = 0
            valid_electrodes = np.where(self.valid_electrodes)[0]
            
            log(f"Processing {len(valid_electrodes)} valid electrodes...", is_debug=True)
            
            for i in valid_electrodes:
                electrode_start = time.time()
                
                nbrs = self.neighbor_indices[i]
                dx_dy = self.dx_dy_arrays[i]
                weights = self.weight_arrays[i]
                
                # Extract phases for this electrode and its neighbors
                phases_center = phase_data_gpu[i]
                phases_nbrs = phase_data_gpu[nbrs].T  # shape: (n_timepoints, n_neighbors)
                
                if self.use_gpu:
                    # Move to GPU
                    gpu_transfer_start = time.time()
                    dx_dy_gpu = self.xp.asarray(dx_dy)
                    weights_gpu = self.xp.asarray(weights)
                    log(f"Electrode {i} data GPU transfer: {time.time() - gpu_transfer_start:.6f} seconds", is_debug=True)
                else:
                    dx_dy_gpu = dx_dy
                    weights_gpu = weights
                
                # Compute phase differences with wrapping
                phase_diff_start = time.time()
                phase_diffs = phases_nbrs - phases_center.reshape(-1, 1)
                phase_diffs = ((phase_diffs + self.xp.pi) % (2 * self.xp.pi)) - self.xp.pi
                log(f"Electrode {i} phase diff computation: {time.time() - phase_diff_start:.6f} seconds", is_debug=True)
                
                # For each timepoint
                for t in range(n_timepoints):
                    # Skip if all neighbors have zero/nan phases
                    if self.xp.all(self.xp.isnan(phase_diffs[t])) or self.xp.all(phase_diffs[t] == 0):
                        continue
                    
                    # Set up weighted least squares problem
                    A = dx_dy_gpu  # shape: (n_neighbors, 2)
                    b = phase_diffs[t]  # shape: (n_neighbors,)
                    w = weights_gpu  # shape: (n_neighbors,)
                    
                    # Weighted design matrix
                    w_reshape = w.reshape(-1, 1)
                    ATA = A.T @ (w_reshape * A) + self.ridge_lambda * self.xp.eye(2)
                    ATb = A.T @ (w * b)
                    
                    try:
                        # Solve least squares problem
                        g = self.xp.linalg.solve(ATA, ATb)
                        grad_x_batch[t, i] = g[0]
                        grad_y_batch[t, i] = g[1]
                    except:
                        # Skip if singular
                        pass
                
                electrode_time = time.time() - electrode_start
                electrode_time_total += electrode_time
                electrode_count += 1
                
                if debug and i % 50 == 0:
                    log(f"Processed electrode {i}/{len(valid_electrodes)} in {electrode_time:.6f} seconds", is_debug=True)
            
            if electrode_count > 0:
                log(f"Average time per electrode: {electrode_time_total/electrode_count:.6f} seconds", is_debug=True)
            
            # Move back to CPU if necessary
            if self.use_gpu:
                cpu_transfer_start = time.time()
                log(f"Moving gradient data back to CPU...", is_debug=True)
                grad_x_batch = self.xp.asnumpy(grad_x_batch)
                grad_y_batch = self.xp.asnumpy(grad_y_batch)
                log(f"CPU transfer completed in {time.time() - cpu_transfer_start:.4f} seconds", is_debug=True)
            
            log(f"Batch gradient computation completed in {time.time() - compute_start:.4f} seconds")
            
            return grad_x_batch, grad_y_batch
            
        def compute_pgd_batch(self, phase_data_batch):
            """
            Compute PGD values for a batch of time points.
            
            [Method docstring remains the same as in notebook]
            """
            pgd_start = time.time()
            log(f"Computing PGD for batch of {phase_data_batch.shape[1]} timepoints...", is_debug=True)
            
            # Compute gradients
            grad_start = time.time()
            grad_x_batch, grad_y_batch = self.compute_batch(phase_data_batch)
            log(f"Gradient computation time: {time.time() - grad_start:.4f} seconds", is_debug=True)
            
            n_timepoints, n_electrodes = grad_x_batch.shape
            
            # Initialize PGD values
            pgd_values = np.zeros(n_timepoints, dtype=np.float32)
            
            # Compute PGD for each timepoint
            pgd_calc_start = time.time()
            
            for t in range(n_timepoints):
                # Stack gradients and compute magnitudes
                gradients = np.column_stack((grad_x_batch[t], grad_y_batch[t]))
                grad_magnitudes = np.sqrt(np.sum(gradients**2, axis=1))
                
                # Mask out invalid gradients
                valid_mask = grad_magnitudes >= self.min_gradient
                valid_gradients = gradients[valid_mask]
                valid_magnitudes = grad_magnitudes[valid_mask]
                
                if len(valid_gradients) > 0:
                    # Calculate PGD: ||∇φ|| / ||∇φ||
                    mean_gradient = np.mean(valid_gradients, axis=0)
                    mean_gradient_magnitude = np.linalg.norm(mean_gradient)
                    mean_magnitude = np.mean(valid_magnitudes)
                    
                    if mean_magnitude > 0:
                        pgd_values[t] = mean_gradient_magnitude / mean_magnitude
            
            log(f"PGD calculation time: {time.time() - pgd_calc_start:.4f} seconds", is_debug=True)
            log(f"Total PGD computation time: {time.time() - pgd_start:.4f} seconds", is_debug=True)
            
            # Report statistics about PGD values
            non_zero_pgd = pgd_values[pgd_values > 0]
            if len(non_zero_pgd) > 0:
                log(f"PGD statistics - Mean: {np.mean(non_zero_pgd):.4f}, Max: {np.max(non_zero_pgd):.4f}, "
                    f"Min: {np.min(non_zero_pgd):.4f}, Non-zero: {len(non_zero_pgd)}/{len(pgd_values)}", is_debug=True)
            else:
                log(f"WARNING: No non-zero PGD values computed in this batch!", is_debug=True)
            
            return pgd_values
    
    # Initialize the GPU-accelerated gradient calculator
    init_start = time.time()
    log("Initializing GPU-accelerated gradient calculator...")
    grad_calc = GPURegularizedGradient(
        lfp_processor.locations, 
        min_gradient=min_gradient,
        use_gpu=use_gpu
    )
    log(f"Gradient calculator initialization completed in {time.time() - init_start:.4f} seconds")
    log(f"Memory usage after initialization: {get_memory_usage():.2f} MB")
    
    # Process data in batches for better GPU utilization
    n_times = len(range(0, window_length, downsample_factor))
    pgd_values = np.zeros(n_times, dtype=np.float32)
    
    start_time = time.time()
    log(f"Computing PGD for {n_times} timepoints with batch size {batch_size}...")
    
    # Create downsampled indices
    downsampled_indices = list(range(0, window_length, downsample_factor))
    if len(downsampled_indices) > n_times:
        downsampled_indices = downsampled_indices[:n_times]
    
    # Process using GPU or threading
    if use_gpu:
        # Process in batches for GPU
        batch_processing_start = time.time()
        log(f"Starting GPU batch processing for {n_times} timepoints...")
        
        n_batches = (n_times + batch_size - 1) // batch_size
        total_batch_time = 0
        
        for batch_idx in range(n_batches):
            batch_start = time.time()
            idxs, pgd_batch = _pgd_process_batch(batch_idx, batch_size, phase_data, downsampled_indices, grad_calc, verbose, debug)
            batch_time = time.time() - batch_start
            total_batch_time += batch_time
            
            if len(idxs) > 0:
                pgd_values[idxs[0]:idxs[-1]+1] = pgd_batch
            
            # Print progress
            if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
                progress = (batch_idx+1)/n_batches*100
                eta = (total_batch_time/(batch_idx+1)) * (n_batches-batch_idx-1) if batch_idx < n_batches-1 else 0
                log(f"Processed batch {batch_idx+1}/{n_batches} ({progress:.1f}%) - "
                    f"Last batch: {batch_time:.2f}s - ETA: {eta:.2f}s")
            
            # Report GPU memory usage periodically
            if use_gpu and (batch_idx % 10 == 0 or batch_idx == n_batches - 1):
                try:
                    import cupy as cp
                    device = cp.cuda.Device()
                    log(f"GPU Memory: Used {(device.mem_info[1] - device.mem_info[0]) / 1024**3:.2f} GB, "
                        f"Free {device.mem_info[0] / 1024**3:.2f} GB")
                except (ImportError, AttributeError):
                    pass
        
        log(f"GPU batch processing completed in {time.time() - batch_processing_start:.4f} seconds")
    else:
        # Use threading for CPU parallelization
        import threading
        if n_processes is None:
            n_processes = os.cpu_count()
        
        thread_start = time.time()
        log(f"Using {n_processes} threads for parallel computation")
        
        # Create appropriately sized batches for threading
        # For threading, we don't need to make batches as small as for multiprocessing
        thread_batch_size = batch_size  # Can keep the original batch size
        n_batches = (n_times + thread_batch_size - 1) // thread_batch_size
        
        log(f"Splitting work into {n_batches} batches with {thread_batch_size} timepoints per batch")
        
        # This is where threading happens - key change from multiprocessing!
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_processes) as executor:
            futures = []
            # Submit tasks for each batch
            for batch_idx in range(n_batches):
                futures.append(executor.submit(
                    _pgd_process_batch, batch_idx, thread_batch_size, phase_data, 
                    downsampled_indices, grad_calc, verbose, debug
                ))
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                idxs, pgd_batch = future.result()
                if len(idxs) > 0:
                    pgd_values[idxs[0]:idxs[-1]+1] = pgd_batch
                
                # Print progress
                completed += 1
                if completed % 5 == 0 or completed == len(futures):
                    progress = completed/len(futures)*100
                    log(f"Processed {completed}/{len(futures)} batches ({progress:.1f}%)")
        
        log(f"Threaded computation completed in {time.time() - thread_start:.4f} seconds")
    
    log(f"PGD computation completed in {time.time() - start_time:.4f} seconds")
    log(f"Memory usage after PGD computation: {get_memory_usage():.2f} MB")
    
    # Apply Gaussian smoothing
    smoothing_start = time.time()
    log(f"Applying Gaussian smoothing with sigma={smoothing_sigma}...")
    smoothed_pgd = gaussian_filter1d(pgd_values, sigma=smoothing_sigma)
    log(f"Smoothing completed in {time.time() - smoothing_start:.4f} seconds")
    
    # Peak detection
    peak_detection_start = time.time()
    log(f"Detecting PGD peaks...")
    
    # Calculate threshold based on RMS
    rms = np.sqrt(np.mean(smoothed_pgd**2))
    peak_threshold = rms * threshold
    
    # Initialize pgd peaks dictionary
    peaks = {
        'peak_times': [],
        'peak_heights': [],
        'intervals': [],
        'durations': [],
        'threshold': peak_threshold
    }
    
    log(f"Using detection threshold: {peak_threshold:.4f} (RMS={rms:.4f}, multiplier={threshold})")
    
    # Find peaks above threshold
    find_peaks_start = time.time()
    peak_indices, peak_properties = find_peaks(
        smoothed_pgd, 
        height=peak_threshold, 
        distance=int(min_interval * lfp_processor.fs / downsample_factor)
    )
    log(f"Found {len(peak_indices)} raw peaks above threshold. Peak finding time: {time.time() - find_peaks_start:.4f} seconds")
    
    if len(peak_indices) > 0:
        # Calculate widths for each peak
        widths_start = time.time()
        widths, width_heights, left_ips, right_ips = peak_widths(
            smoothed_pgd, peak_indices, rel_height=0.9)
        log(f"Peak width calculation time: {time.time() - widths_start:.4f} seconds")
        
        # Convert indices to times
        peak_times = time_points[peak_indices]
        peak_heights = smoothed_pgd[peak_indices]
        
        # Create time interpolator for accurate conversion between indices and times
        interp_start = time.time()
        time_interpolator = interp1d(
            np.arange(len(time_points)), 
            time_points, 
            bounds_error=False, 
            fill_value="extrapolate"
        )
        log(f"Time interpolator creation time: {time.time() - interp_start:.4f} seconds")
        
        # Convert left and right indices to times
        left_times = time_interpolator(left_ips)
        right_times = time_interpolator(right_ips)
        
        # Filter peaks based on minimum duration
        filter_start = time.time()
        valid_peaks = []
        for i, (left, right) in enumerate(zip(left_times, right_times)):
            duration = right - left
            if duration >= min_duration:
                valid_peaks.append(i)
        
        log(f"Peak filtering time: {time.time() - filter_start:.4f} seconds")
        log(f"Found {len(valid_peaks)} peaks after filtering for minimum duration of {min_duration}s")
        
        # Store peak information for valid peaks
        if valid_peaks:
            peaks['peak_times'] = peak_times[valid_peaks]
            peaks['peak_heights'] = peak_heights[valid_peaks]
            peaks['intervals'] = list(zip(left_times[valid_peaks], right_times[valid_peaks]))
            peaks['durations'] = right_times[valid_peaks] - left_times[valid_peaks]
            
            log(f"Peak statistics: Mean duration={np.mean(peaks['durations']):.3f}s, "
                f"Mean height={np.mean(peaks['peak_heights']):.4f}")
    
    log(f"Peak detection completed in {time.time() - peak_detection_start:.4f} seconds")
    
    # Add raw data if requested
    if return_raw:
        peaks['time_bins'] = time_points
        peaks['pgd'] = smoothed_pgd
        log(f"Added raw data to results")
    
    # Plot results if requested
    if plot_results:
        plot_start = time.time()
        log(f"Generating PGD peak visualization...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot PGD
        ax.plot(time_points, smoothed_pgd, color=color, linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Phase Gradient Directionality', fontsize=12, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        
        # Add fill under curve for better visibility
        ax.fill_between(time_points, 0, smoothed_pgd, color=color, alpha=0.2)
        
        # Plot threshold line
        ax.axhline(y=peak_threshold, color='gray', linestyle='--', alpha=0.7, 
                  label=f'Threshold ({threshold}×RMS)')
        
        # Highlight detected peaks
        if 'intervals' in peaks and peaks['intervals']:
            highlight_start = time.time()
            log(f"Highlighting {len(peaks['intervals'])} peaks in plot...", is_debug=True)
            
            for i, (start, end) in enumerate(peaks['intervals']):
                ax.axvspan(start, end, color=highlight_color, alpha=highlight_alpha)
                
                # Add peak marker
                if len(valid_peaks) > 0:
                    peak_idx = np.where(peaks['peak_times'] == peak_times[valid_peaks][i])[0][0]
                    peak_time = peaks['peak_times'][peak_idx]
                    peak_value = peaks['peak_heights'][peak_idx]
                    ax.plot(peak_time, peak_value, 'o', color=highlight_color, 
                           markersize=8, alpha=0.8)
            
            log(f"Peak highlighting time: {time.time() - highlight_start:.4f} seconds", is_debug=True)
        
        # Set title
        if title is None:
            title = f"{data_type.upper()} Phase Gradient Directionality Peaks"
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
        
        # Store figure and axis in the result
        peaks['fig'] = fig
        peaks['ax'] = ax
        
        log(f"Plot generation completed in {time.time() - plot_start:.4f} seconds")
    
    # Print performance summary
    total_time = time.time() - start_time_total
    memory_used = get_memory_usage() - initial_memory
    
    log(f"\n===== Performance Summary =====")
    log(f"Total processing time: {total_time:.2f} seconds")
    log(f"Memory used: {memory_used:.2f} MB")
    log(f"Timepoints processed: {n_times} ({n_times/total_time:.2f} timepoints/second)")
    log(f"Detected {len(peaks['peak_times'])} PGD peaks above threshold {peak_threshold:.4f}")
    
    # Detailed timing breakdown if available
    if 'phase_data_time' in locals():
        log(f"\n===== Timing Breakdown =====")
        log(f"Phase data extraction: {phase_data_time:.2f}s ({phase_data_time/total_time*100:.1f}%)")
        log(f"Gradient calculator initialization: {time.time() - init_start:.2f}s ({(time.time() - init_start)/total_time*100:.1f}%)")
        log(f"PGD computation: {time.time() - start_time:.2f}s ({(time.time() - start_time)/total_time*100:.1f}%)")
        log(f"Peak detection: {time.time() - peak_detection_start:.2f}s ({(time.time() - peak_detection_start)/total_time*100:.1f}%)")
        if plot_results:
            log(f"Plot generation: {time.time() - plot_start:.2f}s ({(time.time() - plot_start)/total_time*100:.1f}%)")
    
    log(f"==============================\n")
    
    return peaks


def main():
    """
    Main function executing the complete spectral coherence analysis workflow.
    
    Workflow:
    1. Load data and initialize processor
    2. Detect energy peaks → Extract intervals → Spectral analysis
    3. Detect PGD peaks → Extract intervals → Spectral analysis
    """
    print("Starting Spectral Coherence Analysis")
    print("=" * 50)
    
    # Data paths
    folder_dir = "D:\\dvorak\\" 
    raw_path = f"{folder_dir}dvorak_3.raw.h5"
    spike_path = f"{folder_dir}dvorak_3_acqm.zip" 
    lfp_path = "C:\\Users\\visio\\OneDrive\\Desktop\\special_bandsdvorak_3_special_bands_20_[0, 120].npz"
    
    # Load data
    print("\nLoading data...")
    train, neuron_data, config, fs = load_curation(spike_path)
    train = [np.array(t)*1000 for t in train]
    spike_data = analysis.SpikeData(train, neuron_data={0: neuron_data})
    
    # Raw data
    version, time_stamp, config_df, raster_df = load_info_maxwell(raw_path)
    
    # LFP data
    waves = np.load(lfp_path)
    
    # Initialize processor
    print("\nInitializing LFPDataProcessor...")
    x_mod = np.load("D:\\testing_new_lfp_class\\wave_prop\\ladybugs\\ai_examples\\figures_code\\scripts\\x_mod.npy")
    y_mod = np.load("D:\\testing_new_lfp_class\\wave_prop\\ladybugs\\ai_examples\\figures_code\\scripts\\y_mod.npy")
    processor = LFPDataProcessor(waves, x_mod, y_mod, config_df)
    processor.add_frequency_band(1, 30, band_name="sharpWave", use_gpu=True, store_analytical=True)
    
    # Define optogenetic intervals
    opto_intervals = [(26.5, 27), (30, 40), (60, 70), (90, 100.5)]
    
    # Get current directory for saving outputs
    current_dir = os.getcwd()
    
    # WORKFLOW 1: Detect energy peaks → Extract intervals → Spectral analysis
    print("\n" + "="*50)
    print("WORKFLOW 1: Energy-based event detection and analysis")
    print("="*50)
    
    # Detect LFP energy peaks
    print("\nDetecting LFP energy peaks...")
    lfp_energy_peaks = detect_energy_peaks(
        processor,
        data_type='lfp',
        window_start=14000,
        window_length=95000,
        smoothing_sigma=20,
        threshold=1.3,
        min_duration=0.1,
        plot_results=True,
        optogenetic_intervals=opto_intervals,
    )
    
    # Extract intervals from energy peaks
    lfp_intervals = lfp_energy_peaks['intervals']
    print(f"Found {len(lfp_intervals)} LFP energy peak intervals")
    
    # Perform spectral analysis on all events (full spectrum)
    print("\nPerforming spectral analysis on all LFP energy events (1-150 Hz)...")
    save_path_full_spectra = os.path.join(current_dir, "coherence_spectral_analysis_all_events_all_freqs.png")
    
    fig1, axes1, results1 = plot_coherence_spectral_analysis(
        processor,
        lfp_intervals[0:10],  # First 10 events
        freq_range=(1, 150),
        coherence_method='kl_divergence',
        n_wavelets=90,
        wavelet_width=10,
        smoothing_sigma=4,
        save_path=save_path_full_spectra,
        lpf_cutoff=170,
        nfft=4096,
        max_nperseg=1024,
        overlap_fraction=0.8,
        scaling_factor=8,
        detrend_method='linear',
        window_type='hann',
        figure_size=(20, 15),
        x_left_lim=2.5
    )
    
    # Perform spectral analysis focusing on sharp wave band
    print("\nPerforming spectral analysis on LFP energy events (2-30 Hz focus)...")
    save_path_sw = os.path.join(current_dir, "coherence_spectral_analysis_all_events_sw.png")
    
    fig2, axes2, results2 = plot_coherence_spectral_analysis(
        processor,
        lfp_intervals,
        freq_range=(2, 30),
        coherence_method='kl_divergence',
        n_wavelets=80,
        wavelet_width=12,
        smoothing_sigma=3,
        save_path=save_path_sw,
        lpf_cutoff=35,
        nfft=4096,
        max_nperseg=2048,
        overlap_fraction=0.8,
        scaling_factor=10,
        detrend_method='linear',
        window_type='hann',
        figure_size=(20, 15)
    )
    
    # Perform spectral analysis focusing on ripple band
    print("\nPerforming spectral analysis on LFP energy events (50-300 Hz focus)...")
    save_path_ripples = os.path.join(current_dir, "coherence_spectral_analysis_all_events_ripples_20.png")
    
    fig3, axes3, results3 = plot_coherence_spectral_analysis(
        processor,
        lfp_intervals,
        freq_range=(50, 300),
        coherence_method='kl_divergence',
        n_wavelets=100,
        wavelet_width=8,
        smoothing_sigma=5,
        save_path=save_path_ripples,
        lpf_cutoff=350,
        nfft=2048,
        max_nperseg=512,
        overlap_fraction=0.75,
        scaling_factor=5,
        detrend_method='linear',
        window_type='hann',
        figure_size=(20, 15),
        show_fband_markers=True,
        dpi=300
    )
    
    # WORKFLOW 2: Detect PGD peaks → Extract intervals → Spectral analysis
    print("\n" + "="*50)
    print("WORKFLOW 2: PGD-based event detection and analysis")
    print("="*50)
    
    # Detect sharp wave PGD peaks
    print("\nDetecting sharp wave PGD peaks...")
    window_start = 14000
    window_length = 14100
    
    sharp_wave_pgd = detect_pgd_peaks_accelerated(
        processor,
        data_type='sharpWave',
        window_start=window_start,
        window_length=window_length,
        threshold=1.4,
        smoothing_sigma=20,
        min_duration=0.1,
        plot_results=True,
        color='blue',
        highlight_color='lightskyblue'
    )
    
    # Extract intervals from PGD peaks
    sw_intervals = sharp_wave_pgd['intervals']
    print(f"Found {len(sw_intervals)} sharp wave PGD peak intervals")
    
    # Perform spectral analysis on PGD events
    if len(sw_intervals) > 0:
        print("\nPerforming spectral analysis on sharp wave PGD events...")
        save_path_sw_test = os.path.join(current_dir, "coherence_spectral_analysis_sw_events_wide_spectra_test.png")
        
        fig4, axes4, results4 = plot_coherence_spectral_analysis(
            processor,
            sw_intervals[0:5],  # First 5 PGD events
            freq_range=(1, 150),
            coherence_method='kl_divergence',
            n_wavelets=90,
            wavelet_width=10,
            smoothing_sigma=4,
            save_path=save_path_sw_test,
            lpf_cutoff=170,
            nfft=4096,
            max_nperseg=1024,
            overlap_fraction=0.8,
            scaling_factor=8,
            detrend_method='linear',
            window_type='hann',
            figure_size=(20, 15),
            x_left_lim=2.5
        )
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("="*50)
    
    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()