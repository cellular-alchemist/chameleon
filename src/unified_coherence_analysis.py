#!/usr/bin/env python3
"""
Unified Traveling Waves Analysis Pipeline
Combines planar waves, ripples, and spectral coherence analysis
with optimized computation flow
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import functools
import pickle
from pathlib import Path
import gc

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class WaveAnalysisConfig:
    """Central configuration for all analyses"""
    # Gradient computation
    max_neighbor_dist: float = 50.0
    spatial_sigma: float = 45.0
    ridge_lambda: float = 1e-5
    coverage_angle_gap: float = 120.0
    min_gradient: float = 1e-5
    
    # PGD computation
    pgd_downsample_factor: int = 1
    pgd_smoothing_sigma: float = 15
    pgd_batch_size: int = 100
    
    # Event detection
    energy_threshold: float = 1.3
    pgd_threshold: float = 1.4
    min_event_duration: float = 0.1
    min_event_interval: float = 0.2
    
    # Ripple detection
    ripple_low_threshold: float = 3.5
    ripple_high_threshold: float = 5.0
    ripple_min_duration: int = 20
    ripple_max_duration: int = 200
    
    # Spectral analysis
    wavelet_freqs: Tuple[float, float] = (1, 150)
    n_wavelets: int = 90
    wavelet_width: float = 10.0
    coherence_method: str = 'kl_divergence'
    
    # Performance
    use_gpu: bool = True
    cache_computations: bool = True
    n_processes: Optional[int] = None


@dataclass 
class ComputationCache:
    """Cache for expensive computations"""
    pgd_data: Dict[str, np.ndarray] = None
    gradient_calculator: 'UnifiedGradientCalculator' = None
    phase_data: Dict[str, np.ndarray] = None
    analytical_data: Dict[str, dict] = None
    spectral_data: Dict[str, dict] = None
    
    def save(self, filepath: Path):
        """Save cache to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Path):
        """Load cache from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# UNIFIED GRADIENT CALCULATOR
# ============================================================================

class UnifiedGradientCalculator:
    """
    Single gradient calculator used across all analyses
    Combines functionality from both pipelines
    """
    
    def __init__(self, locations: np.ndarray, config: WaveAnalysisConfig):
        self.locations = locations
        self.config = config
        self.n_electrodes = len(locations)
        
        # Precompute all neighbor relationships once
        self._precompute_neighbors()
        
        # Cache for frequently used computations
        self._gradient_cache = {}
    
    @property
    def cfg(self) -> dict:
        """
        Provide backward compatibility with original RegularizedGradient interface.
        Returns a dictionary with uppercase keys as expected by plot_pgd_wave_analysis_optimized.
        """
        return {
            'MAX_NEIGHBOR_DIST': self.config.max_neighbor_dist,
            'SPATIAL_SIGMA': self.config.spatial_sigma,
            'RIDGE_LAMBDA': self.config.ridge_lambda,
            'COVERAGE_ANGLE_GAP': self.config.coverage_angle_gap,
            'MIN_GRADIENT': self.config.min_gradient,
            'V_MIN': 0.0,  # Default from original WAVE_CONFIG
            'V_MAX': 300_000.0,  # Default from original WAVE_CONFIG (μm/s)
        }
    
    def _precompute_neighbors(self):
        """Precompute neighbor relationships once for all electrodes"""
        from scipy.spatial import cKDTree
        
        print("Precomputing electrode neighbor relationships...")
        
        # Build KD-tree
        self.tree = cKDTree(self.locations)
        
        # Find neighbors for all electrodes
        dists, idxs = self.tree.query(
            self.locations,
            k=min(self.n_electrodes, 20),
            distance_upper_bound=self.config.max_neighbor_dist
        )
        
        # Store preprocessed neighbor data
        self.neighbor_indices = []
        self.neighbor_weights = []
        self.neighbor_matrices = []
        self.valid_electrodes = np.zeros(self.n_electrodes, dtype=bool)
        
        for i in range(self.n_electrodes):
            # Get valid neighbors
            valid = dists[i] < np.inf
            nbrs = idxs[i][valid]
            dist = dists[i][valid]
            
            # Exclude self
            mask = nbrs != i
            nbrs = nbrs[mask]
            dist = dist[mask]
            
            if len(nbrs) >= 3:
                # Compute spatial relationships
                dx = self.locations[nbrs, 0] - self.locations[i, 0]
                dy = self.locations[nbrs, 1] - self.locations[i, 1]
                
                # Check angular coverage
                angles = np.degrees(np.mod(np.arctan2(dy, dx), 2*np.pi))
                angles_sorted = np.sort(angles)
                gaps = np.diff(np.concatenate([angles_sorted, angles_sorted[:1] + 360]))
                
                if np.max(gaps) <= self.config.coverage_angle_gap:
                    # Precompute matrices for least squares
                    A = np.column_stack([dx, dy])
                    weights = np.exp(-0.5 * (dist / self.config.spatial_sigma)**2)
                    
                    # Weighted least squares matrix
                    w_diag = np.diag(weights)
                    ATA = A.T @ w_diag @ A + self.config.ridge_lambda * np.eye(2)
                    
                    # Store precomputed data
                    self.neighbor_indices.append(nbrs)
                    self.neighbor_weights.append(weights)
                    self.neighbor_matrices.append({
                        'A': A,
                        'ATA_inv': np.linalg.inv(ATA),
                        'weights': weights
                    })
                    self.valid_electrodes[i] = True
                    continue
            
            # Invalid electrode
            self.neighbor_indices.append(np.array([], dtype=int))
            self.neighbor_weights.append(np.array([]))
            self.neighbor_matrices.append(None)
        
        print(f"Valid electrodes: {np.sum(self.valid_electrodes)}/{self.n_electrodes}")
    
    def compute_gradients_batch(self, phase_data: np.ndarray, 
                               use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for a batch of timepoints
        Returns: (grad_x, grad_y) arrays of shape (n_electrodes, n_timepoints)
        """
        n_electrodes, n_timepoints = phase_data.shape
        
        # Check cache
        cache_key = hash(phase_data.tobytes()) if use_cache else None
        if use_cache and cache_key in self._gradient_cache:
            return self._gradient_cache[cache_key]
        
        # Initialize output arrays
        grad_x = np.zeros((n_electrodes, n_timepoints), dtype=np.float32)
        grad_y = np.zeros((n_electrodes, n_timepoints), dtype=np.float32)
        
        # Process each valid electrode
        for i in np.where(self.valid_electrodes)[0]:
            nbrs = self.neighbor_indices[i]
            matrix_data = self.neighbor_matrices[i]
            
            # Compute phase differences for all timepoints
            phase_diffs = phase_data[nbrs, :] - phase_data[i, :]
            phase_diffs = ((phase_diffs + np.pi) % (2 * np.pi)) - np.pi
            
            # Solve weighted least squares for each timepoint
            A = matrix_data['A']
            ATA_inv = matrix_data['ATA_inv']
            weights = matrix_data['weights']
            
            for t in range(n_timepoints):
                ATb = A.T @ (weights * phase_diffs[:, t])
                g = ATA_inv @ ATb
                grad_x[i, t] = g[0]
                grad_y[i, t] = g[1]
        
        # Cache result
        if use_cache and cache_key is not None:
            self._gradient_cache[cache_key] = (grad_x, grad_y)
        
        return grad_x, grad_y
    
    def compute_pgd_values(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """
        Compute PGD from gradient components
        Returns: PGD values for each timepoint
        """
        # Compute gradient magnitudes
        grad_magnitudes = np.sqrt(grad_x**2 + grad_y**2)
        
        # Mask valid gradients
        valid_mask = grad_magnitudes >= self.config.min_gradient
        
        # Compute PGD for each timepoint
        n_timepoints = grad_x.shape[1]
        pgd_values = np.zeros(n_timepoints, dtype=np.float32)
        
        for t in range(n_timepoints):
            valid = valid_mask[:, t]
            if np.sum(valid) >= 3:
                mean_grad = np.array([
                    np.mean(grad_x[valid, t]),
                    np.mean(grad_y[valid, t])
                ])
                mean_mag = np.mean(grad_magnitudes[valid, t])
                
                if mean_mag > 0:
                    pgd_values[t] = np.linalg.norm(mean_grad) / mean_mag
        
        return pgd_values


# ============================================================================
# UNIFIED EVENT DETECTOR
# ============================================================================

class UnifiedEventDetector:
    """
    Unified event detection for all event types
    Prevents duplicate computation of similar events
    """
    
    def __init__(self, lfp_processor, config: WaveAnalysisConfig):
        self.processor = lfp_processor
        self.config = config
        self.fs = lfp_processor.fs
        
    def detect_all_events(self, window_start: int, window_length: int,
                         optogenetic_intervals: Optional[List[Tuple[float, float]]] = None,
                         cache: Optional[ComputationCache] = None) -> Dict:
        """
        Detect all event types in a single pass through the data
        Returns dictionary with all event types
        """
        results = {
            'energy_events': {},
            'pgd_events': {},
            'ripple_events': {},
            'metadata': {
                'window_start': window_start,
                'window_length': window_length,
                'window_end': window_start + window_length
            }
        }
        
        # 1. Detect energy-based events for all frequency bands
        print("Detecting energy-based events...")
        for band_name in ['lfp', 'sharpWave', 'narrowRipples']:
            if band_name in self.processor.waves:
                events = self._detect_energy_events(
                    band_name, window_start, window_length, 
                    optogenetic_intervals
                )
                results['energy_events'][band_name] = events
        
        # 2. Detect PGD-based events (compute PGD only once)
        print("Computing PGD for all frequency bands...")
        
        # Use cached gradient calculator if available
        if cache and cache.gradient_calculator:
            grad_calc = cache.gradient_calculator
        else:
            grad_calc = UnifiedGradientCalculator(
                self.processor.locations, 
                self.config
            )
            if cache:
                cache.gradient_calculator = grad_calc
        
        # Compute PGD for each frequency band
        for band_name in ['sharpWave', 'narrowRipples']:
            if band_name in self.processor.waves:
                print(f"  Processing {band_name} band...")
                
                # Check cache first
                if cache and cache.pgd_data and band_name in cache.pgd_data:
                    pgd_data = cache.pgd_data[band_name]
                else:
                    pgd_data = self._compute_pgd_for_band(
                        band_name, window_start, window_length, grad_calc
                    )
                    if cache:
                        if cache.pgd_data is None:
                            cache.pgd_data = {}
                        cache.pgd_data[band_name] = pgd_data
                
                # Detect peaks in PGD
                events = self._detect_pgd_peaks(pgd_data, band_name)
                results['pgd_events'][band_name] = events
        
        # 3. Detect ripple events
        print("Detecting ripple events...")
        ripple_events = self.processor.detect_ripples(
            narrowband_key='narrowRipples',
            wideband_key='broadRipples',
            low_threshold=self.config.ripple_low_threshold,
            high_threshold=self.config.ripple_high_threshold,
            min_duration=self.config.ripple_min_duration,
            max_duration=self.config.ripple_max_duration,
            sharp_wave_threshold=3,
            sharp_wave_band=(0.1, 30),
            require_sharp_wave=True
        )
        results['ripple_events'] = ripple_events
        
        return results
    
    def _detect_energy_events(self, band_name: str, window_start: int, 
                            window_length: int, optogenetic_intervals: Optional[List] = None) -> Dict:
        """Detect energy-based events"""
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks, peak_widths
        
        # Compute instantaneous energy
        time_points, mean_energy, _ = self.processor.compute_instantaneous_energy(
            band_name, window_start, window_length, 
            self.config.pgd_downsample_factor
        )
        
        # Smooth energy
        smoothed_energy = gaussian_filter1d(
            mean_energy, 
            sigma=self.config.pgd_smoothing_sigma
        )
        
        # Detect peaks
        rms = np.sqrt(np.mean(smoothed_energy**2))
        threshold = rms * self.config.energy_threshold
        
        peak_indices, _ = find_peaks(
            smoothed_energy,
            height=threshold,
            distance=int(self.config.min_event_interval * self.fs / 
                       self.config.pgd_downsample_factor)
        )
        
        # Calculate peak properties
        events = {
            'time_points': time_points,
            'energy': smoothed_energy,
            'threshold': threshold,
            'peak_times': [],
            'peak_heights': [],
            'intervals': [],
            'durations': []
        }
        
        if len(peak_indices) > 0:
            widths, _, left_ips, right_ips = peak_widths(
                smoothed_energy, peak_indices, rel_height=0.9
            )
            
            # Convert to times and filter
            from scipy.interpolate import interp1d
            time_interp = interp1d(
                np.arange(len(time_points)), 
                time_points,
                bounds_error=False, 
                fill_value="extrapolate"
            )
            
            left_times = time_interp(left_ips)
            right_times = time_interp(right_ips)
            peak_times = time_points[peak_indices]
            
            # Filter by duration and optogenetic intervals
            for i, (left, right, peak_time) in enumerate(zip(left_times, right_times, peak_times)):
                duration = right - left
                
                # Check duration
                if duration < self.config.min_event_duration:
                    continue
                
                # Check optogenetic overlap
                if optogenetic_intervals:
                    overlap = any(
                        (left < opto_end and right > opto_start)
                        for opto_start, opto_end in optogenetic_intervals
                    )
                    if overlap:
                        continue
                
                # Add valid event
                events['peak_times'].append(peak_time)
                events['peak_heights'].append(smoothed_energy[peak_indices[i]])
                events['intervals'].append((left, right))
                events['durations'].append(duration)
        
        # Convert lists to arrays
        for key in ['peak_times', 'peak_heights', 'durations']:
            events[key] = np.array(events[key])
        
        return events
    
    def _compute_pgd_for_band(self, band_name: str, window_start: int, 
                            window_length: int, grad_calc: UnifiedGradientCalculator) -> Dict:
        """Compute PGD for a frequency band using streaming/chunked processing"""
        
        # Define chunk size based on available memory
        # Process 10 seconds of data at a time (at 20kHz = 200k samples)
        chunk_samples = min(40000, window_length)
        n_chunks = (window_length + chunk_samples - 1) // chunk_samples
        
        print(f"Processing {band_name} PGD in {n_chunks} chunks of {chunk_samples} samples")
        
        # Pre-allocate output arrays
        total_downsampled_points = window_length // self.config.pgd_downsample_factor
        pgd_values = np.zeros(total_downsampled_points, dtype=np.float32)
        time_points = np.zeros(total_downsampled_points, dtype=np.float32)
        
        # Keep track of output position
        output_idx = 0
        
        # Process each chunk
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_samples
            chunk_end = min((chunk_idx + 1) * chunk_samples, window_length)
            chunk_size = chunk_end - chunk_start
            
            if chunk_size <= 0:
                continue
                
            print(f"  Processing chunk {chunk_idx+1}/{n_chunks} ({chunk_start}-{chunk_end})")
            
            # Get analytical data for just this chunk
            chunk_analytical = self.processor._get_analytical_data(
                band_name, 
                window_start + chunk_start, 
                window_start + chunk_end
            )
            chunk_phase = chunk_analytical['phase']
            
            # Downsample indices for this chunk
            chunk_indices = np.arange(0, chunk_size, self.config.pgd_downsample_factor)
            chunk_pgd_values = np.zeros(len(chunk_indices), dtype=np.float32)
            
            # Process in batches within the chunk
            n_timepoints = len(chunk_indices)
            n_batches = (n_timepoints + self.config.pgd_batch_size - 1) // self.config.pgd_batch_size
            
            for batch_idx in range(n_batches):
                batch_start_idx = batch_idx * self.config.pgd_batch_size
                batch_end_idx = min(batch_start_idx + self.config.pgd_batch_size, n_timepoints)
                
                # Get batch indices relative to chunk
                batch_indices = chunk_indices[batch_start_idx:batch_end_idx]
                phase_batch = chunk_phase[:, batch_indices]
                
                # Compute gradients
                grad_x, grad_y = grad_calc.compute_gradients_batch(phase_batch)
                
                # Compute PGD
                pgd_batch = grad_calc.compute_pgd_values(grad_x, grad_y)
                chunk_pgd_values[batch_start_idx:batch_end_idx] = pgd_batch
            
            # Store results in output arrays
            n_chunk_points = len(chunk_indices)
            pgd_values[output_idx:output_idx + n_chunk_points] = chunk_pgd_values
            
            # Calculate time points for this chunk
            chunk_time_points = (window_start + chunk_start + chunk_indices) / self.fs
            time_points[output_idx:output_idx + n_chunk_points] = chunk_time_points
            
            output_idx += n_chunk_points
            
            # Free memory from this chunk
            del chunk_analytical
            del chunk_phase
            del chunk_pgd_values
            gc.collect()
        
        # Trim arrays to actual size (in case of rounding)
        pgd_values = pgd_values[:output_idx]
        time_points = time_points[:output_idx]
        
        # Smooth PGD values
        from scipy.ndimage import gaussian_filter1d
        pgd_smooth = gaussian_filter1d(pgd_values, sigma=self.config.pgd_smoothing_sigma)
        
        # Return results without storing full phase data
        return {
            'time_points': time_points,
            'pgd_raw': pgd_values,
            'pgd_smooth': pgd_smooth,
            'grad_calc': grad_calc,
            'window_start': window_start,
            'window_end': window_start + window_length,
            'downsample_factor': self.config.pgd_downsample_factor,
            # Don't store phase_data or analytical_data to save memory!
        }
    
    def _detect_pgd_peaks(self, pgd_data: Dict, band_name: str) -> Dict:
        """Detect peaks in PGD data"""
        from scipy.signal import find_peaks, peak_widths
        
        pgd_smooth = pgd_data['pgd_smooth']
        time_points = pgd_data['time_points']
        
        # Calculate threshold
        rms = np.sqrt(np.mean(pgd_smooth**2))
        threshold = rms * self.config.pgd_threshold
        
        # Find peaks
        peak_indices, _ = find_peaks(
            pgd_smooth,
            height=threshold,
            distance=int(self.config.min_event_interval * self.fs / 
                       self.config.pgd_downsample_factor)
        )
        
        events = {
            'time_points': time_points,
            'pgd': pgd_smooth,
            'threshold': threshold,
            'peak_times': [],
            'peak_heights': [],
            'intervals': [],
            'durations': [],
            'pgd_data': pgd_data  # Store full PGD data for later use
        }
        
        if len(peak_indices) > 0:
            # Get peak properties
            widths, _, left_ips, right_ips = peak_widths(
                pgd_smooth, peak_indices, rel_height=0.3
            )
            
            # Convert to times
            from scipy.interpolate import interp1d
            time_interp = interp1d(
                np.arange(len(time_points)), 
                time_points,
                bounds_error=False, 
                fill_value="extrapolate"
            )
            
            left_times = time_interp(left_ips)
            right_times = time_interp(right_ips)
            peak_times = time_points[peak_indices]
            
            # Filter by duration
            for i, (left, right, peak_time) in enumerate(zip(left_times, right_times, peak_times)):
                duration = right - left
                
                if duration >= self.config.min_event_duration:
                    events['peak_times'].append(peak_time)
                    events['peak_heights'].append(pgd_smooth[peak_indices[i]])
                    events['intervals'].append((left, right))
                    events['durations'].append(duration)
        
        # Convert to arrays
        for key in ['peak_times', 'peak_heights', 'durations']:
            events[key] = np.array(events[key])
        
        return events


# ============================================================================
# UNIFIED ANALYSIS ENGINE
# ============================================================================

class UnifiedWaveAnalysis:
    """
    Main class that orchestrates all analyses with minimal redundancy
    """
    
    def __init__(self, lfp_processor, config: Optional[WaveAnalysisConfig] = None):
        self.processor = lfp_processor
        self.config = config or WaveAnalysisConfig()
        
        # Initialize components
        self.event_detector = UnifiedEventDetector(lfp_processor, self.config)
        self.cache = ComputationCache()
        
        # Results storage
        self.events = None
        self.analysis_results = {}
        
    def run_complete_analysis(self, window_start: int, window_length: int,
                            optogenetic_intervals: Optional[List] = None,
                            analyses_to_run: Optional[List[str]] = None) -> Dict:
        """
        Run complete analysis pipeline with all optimizations
        
        Args:
            window_start: Starting frame
            window_length: Number of frames to analyze  
            optogenetic_intervals: List of (start, end) times to exclude
            analyses_to_run: List of analyses to perform 
                           ['temporal', 'spectral', 'spatial', 'all']
        
        Returns:
            Dictionary with all results
        """
        if analyses_to_run is None:
            analyses_to_run = ['all']
        
        print("="*60)
        print("UNIFIED TRAVELING WAVES ANALYSIS PIPELINE")
        print("="*60)
        
        # Step 1: Detect all events in a single pass
        print("\nStep 1: Detecting all events...")
        self.events = self.event_detector.detect_all_events(
            window_start, window_length, 
            optogenetic_intervals, 
            self.cache
        )
        
        # Print event summary
        self._print_event_summary()
        
        # Step 2: Run requested analyses
        if 'all' in analyses_to_run or 'temporal' in analyses_to_run:
            print("\nStep 2: Running temporal analysis...")
            self.analysis_results['temporal'] = self._run_temporal_analysis()
        
        if 'all' in analyses_to_run or 'spectral' in analyses_to_run:
            print("\nStep 3: Running spectral analysis...")
            self.analysis_results['spectral'] = self._run_spectral_analysis()
        
        if 'all' in analyses_to_run or 'spatial' in analyses_to_run:
            print("\nStep 4: Running spatial analysis...")
            self.analysis_results['spatial'] = self._run_spatial_analysis()
        
        # Step 3: Generate integrated visualizations
        print("\nStep 5: Generating visualizations...")
        self.analysis_results['figures'] = self._generate_visualizations()
        
        # Step 4: Generate comprehensive report
        print("\nStep 6: Generating analysis report...")
        self.analysis_results['report'] = self._generate_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return self.analysis_results
    
    def _print_event_summary(self):
        """Print summary of detected events"""
        print("\nEvent Detection Summary:")
        print("-" * 40)
        
        # Energy events
        for band, events in self.events['energy_events'].items():
            n_events = len(events['peak_times'])
            print(f"  {band} energy events: {n_events}")
        
        # PGD events  
        for band, events in self.events['pgd_events'].items():
            n_events = len(events['peak_times'])
            print(f"  {band} PGD events: {n_events}")
        
        # Ripple events
        n_ripples = sum(len(self.events['ripple_events'].get(ch, [])) 
                       for ch in self.events['ripple_events'] 
                       if isinstance(ch, int))
        print(f"  Ripple events: {n_ripples}")
    
    def _run_temporal_analysis(self) -> Dict:
        """
        Analyze temporal relationships between different event types
        """
        from planar_waves_and_ripples import analyze_planar_waves_and_ripples_optimized
        
        results = {}
        
        # Analyze SW-ripple relationships
        if 'sharpWave' in self.events['pgd_events']:
            sw_pgd = self.events['pgd_events']['sharpWave']
            
            # Use cached PGD data
            pgd_data_dict = {
                band: events['pgd_data'] 
                for band, events in self.events['pgd_events'].items()
            }
            
            results['sw_ripple_analysis'] = analyze_planar_waves_and_ripples_optimized(
                self.processor,
                sw_pgd,
                self.events['ripple_events'],
                pgd_data_dict,
                bands=['sharpWave', 'narrowRipples'],
                window_size=0.5,
                smoothing_sigma=15,
                plot_individual=False,  # Skip individual plots for efficiency
                generate_summary=True
            )
        
        return results
    
    def _run_spectral_analysis(self) -> Dict:
        """
        Run spectral coherence analysis on detected events
        """
        from spectral_coherence_analysis import plot_coherence_spectral_analysis
        
        results = {}
        
        # Analyze different event types
        event_groups = [
            ('energy_lfp', self.events['energy_events'].get('lfp', {}).get('intervals', [])),
            ('energy_sw', self.events['energy_events'].get('sharpWave', {}).get('intervals', [])),
            ('pgd_sw', self.events['pgd_events'].get('sharpWave', {}).get('intervals', []))
        ]
        
        for event_type, intervals in event_groups:
            if len(intervals) > 0:
                # Limit to first 10 events for efficiency
                intervals_subset = intervals[:10] if len(intervals) > 10 else intervals
                
                fig, axes, spectral_results = plot_coherence_spectral_analysis(
                    self.processor,
                    intervals_subset,
                    freq_range=self.config.wavelet_freqs,
                    coherence_method=self.config.coherence_method,
                    n_wavelets=self.config.n_wavelets,
                    wavelet_width=self.config.wavelet_width
                )
                
                results[event_type] = spectral_results
                
                # Close figure to save memory
                import matplotlib.pyplot as plt
                plt.close(fig)
        
        return results
    
    def _run_spatial_analysis(self) -> Dict:
        """
        Analyze spatial properties of traveling waves
        """
        from planar_waves_and_ripples import plot_pgd_wave_analysis_optimized
        
        results = {}
        
        # Analyze wave properties for each PGD event type
        for band, events in self.events['pgd_events'].items():
            if len(events['peak_times']) > 0:
                figs, axes, wave_results = plot_pgd_wave_analysis_optimized(
                    self.processor,
                    events,
                    events['pgd_data'],
                    data_type=band,
                    use_joypy=False
                )
                
                results[band] = wave_results
                
                # Close figures to save memory
                import matplotlib.pyplot as plt
                for fig in figs.values():
                    plt.close(fig)
        
        return results
    
    def _generate_visualizations(self) -> Dict:
        """
        Generate integrated visualizations showing relationships across analyses
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        figures = {}
        
        # Create integrated summary figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Event timeline
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_event_timeline(ax1)
        
        # Panel 2: Event statistics
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_event_statistics(ax2)
        
        # Panel 3: Temporal relationships
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_temporal_relationships(ax3)
        
        # Panel 4: Spectral summary
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_spectral_summary(ax4)
        
        # Panel 5: Spatial properties
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_spatial_summary(ax5)
        
        # Panel 6: Key findings
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_key_findings(ax6)
        
        fig.suptitle('Unified Traveling Waves Analysis Summary', fontsize=16)
        figures['summary'] = fig
        
        return figures
    
    def _plot_event_timeline(self, ax):
        """Plot timeline of all detected events"""
        # Implementation would show all event types on a common timeline
        ax.set_title('Event Timeline')
        ax.set_xlabel('Time (s)')
        
        # Plot different event types at different y-levels
        y_levels = {'energy': 0, 'pgd': 1, 'ripple': 2}
        colors = {'energy': 'blue', 'pgd': 'green', 'ripple': 'red'}
        
        # Add events to timeline
        # ... (implementation details)
    
    def _plot_event_statistics(self, ax):
        """Plot event count statistics"""
        # Bar plot of event counts by type
        ax.set_title('Event Counts by Type')
        # ... (implementation details)
    
    def _plot_temporal_relationships(self, ax):
        """Plot temporal relationships between event types"""
        ax.set_title('Event Temporal Relationships')
        # ... (implementation details)
    
    def _plot_spectral_summary(self, ax):
        """Plot spectral analysis summary"""
        ax.set_title('Spectral Properties Summary')
        # ... (implementation details)
    
    def _plot_spatial_summary(self, ax):
        """Plot spatial analysis summary"""
        ax.set_title('Wave Propagation Properties')
        # ... (implementation details)
    
    def _plot_key_findings(self, ax):
        """Display key findings as text"""
        ax.set_title('Key Findings')
        ax.axis('off')
        
        findings_text = self._generate_key_findings()
        ax.text(0.1, 0.9, findings_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top')
    
    def _generate_key_findings(self) -> str:
        """Generate text summary of key findings"""
        findings = []
        
        # Add key statistics and relationships
        # ... (implementation details)
        
        return '\n'.join(findings)
    
    def _generate_report(self) -> str:
        """Generate comprehensive text report"""
        report = f"""
        ========================================
        UNIFIED TRAVELING WAVES ANALYSIS REPORT
        ========================================
        
        Analysis Window: {self.events['metadata']['window_start']} - {self.events['metadata']['window_end']} frames
        
        1. EVENT DETECTION SUMMARY
        --------------------------
        {self._generate_event_summary()}
        
        2. TEMPORAL ANALYSIS
        --------------------
        {self._generate_temporal_summary()}
        
        3. SPECTRAL ANALYSIS
        --------------------
        {self._generate_spectral_summary()}
        
        4. SPATIAL ANALYSIS
        -------------------
        {self._generate_spatial_summary()}
        
        5. KEY FINDINGS
        ---------------
        {self._generate_key_findings()}
        
        ========================================
        """
        
        return report
    
    def _generate_event_summary(self) -> str:
        """Generate text summary of detected events"""
        # ... (implementation details)
        return "Event summary details..."
    
    def _generate_temporal_summary(self) -> str:
        """Generate text summary of temporal analysis"""
        # ... (implementation details)
        return "Temporal analysis summary..."
    
    def _generate_spectral_summary(self) -> str:
        """Generate text summary of spectral analysis"""
        # ... (implementation details)
        return "Spectral analysis summary..."
    
    def _generate_spatial_summary(self) -> str:
        """Generate text summary of spatial analysis"""
        # ... (implementation details)
        return "Spatial analysis summary..."
    
    def save_results(self, output_dir: Union[str, Path]):
        """Save all results and cache to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cache
        self.cache.save(output_dir / 'computation_cache.pkl')
        
        # Save results
        import pickle
        with open(output_dir / 'analysis_results.pkl', 'wb') as f:
            pickle.dump(self.analysis_results, f)
        
        # Save figures
        import matplotlib.pyplot as plt
        for name, fig in self.analysis_results.get('figures', {}).items():
            fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight')
            fig.savefig(output_dir / f'{name}.svg', format='svg', bbox_inches='tight')
        
        # Save report
        with open(output_dir / 'analysis_report.txt', 'w') as f:
            f.write(self.analysis_results.get('report', ''))
        
        print(f"\nResults saved to: {output_dir}")


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_unified_analysis(lfp_processor, window_start: int, window_length: int,
                        config: Optional[WaveAnalysisConfig] = None,
                        optogenetic_intervals: Optional[List] = None,
                        output_dir: Optional[Union[str, Path]] = None) -> Dict:
    """
    Main entry point for running the unified analysis
    
    Example:
        config = WaveAnalysisConfig(
            pgd_threshold=1.4,
            energy_threshold=1.3,
            use_gpu=True
        )
        
        results = run_unified_analysis(
            lfp_processor,
            window_start=14000,
            window_length=44000,
            config=config,
            optogenetic_intervals=[(26.5, 27), (30, 40)],
            output_dir='results/unified_analysis'
        )
    """
    # Initialize unified analysis
    analyzer = UnifiedWaveAnalysis(lfp_processor, config)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(
        window_start, window_length,
        optogenetic_intervals,
        analyses_to_run=['all']
    )
    
    # Save results if output directory specified
    if output_dir:
        analyzer.save_results(output_dir)
    
    return results