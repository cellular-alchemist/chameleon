import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import time
import pickle
import networkx as nx
try:
    import cupy as cp
except ImportError:
    cp = None
try:
    import tqdm
except ImportError:
    tqdm = None
import sys
from typing import Tuple, Optional, Union, Dict, Any
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
import cmcrameri.cm as cmc

N = 256
romaO = cmc.romaO
colors = romaO(np.linspace(0, 1, N))

# 2) roll by half the cycle (180°)
shifted = np.roll(colors, N//2, axis=0)

# 3) build your rotated, continuous colormap
romaO_rotated = LinearSegmentedColormap.from_list('romaO_rotated_180', shifted, N=N)

class GradientConfig:
        MAX_NEIGHBOR_DIST = 50.0    # µm
        SPATIAL_SIGMA    = 45.0     # µm for Gaussian kernel
        RIDGE_LAMBDA     = 1e-5     # regularization strength
        MIN_GRADIENT     = 1e-5     # rad/µm
        
class RegularizedGradient:
    """
    Compute spatial phase gradients at electrode locations by
    solving a per‐point regularized least-squares problem.
    """
    def __init__(self, locations: np.ndarray, cfg: GradientConfig):
        self.locations = locations
        self.cfg       = cfg
        # build a KD‐tree on electrode positions
        self.tree = cKDTree(locations)
        # find up to 20 neighbors within MAX_NEIGHBOR_DIST
        dists, idxs = self.tree.query(
            locations,
            k=min(len(locations), 20),
            distance_upper_bound=cfg.MAX_NEIGHBOR_DIST
        )
        # trim out the infinite distances
        self.neighbor_indices = [
            idxs[i][dists[i] < np.inf]
            for i in range(len(locations))
        ]
        self.neighbor_dists = [
            dists[i][dists[i] < np.inf]
            for i in range(len(locations))
        ]

    def compute(self, phase_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        phase_data: shape (n_channels,)
        returns grad_x, grad_y each shape (n_channels,)
        """
        n = phase_data.shape[0]
        grad_x = np.zeros(n, dtype=np.float64)
        grad_y = np.zeros(n, dtype=np.float64)

        for i in range(n):
            nbrs = self.neighbor_indices[i]
            dists = self.neighbor_dists[i]

            # exclude self‐match
            mask = nbrs != i
            nbrs = nbrs[mask]
            dists = dists[mask]
            if len(nbrs) < 3:
                continue

            # relative positions
            dx = self.locations[nbrs, 0] - self.locations[i, 0]
            dy = self.locations[nbrs, 1] - self.locations[i, 1]

            # circular phase differences
            diffs = ((phase_data[nbrs] - phase_data[i] + np.pi) % (2*np.pi)) - np.pi

            # Gaussian spatial weighting
            w = np.exp(-0.5 * (dists / self.cfg.SPATIAL_SIGMA)**2)

            # design matrix
            A = np.stack([dx, dy], axis=1)            # shape (n_nbrs, 2)
            W = w.reshape(-1, 1)                      # shape (n_nbrs, 1)

            # regularized normal equations
            ATA = A.T @ (W * A) + self.cfg.RIDGE_LAMBDA * np.eye(2)
            ATb = A.T @ (w * diffs)

            try:
                g = np.linalg.solve(ATA, ATb)
                grad_x[i], grad_y[i] = g[0], g[1]
            except np.linalg.LinAlgError:
                # singular matrix, skip
                continue

        return grad_x, grad_y
 

class LFPDataProcessor:

    def __init__(self, waves, x_mod, y_mod, config_df, downsample_factor=10, use_cuda=False, gpu_memory_fraction=0.8):
        """
        Initialize the LFPDataProcessor with memory-optimized data structures.
        
        Parameters:
        -----------
        waves : dict
            Dictionary containing LFP and frequency band data
        x_mod, y_mod : array
            Modified electrode coordinates
        config_df : pandas.DataFrame
            Configuration DataFrame containing electrode information (pos_x, pos_y, channel, electrode)
        downsample_factor : int
            Factor by which to downsample the spatial resolution
        """
        
        #Initialize with GPU configuration
        self.use_cuda = use_cuda
        if self.use_cuda:
            try:
                import cupy as cp
                self.cuda_available = True
                self.xp = cp
                self.configure_gpu(gpu_memory_fraction)
                self.check_gpu_resources()
                print("CUDA acceleration enabled for LFP processing")
            except ImportError:
                self.cuda_available = False
                self.xp = np
                print("CUDA not available, falling back to CPU")
        else:
            self.cuda_available = False
            self.xp = np
        
        # Store basic parameters with efficient data types
        self.x_mod = np.asarray(x_mod, dtype=np.int16)
        self.y_mod = np.asarray(y_mod, dtype=np.int16)
        self.downsample_factor = downsample_factor
        self.fs = waves['fs']
        
        # Store electrode locations efficiently
        self.locations = np.asarray(waves['location'], dtype=np.int16)
        
        # Store wave data efficiently
        self._store_wave_data(waves)
        
        # Create DataFrame with complete electrode information
        self.data_df = pd.DataFrame({
            'x': self.locations[:, 0],
            'y': self.locations[:, 1],
            'x_mod': self.x_mod,
            'y_mod': self.y_mod,
            'channel': config_df['channel'].values,  # Add channel information
            'electrode': config_df['electrode'].values  # Add electrode IDs
        })
        
        # Verify the DataFrame has all required columns
        expected_columns = ['x', 'y', 'x_mod', 'y_mod', 'channel', 'electrode']
        missing_columns = [col for col in expected_columns if col not in self.data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data_df: {missing_columns}")
        
        # Clear input data to free memory
        del waves 
        
    def check_gpu_resources(self):
        """Check available GPU resources and current usage"""
        try:
            import cupy as cp
            device = cp.cuda.Device()
            
            print("\nGPU Resource Information:")
            print(f"Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            print(f"Total Memory: {device.mem_info[1] / 1024**3:.2f} GB")
            print(f"Free Memory: {device.mem_info[0] / 1024**3:.2f} GB")
            print(f"Current Memory Usage: {(device.mem_info[1] - device.mem_info[0]) / 1024**3:.2f} GB")
            
            return True
        except ImportError:
            print("CUDA not available")
            return False 
        
    def configure_gpu(self, memory_fraction=0.8):
        """Configure GPU memory usage"""
        try:
            import cupy as cp
            # Set up memory pool with the specified fraction of GPU memory
            total_memory = cp.cuda.Device().mem_info[1]
            pool_size = int(total_memory * memory_fraction)
            mempool = cp.cuda.MemoryPool(allocator=cp.cuda.memory.malloc_managed)
            cp.cuda.set_allocator(mempool.malloc)
            print(f"GPU memory pool size set to {pool_size / 1024**3:.2f} GB ({memory_fraction*100}% of total)")
            return True
        except ImportError:
            print("CUDA not available")
            return False
                    
    def _prepare_gpu_data(self, data):
        """Efficiently move data to GPU"""
        if self.cuda_available:
            try:
                return self.xp.asarray(data, dtype=self.xp.float32)  # Use float32 for better performance
            except Exception as e:
                print(f"Error moving data to GPU: {str(e)}")
                return data
        return data
        
    def _store_wave_data(self, waves):
        """Store wave data efficiently using float32"""
        self.waves = {}
        for key, data in waves.items():
            if key not in ['location', 'fs']:
                self.waves[key] = np.asarray(data, dtype=np.float32)
   

    def _get_analytical_data(self, band, start_frame=None, end_frame=None):
        """
        Compute analytical signal, phase, and amplitude for a specific band and time window.
        """
        if band not in self.waves:
            raise KeyError(f"Band {band} not found in wave data")
        
        # Get the data first
        data = self.waves[band]
        
        # Then do the error checking
        if start_frame is not None and end_frame is not None:
            if start_frame >= end_frame:
                raise ValueError("start_frame must be less than end_frame")
            if start_frame < 0 or end_frame > data.shape[1]:
                raise ValueError("Time window out of bounds")
            data = data[:, start_frame:end_frame]
        
        # Compute analytical signal
        analytical = signal.hilbert(data)
        
        return {
            'analytical': analytical,
            'phase': np.angle(analytical),
            'amplitude': np.abs(analytical)
        }
        
    
    def add_frequency_band(self, low_freq, high_freq, band_name=None, filter_order=4, 
                     filter_type='bandpass', store_analytical=False, fs=None, 
                     show_progress=True, use_gpu=None):
        """
        Add a new frequency band to the LFP processor by filtering the raw LFP data.
        
        Parameters:
        -----------
        low_freq : float
            Lower cutoff frequency in Hz
        high_freq : float
            Upper cutoff frequency in Hz
        band_name : str, optional
            Name for the new frequency band. If None, a name will be generated based on frequency range
        filter_order : int, optional
            Order of the filter (default: 4)
        filter_type : str, optional
            Type of filtering to perform: 'bandpass' or 'bandstop' (default: 'bandpass')
        store_analytical : bool, optional
            Whether to pre-compute and store analytical signal data (default: False)
        fs : float, optional
            Sampling rate to use for filtering (default: use the processor's fs value)
        show_progress : bool, optional
            Whether to display a progress bar during processing (default: True)
        use_gpu : bool, optional
            Whether to use GPU acceleration if available. If None, uses the processor's setting.
        
        Returns:
        --------
        str
            Name of the added frequency band
        """
        from scipy.signal import butter, filtfilt, hilbert
        import time
        import numpy as np
        import sys
        
        # Determine whether to use GPU
        if use_gpu is None:
            use_gpu = self.use_cuda if hasattr(self, 'use_cuda') else False
        
        # Set up appropriate numerical library (numpy or cupy)
        xp = np
        cp = None
        if use_gpu and hasattr(self, 'cuda_available') and self.cuda_available:
            try:
                import cupy as cp
                xp = cp
                print("Using GPU acceleration for filtering")
            except ImportError:
                print("GPU acceleration requested but cupy not available. Using CPU.")
        
        # Check if raw LFP data is available
        if 'lfp' not in self.waves:
            raise ValueError("Raw LFP data not found in waves dictionary. Cannot filter.")
        
        # Use processor's sampling rate if not specified
        if fs is None:
            fs = self.fs
        
        # Validate frequency parameters
        if not (0 < low_freq < high_freq < fs / 2):
            raise ValueError(f"Invalid frequency range. Ensure 0 < low_freq < high_freq < {fs/2}")
        
        # Validate filter type
        if filter_type not in ['bandpass', 'bandstop']:
            raise ValueError("filter_type must be either 'bandpass' or 'bandstop'")
        
        # Generate band name if not provided
        if band_name is None:
            if filter_type == 'bandpass':
                # Use standard names for common frequency bands
                if 0.5 <= low_freq < 4 and 4 <= high_freq <= 8:
                    band_name = "delta"
                elif 4 <= low_freq < 8 and 8 <= high_freq <= 13:
                    band_name = "theta"
                elif 8 <= low_freq < 13 and 13 <= high_freq <= 30:
                    band_name = "alpha"
                elif 13 <= low_freq < 30 and 30 <= high_freq <= 80:
                    band_name = "beta"
                elif 30 <= low_freq < 80 and 80 <= high_freq <= 150:
                    band_name = "gamma"
                elif 80 <= low_freq < 150 and 150 <= high_freq <= 300:
                    band_name = "high_gamma"
                else:
                    band_name = f"band_{int(low_freq)}_{int(high_freq)}"
            else:  # bandstop
                band_name = f"notch_{int(low_freq)}_{int(high_freq)}"
        
        # Check if band already exists
        if band_name in self.waves:
            # Generate a unique name by appending a suffix
            i = 1
            while f"{band_name}_{i}" in self.waves:
                i += 1
            band_name = f"{band_name}_{i}"
        
        print(f"Applying {filter_type} filter ({low_freq}-{high_freq} Hz) as '{band_name}'...")
        
        # Design the filter
        nyquist = fs / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = butter(filter_order, [low, high], btype=filter_type)
        
        # Apply zero-phase filtering to each channel
        raw_data = self.waves['lfp']
        n_channels = raw_data.shape[0]
        n_samples = raw_data.shape[1]
        
        # Initialize filtered data array
        filtered_data = np.zeros((n_channels, n_samples), dtype=np.float32)
        
        # Setup progress tracking
        start_time = time.time()
        try:
            import tqdm.auto as tqdm
            iterator = tqdm.tqdm(range(n_channels)) if show_progress else range(n_channels)
        except ImportError:
            # Fallback to simple progress reporting if tqdm is not available
            iterator = range(n_channels)
            if show_progress:
                print(f"Processing {n_channels} channels...")
        
        # Process each channel
        for i in iterator:
            # For GPU-accelerated version, we need to handle the case where filtfilt isn't available
            if use_gpu and cp is not None:
                # Transfer data to GPU
                channel_data = cp.asarray(raw_data[i])
                
                # For now, we need to move back to CPU for filtfilt
                cpu_data = cp.asnumpy(channel_data)
                filtered = filtfilt(b, a, cpu_data)
                
                # Store result
                filtered_data[i] = filtered
            else:
                # Pure CPU version
                filtered_data[i] = filtfilt(b, a, raw_data[i])
            
            # Show basic progress if tqdm is not available
            if show_progress and 'tqdm' not in sys.modules and (i+1) % max(1, n_channels//10) == 0:
                elapsed = time.time() - start_time
                progress = (i+1) / n_channels
                eta = elapsed / progress * (1 - progress) if progress > 0 else 0
                print(f"Progress: {progress*100:.1f}% - ETA: {eta:.1f}s")
        
        # Add filtered data to waves dictionary
        self.waves[band_name] = filtered_data
        
        elapsed = time.time() - start_time
        print(f"Filtering completed in {elapsed:.2f} seconds.")
        
        # Optionally compute and store analytical signal
        if store_analytical:
            print("Computing analytical signal...")
            analytical_key = f"{band_name}_analytical"
            phase_key = f"{band_name}_phase"
            amplitude_key = f"{band_name}_amplitude"
            
            # Initialize arrays
            analytical_data = np.zeros((n_channels, n_samples), dtype=np.complex64)
            phase_data = np.zeros((n_channels, n_samples), dtype=np.float32)
            amplitude_data = np.zeros((n_channels, n_samples), dtype=np.float32)
            
            # Process each channel with progress reporting
            start_time = time.time()
            try:
                iterator = tqdm.tqdm(range(n_channels)) if show_progress else range(n_channels)
            except:
                iterator = range(n_channels)
            
            for i in iterator:
                # Hilbert transform implementation
                analytical_data[i] = hilbert(filtered_data[i])
                phase_data[i] = np.angle(analytical_data[i])
                amplitude_data[i] = np.abs(analytical_data[i])
            
            # Store analytical data
            self.waves[analytical_key] = analytical_data
            self.waves[phase_key] = phase_data
            self.waves[amplitude_key] = amplitude_data
            
            elapsed = time.time() - start_time
            print(f"Analytical signal computation completed in {elapsed:.2f} seconds.")
            print(f"Pre-computed analytical signal data stored as: {analytical_key}, {phase_key}, {amplitude_key}")
        
        print(f"Successfully added new band '{band_name}' ({filter_type}, {low_freq}-{high_freq} Hz)")
        return band_name


    def create_matrix_sequence(
        self,
        data_type: str,
        initial_frame: int,
        final_frame: int,
        compute_gradients: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create sequence of downsampled matrices and (optionally) compute
        spatial gradients using RegularizedGradient for phase data.
        """
        # 1) dimensions
        num_rows = int(self.data_df['y_mod'].max() + 1)
        num_cols = int(self.data_df['x_mod'].max() + 1)
        ds_rows = num_rows // self.downsample_factor + 1
        ds_cols = num_cols // self.downsample_factor + 1
        num_frames = final_frame - initial_frame

        # 2) allocate
        filtered_matrices   = np.zeros((ds_rows, ds_cols, num_frames), dtype=np.float32)
        if compute_gradients:
            gradx_matrices = np.zeros_like(filtered_matrices)
            grady_matrices = np.zeros_like(filtered_matrices)
        else:
            gradx_matrices = grady_matrices = None

        # 3) set up regularized gradient (only once)
        cfg      = GradientConfig()
        grad_calc = RegularizedGradient(self.locations, cfg)

        # 4) chunked processing
        chunk_size = min(100, num_frames)
        for chunk_start in range(initial_frame, final_frame, chunk_size):
            chunk_end = min(chunk_start + chunk_size, final_frame)
            slice_out = slice(chunk_start - initial_frame, chunk_end - initial_frame)

            # get raw channel × time data
            if data_type in self.waves:
                chunk_data = self.waves[data_type][:, chunk_start:chunk_end]
            else:
                # derivative types
                band = data_type.split('_')[0]
                analytic = self._get_analytical_data(band, chunk_start, chunk_end)
                if data_type.endswith('_phase'):
                    chunk_data = analytic['phase']
                elif data_type.endswith('_amplitude'):
                    chunk_data = analytic['amplitude']
                else:
                    raise ValueError(f"Unknown data type: {data_type}")

            # fill the downsampled volume
            for chan_idx, (x_mod, y_mod) in enumerate(
                zip(self.data_df['x_mod'], self.data_df['y_mod'])
            ):
                x_ds = int(x_mod) // self.downsample_factor
                y_ds = int(y_mod) // self.downsample_factor
                filtered_matrices[y_ds, x_ds, slice_out] = chunk_data[chan_idx]

            # compute gradients if requested
            if compute_gradients:
                for t in range(chunk_end - chunk_start):
                    frame_idx = slice_out.start + t

                    if data_type.endswith('_phase'):
                        # use regularized electrode‐based phase gradient
                        phase_at_t = chunk_data[:, t]
                        gx_chan, gy_chan = grad_calc.compute(phase_at_t)
                        # map channel gradients back to grid
                        for i, (x_mod, y_mod) in enumerate(
                            zip(self.data_df['x_mod'], self.data_df['y_mod'])
                        ):
                            x_ds = int(x_mod) // self.downsample_factor
                            y_ds = int(y_mod) // self.downsample_factor
                            gradx_matrices[y_ds, x_ds, frame_idx] = gx_chan[i]
                            grady_matrices[y_ds, x_ds, frame_idx] = gy_chan[i]
                    else:
                        # leave the old finite-difference for non-phase data
                        mat = filtered_matrices[:, :, frame_idx]
                        gradx_matrices[:, :, frame_idx] = np.gradient(mat, axis=1)
                        grady_matrices[:, :, frame_idx] = np.gradient(mat, axis=0)

        return filtered_matrices, gradx_matrices, grady_matrices
    


    def _rescale_gradient(self, grad_x, grad_y, min_length=0.8, max_length=1.9, percentile=65, sigma=1):
        """
        Rescale gradient magnitudes with smoothing and percentile thresholding
        
        Parameters:
        -----------
        grad_x, grad_y : array
            Gradient components
        min_length : float
            Minimum length for rescaled vectors
        max_length : float
            Maximum length for rescaled vectors
        percentile : float
            Percentile threshold for zeroing out small magnitude vectors
        sigma : float
            Standard deviation for Gaussian smoothing
        """
        # Apply Gaussian smoothing to gradient components
        grad_x_smooth = gaussian_filter(grad_x, sigma=sigma)
        grad_y_smooth = gaussian_filter(grad_y, sigma=sigma)
        
        # Compute magnitude of smoothed gradients
        magnitude = np.sqrt(grad_x_smooth**2 + grad_y_smooth**2)
        
        # Compute threshold and create mask
        threshold = np.percentile(magnitude, percentile)
        mask = magnitude >= threshold
        
        # Initialize normalized gradients
        grad_x_norm = np.zeros_like(grad_x_smooth, dtype=np.float32)
        grad_y_norm = np.zeros_like(grad_y_smooth, dtype=np.float32)
        
        # Normalize vectors above threshold
        nonzero_mask = magnitude > 0
        grad_x_norm[nonzero_mask] = grad_x_smooth[nonzero_mask] / magnitude[nonzero_mask]
        grad_y_norm[nonzero_mask] = grad_y_smooth[nonzero_mask] / magnitude[nonzero_mask]
        
        # Zero out vectors below threshold
        grad_x_norm[~mask] = 0
        grad_y_norm[~mask] = 0
        
        # Rescale the magnitudes of remaining vectors
        non_zero_magnitudes = magnitude[mask]
        if non_zero_magnitudes.size > 0:
            mag_min = non_zero_magnitudes.min()
            mag_max = non_zero_magnitudes.max()
            if mag_max - mag_min != 0:
                scale = (max_length - min_length) / (mag_max - mag_min)
                scaled_magnitude = np.zeros_like(magnitude, dtype=np.float32)
                scaled_magnitude[mask] = (magnitude[mask] - mag_min) * scale + min_length
                return (grad_x_norm * scaled_magnitude).astype(np.float32), \
                    (grad_y_norm * scaled_magnitude).astype(np.float32)
        
        return grad_x_smooth, grad_y_smooth

    
    def create_csd_animation(self, data_type, initial_frame, final_frame, 
                        filename='csd_animation.mp4', conductivity=0.3,
                        fps=10, figsize=(15, 10)):
        """
        Create and save CSD animation.
        
        Parameters:
        -----------
        data_type : str
            Type of data to analyze (e.g., 'lfp', 'theta', etc.)
        initial_frame, final_frame : int
            Start and end frames for animation
        filename : str
            Output filename for animation
        conductivity : float
            Tissue conductivity in S/m (default 0.3 S/m for neural tissue)
        fps : int
            Frames per second for animation
        figsize : tuple
            Figure size (width, height) in inches
        """
        # Get matrices sequence using existing method
        filtered_matrices, _, _ = self.create_matrix_sequence(
            data_type, initial_frame, final_frame, compute_gradients=False
        )
        
        # Compute inter-electrode spacing from data_df
        y_spacing = np.median(np.diff(np.sort(np.unique(self.data_df['y']))))
        
        # Set up the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Initial LFP plot
        im1 = ax1.imshow(filtered_matrices[:, :, 0], 
                        cmap='viridis',
                        interpolation='bicubic')
        ax1.set_title('LFP')
        plt.colorbar(im1, ax=ax1)
        
        # Compute and plot initial CSD
        csd = self._compute_csd(filtered_matrices[:, :, 0], y_spacing, conductivity)
        im2 = ax2.imshow(csd, 
                        cmap='RdBu_r',  # Red-Blue with reversed colors
                        interpolation='bicubic')
        ax2.set_title('Current Source Density')
        plt.colorbar(im2, ax=ax2)
        
        def update(frame):
            # Update LFP plot
            im1.set_array(filtered_matrices[:, :, frame])
            
            # Update CSD plot
            csd = self._compute_csd(filtered_matrices[:, :, frame], y_spacing, conductivity)
            im2.set_array(csd)
            
            return [im1, im2]
        
        # Create animation
        anim = FuncAnimation(fig, update,
                            frames=final_frame-initial_frame,
                            interval=1000/fps,
                            blit=True)
        
        # Save animation
        writer = FFMpegWriter(fps=fps)
        anim.save(os.path.abspath(filename), writer=writer)
        plt.close()

    def _compute_csd(self, lfp_matrix, spacing, conductivity):
        """
        Compute CSD using the standard second derivative method.
        
        Parameters:
        -----------
        lfp_matrix : array-like
            2D array of LFP values
        spacing : float
            Inter-electrode spacing in micrometers
        conductivity : float
            Tissue conductivity in S/m
        
        Returns:
        --------
        array-like
            CSD estimation
        """
        # Convert spacing to meters
        spacing_m = spacing * 1e-6
        
        # Compute second spatial derivative
        csd = np.zeros_like(lfp_matrix)
        
        # Interior points (using central difference)
        csd[1:-1, :] = (lfp_matrix[:-2, :] - 2*lfp_matrix[1:-1, :] + lfp_matrix[2:, :]) / (spacing_m**2)
        
        # Scale by conductivity
        csd = -conductivity * csd
        
        # Apply spatial smoothing
        csd = gaussian_filter(csd, sigma=1)
        
        return csd
    
    def compute_and_plot_enstrophy(self, bands=['lfp', 'gamma'], window_start=0, window_length=1000, 
                                 sigma=1, apply_smoothing=True):
        """
        Compute and plot enstrophy over time for specified frequency bands with corrected processing order.
        
        Parameters:
        -----------
        bands : list
            List of frequency bands to analyze
        window_start : int
            Start frame of analysis window
        window_length : int
            Length of analysis window in frames
        sigma : float
            Standard deviation for Gaussian smoothing
        apply_smoothing : bool
            Whether to apply Gaussian smoothing to gradients
        
        Returns:
        --------
        tuple
            (results, means, stds)
            Raw enstrophy values, means, and standard deviations
        """
        window_end = window_start + window_length
        results = {}
        means = {}
        stds = {}
        
        # Get grid dimensions
        y_max = int(self.data_df['y_mod'].max())
        x_max = int(self.data_df['x_mod'].max())
        grid_shape = (y_max + 1, x_max + 1)
        
        # Set up plot
        plt.figure(figsize=(12, 6))
        colors = {'lfp': '#1f77b4', 'gamma': '#ff7f0e'}
        alpha_fill = 0.2
        
        # Pre-compute electrode positions
        electrode_positions = np.column_stack((
            self.data_df['y_mod'].astype(np.int16),
            self.data_df['x_mod'].astype(np.int16)
        ))
        
        # Process each band
        chunk_size = min(100, window_length)  # Process in chunks to save memory
        for band in bands:
            print(f"Processing {band} band...")
            all_enstrophies = []
            
            # Process in chunks
            for chunk_start in range(window_start, window_end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, window_end)
                
                # Get raw phase data for this chunk
                analytical_data = self._get_analytical_data(band, chunk_start, chunk_end)
                phase_data = analytical_data['phase']
                
                # Process each time point in the chunk
                for t in range(phase_data.shape[1]):
                    # Create phase matrix
                    phase_matrix = np.zeros(grid_shape, dtype=np.float32)
                    phase_matrix[electrode_positions[:, 0], electrode_positions[:, 1]] = phase_data[:, t]
                    
                    # Compute gradients on RAW phase data
                    gradient_x = np.zeros_like(phase_matrix, dtype=np.float32)
                    gradient_y = np.zeros_like(phase_matrix, dtype=np.float32)
                    
                    # Vectorized gradient computation
                    gradient_x[:, 1:-1] = ((phase_matrix[:, 2:] - phase_matrix[:, :-2] + np.pi) % 
                                       (2 * np.pi) - np.pi) / 2
                    gradient_y[1:-1, :] = ((phase_matrix[2:, :] - phase_matrix[:-2, :] + np.pi) % 
                                       (2 * np.pi) - np.pi) / 2
                    
                    # Handle edges
                    gradient_x[:, 0] = ((phase_matrix[:, 1] - phase_matrix[:, 0] + np.pi) % 
                                    (2 * np.pi) - np.pi)
                    gradient_x[:, -1] = ((phase_matrix[:, -1] - phase_matrix[:, -2] + np.pi) % 
                                    (2 * np.pi) - np.pi)
                    gradient_y[0, :] = ((phase_matrix[1, :] - phase_matrix[0, :] + np.pi) % 
                                    (2 * np.pi) - np.pi)
                    gradient_y[-1, :] = ((phase_matrix[-1, :] - phase_matrix[-2, :] + np.pi) % 
                                    (2 * np.pi) - np.pi)
                    
                    if apply_smoothing:
                        gradient_x = gaussian_filter(gradient_x, sigma=sigma)
                        gradient_y = gaussian_filter(gradient_y, sigma=sigma)
                    
                    # Compute curl components from raw gradients
                    dv_dx = np.gradient(gradient_y, axis=1)
                    du_dy = np.gradient(gradient_x, axis=0)
                    
                    # Compute vorticity and local enstrophy
                    omega = dv_dx - du_dy
                    local_enstrophy = 0.5 * omega**2
                    all_enstrophies.append(local_enstrophy.flatten())
                
                # Clear analytical data to free memory
                del analytical_data, phase_data
            
            # Convert to numpy array for computation
            all_enstrophies = np.array(all_enstrophies, dtype=np.float32)
            
            # Store raw results before any normalization
            results[band] = all_enstrophies
            
            # Compute statistics
            mean_enstrophy = np.mean(all_enstrophies, axis=1)
            std_enstrophy = np.std(all_enstrophies, axis=1)
            
            # Normalize for visualization only
            global_max = np.max(mean_enstrophy)
            if global_max > 0:
                mean_normalized = mean_enstrophy / global_max
                std_normalized = std_enstrophy / global_max
            else:
                mean_normalized = mean_enstrophy
                std_normalized = std_enstrophy
            
            # Store normalized values
            means[band] = mean_normalized
            stds[band] = std_normalized
            
            # Plot
            time_points = np.arange(window_start, window_end) / self.fs
            plt.plot(time_points, mean_normalized, 
                    label=f'{band.upper()} Enstrophy', 
                    color=colors[band],
                    linewidth=1)
            plt.fill_between(time_points, 
                           mean_normalized - std_normalized,
                           mean_normalized + std_normalized,
                           color=colors[band], alpha=alpha_fill)
            
            # Clear arrays to free memory
            del all_enstrophies
        
        if not results:
            raise ValueError("No data could be processed for any of the requested bands")
        
        plt.title('Normalized Enstrophy Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Enstrophy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add information box
        info_text = (
            f'Window: {window_start/self.fs:.2f}s - {window_end/self.fs:.2f}s\n'
            f'Spatial smoothing σ: {sigma}\n'
            f'Grid size: {grid_shape[0]}×{grid_shape[1]}'
        )
        plt.text(1.02, 0.98, info_text, transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8), fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return results, means, stds

    def compute_and_plot_enstrophy(self, bands=['lfp', 'gamma'], window_start=0, window_length=1000, 
                                 sigma=1, apply_smoothing=True):
        """
        Compute and plot enstrophy over time with consistent spatial downsampling.
        Uses the same downsampling approach as create_matrix_sequence for consistency.
        
        Parameters:
        -----------
        bands : list
            List of frequency bands to analyze
        window_start : int
            Start frame of analysis window
        window_length : int
            Length of analysis window in frames
        sigma : float
            Standard deviation for Gaussian smoothing
        apply_smoothing : bool
            Whether to apply Gaussian smoothing to gradients
        
        Returns:
        --------
        dict
            Dictionary containing the raw enstrophy time series for each band
        """
        window_end = window_start + window_length
        results = {}
        
        # Get downsampled dimensions (consistent with create_matrix_sequence)
        num_rows = int(self.data_df['y_mod'].max() + 1)
        num_cols = int(self.data_df['x_mod'].max() + 1)
        ds_rows = num_rows // self.downsample_factor + 1
        ds_cols = num_cols // self.downsample_factor + 1
        grid_shape = (ds_rows, ds_cols)
        
        # Set up plot
        plt.figure(figsize=(12, 6))
        colors = {'lfp': '#1f77b4', 'gamma': 'magenta'}
        
        # Pre-compute downsampled electrode positions
        ds_positions = np.column_stack((
            (self.data_df['y_mod'] // self.downsample_factor).astype(np.int16),
            (self.data_df['x_mod'] // self.downsample_factor).astype(np.int16)
        ))
        
        # Process each band
        for band in bands:
            print(f"Processing {band} band...")
            enstrophy_timeseries = []
            
            # Process in chunks to save memory
            chunk_size = min(100, window_length)
            for chunk_start in range(window_start, window_end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, window_end)
                
                # Get raw phase data for this chunk
                analytical_data = self._get_analytical_data(band, chunk_start, chunk_end)
                phase_data = analytical_data['phase']
                
                # Process each time point in the chunk
                for t in range(phase_data.shape[1]):
                    # Create phase matrix with downsampled dimensions
                    phase_matrix = np.zeros(grid_shape, dtype=np.float32)
                    phase_matrix[ds_positions[:, 0], ds_positions[:, 1]] = phase_data[:, t]
                    
                    # Compute gradients on RAW phase data
                    gradient_x = np.zeros_like(phase_matrix, dtype=np.float32)
                    gradient_y = np.zeros_like(phase_matrix, dtype=np.float32)
                    
                    # Vectorized gradient computation
                    gradient_x[:, 1:-1] = ((phase_matrix[:, 2:] - phase_matrix[:, :-2] + np.pi) % 
                                       (2 * np.pi) - np.pi) / 2
                    gradient_y[1:-1, :] = ((phase_matrix[2:, :] - phase_matrix[:-2, :] + np.pi) % 
                                       (2 * np.pi) - np.pi) / 2
                    
                    # Handle edges
                    gradient_x[:, 0] = ((phase_matrix[:, 1] - phase_matrix[:, 0] + np.pi) % 
                                    (2 * np.pi) - np.pi)
                    gradient_x[:, -1] = ((phase_matrix[:, -1] - phase_matrix[:, -2] + np.pi) % 
                                    (2 * np.pi) - np.pi)
                    gradient_y[0, :] = ((phase_matrix[1, :] - phase_matrix[0, :] + np.pi) % 
                                    (2 * np.pi) - np.pi)
                    gradient_y[-1, :] = ((phase_matrix[-1, :] - phase_matrix[-2, :] + np.pi) % 
                                    (2 * np.pi) - np.pi)
                    
                    if apply_smoothing:
                        gradient_x = gaussian_filter(gradient_x, sigma=sigma)
                        gradient_y = gaussian_filter(gradient_y, sigma=sigma)
                    
                    # Compute curl components from raw gradients
                    dv_dx = np.gradient(gradient_y, axis=1)
                    du_dy = np.gradient(gradient_x, axis=0)
                    
                    # Compute vorticity
                    omega = dv_dx - du_dy
                    
                    # Compute true enstrophy as integral of squared vorticity
                    enstrophy = 0.5 * np.sum(omega**2)  # E = 1/2 ∫∫ ω² dA
                    enstrophy_timeseries.append(enstrophy)
                
                # Clear analytical data to free memory
                del analytical_data, phase_data
            
            # Convert to numpy array
            enstrophy_timeseries = np.array(enstrophy_timeseries, dtype=np.float32)
            
            # Store raw results
            results[band] = enstrophy_timeseries
            
            # Normalize for visualization
            if np.max(enstrophy_timeseries) > np.min(enstrophy_timeseries):
                normalized_enstrophy = (enstrophy_timeseries - np.min(enstrophy_timeseries)) / \
                                     (np.max(enstrophy_timeseries) - np.min(enstrophy_timeseries))
            else:
                normalized_enstrophy = np.zeros_like(enstrophy_timeseries)
            
            # Plot normalized values
            time_points = np.arange(window_start, window_end) / self.fs
            plt.plot(time_points, normalized_enstrophy, 
                    label=f'{band.upper()} Enstrophy', 
                    color=colors[band],
                    linewidth=1)
            
            # Clear arrays to free memory
            del enstrophy_timeseries, normalized_enstrophy
        
        if not results:
            raise ValueError("No data could be processed for any of the requested bands")
        
        plt.title('Normalized Enstrophy Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Enstrophy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-0.1, 1.1)  # Set y-axis limits for normalized values
        
        # Add information box
        info_text = (
            f'Window: {window_start/self.fs:.2f}s - {window_end/self.fs:.2f}s\n'
            f'Spatial smoothing σ: {sigma}\n'
            f'Grid size: {grid_shape[0]}×{grid_shape[1]}'
        )
        plt.text(1.02, 0.98, info_text, transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8), fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    
    def _compute_winding_number(self, grad_x, grad_y, x, y):
        """
        Compute winding number using the 8 neighboring points in a square loop.
        
        Parameters:
        -----------
        grad_x, grad_y : 2D arrays
            Gradient components
        x, y : int
            Center coordinates of the point to analyze
            
        Returns:
        --------
        float
            Winding number (should be close to integer for defects)
        """
        # Define the square path coordinates (clockwise)
        path_x = [x-1, x-1, x-1, x,   x+1, x+1, x+1, x  ]
        path_y = [y-1, y,   y+1, y+1, y+1, y,   y-1, y-1]
        
        # Get gradient values along the path
        grad_x_path = grad_x[path_y, path_x]
        grad_y_path = grad_y[path_y, path_x]
        
        # Compute the path segments
        dx = np.diff(path_x, append=path_x[0]) # differences including wrap-around
        dy = np.diff(path_y, append=path_y[0])
        
        # Compute line integral along square path
        # grad . dl = grad_x*dx + grad_y*dy
        integrand = grad_x_path * dx + grad_y_path * dy
        
        # Sum up the contributions and normalize
        winding = np.sum(integrand) / (2 * np.pi)
        
        return winding



    def _compute_curl(self, gradient_x, gradient_y):
        """Compute curl field of the phase gradient."""
        if gradient_x is None or gradient_y is None:
            raise ValueError("Gradients must be provided to compute curl")
            
        if gradient_x.shape != gradient_y.shape:
            raise ValueError(f"Gradient shapes must match. Got {gradient_x.shape} and {gradient_y.shape}")
            
        print(f"Computing curl for gradients of shape {gradient_x.shape}")  # Debug print
        
        try:
            dvy_dx = np.gradient(gradient_y, axis=1) 
            dvx_dy = np.gradient(gradient_x, axis=0)
            return dvy_dx - dvx_dy
        except Exception as e:
            print(f"Error in curl computation: {str(e)}")
            print(f"Gradient shapes: {gradient_x.shape}, {gradient_y.shape}")
            raise

    def _get_ring_mask(self, center, radius, shape, width=1.0):
        """Get mask for ring at specified radius from center."""
        y, x = np.ogrid[-center[0]:shape[0]-center[0], -center[1]:shape[1]-center[1]]
        distances = np.sqrt(x*x + y*y)
        return (distances >= radius - width/2) & (distances < radius + width/2)

    def _compute_angle_differences(self, gradient_x, gradient_y, center, mask):
        """Compute angle differences between gradient vectors and ideal spiral pattern."""
        y, x = np.ogrid[-center[0]:gradient_x.shape[0]-center[0], 
                        -center[1]:gradient_x.shape[1]-center[1]]
        # Compute radial vectors
        r_norm = np.sqrt(x*x + y*y)
        r_norm[r_norm == 0] = 1
        rx = x / r_norm
        ry = y / r_norm
        
        # Compute tangential vectors for ideal spiral
        tangent_x = -ry
        tangent_y = rx
        
        # Normalize gradient vectors
        grad_norm = np.sqrt(gradient_x**2 + gradient_y**2)
        grad_norm[grad_norm == 0] = 1
        grad_x_norm = gradient_x / grad_norm
        grad_y_norm = gradient_y / grad_norm
        
        # Compute angle differences
        dot_product = grad_x_norm * tangent_x + grad_y_norm * tangent_y
        angles = np.arccos(np.clip(dot_product, -1, 1))
        return angles * mask

    def _count_excluded_points(self, gradient_x, gradient_y, center, mask, 
                            angle_threshold=np.pi/4):
        """Count points in mask that deviate from ideal spiral pattern."""
        angles = self._compute_angle_differences(gradient_x, gradient_y, center, mask)
        return np.sum((angles > angle_threshold) & mask)

    def _compute_spiral_extent(self, phase_data, gradient_x, gradient_y, center, 
                            polarity, max_radius=100, expansion_threshold=1.0):
        """Compute adaptive spiral size using expanding rings method."""
        spiral_mask = np.zeros_like(phase_data, dtype=bool)
        
        # Include core 3x3 region
        y, x = center
        y_core = slice(max(0, y-1), min(phase_data.shape[0], y+2))
        x_core = slice(max(0, x-1), min(phase_data.shape[1], x+2))
        spiral_mask[y_core, x_core] = True
        
        # Expand radius
        for radius in np.arange(2, max_radius, 0.5):
            ring_mask = self._get_ring_mask(center, radius, phase_data.shape)
            excluded = self._count_excluded_points(gradient_x, gradient_y, center, ring_mask)
            
            if excluded > radius * expansion_threshold:
                break
                
            spiral_mask |= ring_mask
        
        return spiral_mask

    def _process_defect_polarity(self, mask, curlz, phase_data, gradient_x, gradient_y,
                            polarity, min_cluster_size=9):
        """Process defects of a single polarity."""
        from scipy import ndimage
        
        defects = []
        labeled, num_features = ndimage.label(mask)
        
        for i in range(1, num_features + 1):
            component = labeled == i
            if np.sum(component) < min_cluster_size:
                continue
            
            # Get center position from maximum curl
            local_curl = curlz * component
            max_idx = np.argmax(np.abs(local_curl))
            y, x = np.unravel_index(max_idx, local_curl.shape)
            
            # Determine spiral size adaptively
            spiral_mask = self._compute_spiral_extent(
                phase_data, gradient_x, gradient_y, (y,x), polarity)
            
            defects.append({
                'position': (y,x),
                'polarity': polarity,
                'strength': local_curl[y,x],
                'mask': spiral_mask,
                'size': np.sum(spiral_mask)
            })
        
        return defects

    def _generate_surrogate_phases(self, phase_data, n_surrogates=100):
        """Generate phase-randomized surrogate data."""
        surrogates = []
        for _ in range(n_surrogates):
            # FFT phase randomization
            fft = np.fft.fft2(phase_data)
            magnitudes = np.abs(fft)
            random_phases = np.random.uniform(0, 2*np.pi, size=fft.shape)
            randomized_fft = magnitudes * np.exp(1j * random_phases)
            surrogate = np.real(np.fft.ifft2(randomized_fft))
            surrogates.append(surrogate)
        return np.array(surrogates)

    def _compute_defect_statistics(self, defects):
        """Compute statistical properties of defects for comparison."""
        if not defects:
            return {'count': 0, 'mean_strength': 0, 'mean_size': 0}
        
        strengths = [d['strength'] for d in defects]
        sizes = [d['size'] for d in defects]
        
        return {
            'count': len(defects),
            'mean_strength': np.mean(np.abs(strengths)),
            'mean_size': np.mean(sizes)
        }

    def _is_significant(self, defect, surrogate_stats, percentile=95):
        """Test if defect is significant compared to surrogate statistics."""
        strength_thresh = np.percentile([s['mean_strength'] for s in surrogate_stats], percentile)
        size_thresh = np.percentile([s['mean_size'] for s in surrogate_stats], percentile)
        
        return (abs(defect['strength']) > strength_thresh and 
                defect['size'] > size_thresh)
        
    def detect_topological_defects(self, phase_matrix, gradient_x, gradient_y,
                             positive_threshold=1.0, negative_threshold=-1.0,
                             min_cluster_size=9):
        """
        Detect topological defects using pre-processed matrices and gradients.
        
        Parameters:
        -----------
        phase_matrix : numpy.ndarray
            2D array of phase values from filtered_matrices
        gradient_x, gradient_y : numpy.ndarray
            2D arrays of scaled gradients from _rescale_gradient
        positive_threshold : float
            Threshold for detecting positive (counterclockwise) defects
        negative_threshold : float
            Threshold for detecting negative (clockwise) defects
        min_cluster_size : int
            Minimum size of defect clusters in pixels
        """
        # Compute curl from scaled gradients
        curlz = self._compute_curl(gradient_x, gradient_y)
        
        # Detect positive defects
        pos_mask = curlz > positive_threshold
        positive_defects = self._process_defect_polarity(
            pos_mask, curlz, phase_matrix, gradient_x, gradient_y, +1, min_cluster_size)
        
        # Detect negative defects
        neg_mask = curlz < negative_threshold
        negative_defects = self._process_defect_polarity(
            neg_mask, curlz, phase_matrix, gradient_x, gradient_y, -1, min_cluster_size)
        
        return positive_defects, negative_defects
    def _interpolate_defect(self, defect1, defect2, frame):
        """
        Interpolate defect properties between two detected defects.
        
        Parameters:
        -----------
        defect1 : dict
            First defect
        defect2 : dict
            Second defect
        frame : int
            Frame number for interpolated defect
            
        Returns:
        --------
        dict
            Interpolated defect with position, strength, and mask
        """
        # Interpolate position linearly
        pos1 = np.array(defect1['position'])
        pos2 = np.array(defect2['position'])
        pos_interp = (pos1 + pos2) / 2
        
        # Interpolate strength
        strength_interp = (defect1['strength'] + defect2['strength']) / 2
        
        # Create interpolated defect dict
        interpolated = {
            'position': tuple(pos_interp.astype(int)),
            'strength': strength_interp,
            'polarity': defect1['polarity'],  # polarity should be same for both
            'frame': frame,
            'interpolated': True  # flag to mark as interpolated
        }
        
        # Interpolate mask if available
        if 'mask' in defect1 and 'mask' in defect2:
            # Use morphological operations to interpolate mask
            from scipy import ndimage
            mask1 = defect1['mask']
            mask2 = defect2['mask']
            
            # Dilate both masks slightly
            mask1_dilated = ndimage.binary_dilation(mask1)
            mask2_dilated = ndimage.binary_dilation(mask2)
            
            # Interpolated mask is intersection of dilated masks
            interpolated['mask'] = mask1_dilated & mask2_dilated
        
        return interpolated

    def detect_topological_defects_sequence(self, phase_matrices, gradient_x, gradient_y,
                                      positive_threshold=1.0, negative_threshold=-1.0,
                                      min_cluster_size=9, max_tracking_distance=5):
        """
        Detect and track defects across a sequence of frames.
        
        Parameters:
        -----------
        phase_matrices : array, shape (n_rows, n_cols, n_frames)
            Sequence of phase matrices
        gradient_x, gradient_y : array, shape (n_rows, n_cols, n_frames)
            Sequence of gradient fields
        max_tracking_distance : float
            Maximum distance for connecting defects between frames
            
        Returns:
        --------
        dict : Contains
            'trajectories': List of tracked defect paths
            'footprints': Dict mapping frame indices to defect masks
            'defect_properties': Additional defect characteristics over time
        """
        n_frames = phase_matrices.shape[2]
        raw_defects = []
        
        # First pass: detect defects in each frame
        for t in range(n_frames):
            pos_defects, neg_defects = self.detect_topological_defects(
                phase_matrices[:,:,t],
                gradient_x[:,:,t],
                gradient_y[:,:,t],
                positive_threshold,
                negative_threshold,
                min_cluster_size
            )
            
            # Add frame information
            for d in pos_defects + neg_defects:
                d['frame'] = t
            raw_defects.append({'positive': pos_defects, 'negative': neg_defects})
        
        # Second pass: track defects across frames
        trajectories = self._track_defects(raw_defects, max_tracking_distance)
        
        # Third pass: validate and extend trajectories
        validated_trajectories = self._validate_trajectories(
            trajectories, min_duration=3)
        extended_trajectories = self._extend_trajectories(
            validated_trajectories, phase_matrices.shape)
            
        # Generate footprints (spatial masks over time)
        footprints = self._generate_trajectory_footprints(
            extended_trajectories, phase_matrices.shape)
        
        return {
            'trajectories': extended_trajectories,
            'footprints': footprints,
            'raw_detections': raw_defects
        }
    def _track_defects(self, raw_defects, max_distance):
        """Track defects across frames based on proximity."""
        trajectories = []
        active_tracks = []
        
        for frame, frame_defects in enumerate(raw_defects):
            current_defects = frame_defects['positive'] + frame_defects['negative']
            
            # Match current defects to active tracks
            matched_indices = set()
            for track in active_tracks:
                last_defect = track[-1]
                best_match = None
                min_dist = max_distance
                
                # Find closest unmatched defect of same polarity
                for i, defect in enumerate(current_defects):
                    if i not in matched_indices and defect['polarity'] == last_defect['polarity']:
                        dist = np.sqrt((defect['position'][0] - last_defect['position'][0])**2 +
                                    (defect['position'][1] - last_defect['position'][1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_match = i
                            
                if best_match is not None:
                    track.append(current_defects[best_match])
                    matched_indices.add(best_match)
                else:
                    # Track ended
                    if len(track) > 1:
                        trajectories.append(track)
                    active_tracks.remove(track)
                    
            # Start new tracks for unmatched defects
            for i, defect in enumerate(current_defects):
                if i not in matched_indices:
                    active_tracks.append([defect])
        
        # Add remaining active tracks
        trajectories.extend(active_tracks)
        return trajectories

    def _validate_trajectories(self, trajectories, min_duration):
        """Validate trajectories based on duration and consistency."""
        validated = []
        for traj in trajectories:
            if len(traj) >= min_duration:
                # Additional validation criteria could go here
                validated.append(traj)
        return validated

    def _extend_trajectories(self, trajectories, shape):
        """Fill small gaps in trajectories."""
        extended = []
        for traj in trajectories:
            frames = [d['frame'] for d in traj]
            for i in range(len(frames)-1):
                if frames[i+1] - frames[i] == 2:  # One frame gap
                    # Interpolate defect properties
                    mid_defect = self._interpolate_defect(
                        traj[i], traj[i+1], frames[i] + 1)
                    traj.insert(i+1, mid_defect)
            extended.append(traj)
        return extended

    def _generate_trajectory_footprints(self, trajectories, shape):
        """Generate spatial masks for each trajectory over time."""
        footprints = np.zeros(shape, dtype=bool)
        
        for traj in trajectories:
            for defect in traj:
                frame = defect['frame']
                if 'mask' in defect:
                    footprints[:,:,frame] |= defect['mask']
        
        return footprints
    def analyze_phase_patterns(self, data_type='gamma', initial_frame=0, final_frame=1000,
                         positive_threshold=1.0, negative_threshold=-1.0, 
                         min_cluster_size=9, max_tracking_distance=5):
        """
        Main analysis pipeline for detecting and tracking phase patterns over time.
        
        Parameters:
        -----------
        data_type : str
            Type of oscillation to analyze ('gamma', 'theta', etc.)
        initial_frame, final_frame : int
            Time window for analysis
        positive_threshold : float
            Threshold for detecting counterclockwise defects
        negative_threshold : float
            Threshold for detecting clockwise defects
        min_cluster_size : int
            Minimum size of defect clusters
        max_tracking_distance : float
            Maximum distance for connecting defects between frames
        
        Returns:
        --------
        dict
            Complete analysis results including trajectories, footprints, 
            and defect properties over time
        """
        # Get matrices and gradients for full sequence
        filtered_matrices, grad_x, grad_y = self.create_matrix_sequence(
            f'{data_type}_phase', 
            initial_frame, 
            final_frame, 
            compute_gradients=True
        )
        
        # Process gradients with smoothing for full sequence
        scaled_gradients = np.zeros_like(grad_x), np.zeros_like(grad_y)
        for t in range(filtered_matrices.shape[2]):
            scaled_gradients[0][..., t], scaled_gradients[1][..., t] = self._rescale_gradient(
                grad_x[..., t], 
                grad_y[..., t],
                min_length=0.8,
                max_length=1.9,
                percentile=65,
                sigma=1
            )
        
        # Detect and track defects across all frames
        defect_info = self.detect_topological_defects_sequence(
            filtered_matrices,
            scaled_gradients[0],
            scaled_gradients[1],
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            min_cluster_size=min_cluster_size,
            max_tracking_distance=max_tracking_distance
        )
        
        # Store results as class attributes
        self.filtered_matrices = filtered_matrices
        self.scaled_gradients = scaled_gradients
        self.defect_info = defect_info
        
        # Add analysis metadata
        defect_info['metadata'] = {
            'data_type': data_type,
            'initial_frame': initial_frame,
            'final_frame': final_frame,
            'thresholds': (positive_threshold, negative_threshold),
            'min_cluster_size': min_cluster_size,
            'max_tracking_distance': max_tracking_distance
        }
        
        # Compute summary statistics
        defect_info['statistics'] = {
            'n_positive_trajectories': len([t for t in defect_info['trajectories'] 
                                        if t[0]['polarity'] > 0]),
            'n_negative_trajectories': len([t for t in defect_info['trajectories'] 
                                        if t[0]['polarity'] < 0]),
            'mean_trajectory_duration': np.mean([len(t) for t in defect_info['trajectories']]),
            'total_frames': final_frame - initial_frame
        }
        
        return defect_info    
    
    def create_defect_animation(self, initial_frame, final_frame, defect_info,
                          filename='defect_animation.mp4', fps=10):
        """
        Create enhanced animation showing defect trajectories and evolution.
        
        Parameters:
        -----------
        defect_info : dict
            Output from detect_topological_defects_sequence containing
            trajectories, footprints, and properties
        """
        filtered_matrices, grad_x, grad_y = self.create_matrix_sequence(
            'gamma_phase', initial_frame, final_frame, compute_gradients=True)
        
        trajectories = defect_info['trajectories']
        footprints = defect_info['footprints']
        
        # Assign unique colors to trajectories
        pos_trajectories = [t for t in trajectories if t[0]['polarity'] > 0]
        neg_trajectories = [t for t in trajectories if t[0]['polarity'] < 0]
        
        colors_pos = plt.cm.Reds(np.linspace(0.5, 1, len(pos_trajectories)))
        colors_neg = plt.cm.Blues(np.linspace(0.5, 1, len(neg_trajectories)))
        Y, X = np.meshgrid(
            np.arange(filtered_matrices.shape[0]),
            np.arange(filtered_matrices.shape[1]),
            indexing='ij'
        )
        fig, ax = plt.subplots(figsize=(15, 10))
        
        def update(frame):
            ax.clear()
            
            # Plot phase field
            im = ax.imshow(filtered_matrices[:,:,frame], cmap='twilight')
            
            # Plot vector field
            scaled_grad_x, scaled_grad_y = self._rescale_gradient(
                grad_x[:,:,frame], grad_y[:,:,frame])
            quiv = ax.quiver(X, Y, scaled_grad_x, scaled_grad_y,
                            color='gainsboro', scale=98)
            
            # Plot active trajectories
            for i, traj in enumerate(pos_trajectories):
                # Plot full trajectory path (faded)
                frames = [d['frame'] for d in traj]
                positions = np.array([d['position'] for d in traj])
                ax.plot(positions[:,1], positions[:,0], '-', 
                    color=colors_pos[i], alpha=0.3)
                
                # Plot current position (bright)
                if frame in frames:
                    idx = frames.index(frame)
                    defect = traj[idx]
                    y, x = defect['position']
                    ax.plot(x, y, '+', color=colors_pos[i], 
                        markersize=10, label=f'CCW Track {i+1}')
                    
                    # Plot footprint
                    if 'mask' in defect:
                        ax.contour(defect['mask'], colors=[colors_pos[i]], 
                                alpha=0.5)
            
            # Repeat for negative trajectories
            for i, traj in enumerate(neg_trajectories):
                frames = [d['frame'] for d in traj]
                positions = np.array([d['position'] for d in traj])
                ax.plot(positions[:,1], positions[:,0], '-', 
                    color=colors_neg[i], alpha=0.3)
                
                if frame in frames:
                    idx = frames.index(frame)
                    defect = traj[idx]
                    y, x = defect['position']
                    ax.plot(x, y, '+', color=colors_neg[i], 
                        markersize=10, label=f'CW Track {i+1}')
                    if 'mask' in defect:
                        ax.contour(defect['mask'], colors=[colors_neg[i]], 
                                alpha=0.5)
            
            # Add legend and title
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            ax.set_title(f'Frame {frame}/{filtered_matrices.shape[2]-1}')
            
            return [im, quiv]
    
   

        anim = FuncAnimation(
            fig, 
            update,
            frames=filtered_matrices.shape[2],
            interval=1000/fps,
            blit=True
        )

        writer = FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer)
        plt.close()
    '''
    def create_defect_animation(self, initial_frame, final_frame, threshold=0.5, 
                          filename='defect_animation.mp4', fps=10):
        """
        Create animation of phase field with detected topological defects.
        """
        # Get matrices sequence
        filtered_matrices, grad_x, grad_y = self.create_matrix_sequence(
            'lfp_phase', 
            initial_frame, 
            final_frame, 
            compute_gradients=True
        )
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Create grid for quiver plot
        Y, X = np.meshgrid(
            np.arange(filtered_matrices.shape[0]),
            np.arange(filtered_matrices.shape[1]),
            indexing='ij'
        )
        
        # Initial plot to set up the artists
        im = ax.imshow(filtered_matrices[:, :, 0], 
                    cmap='twilight',
                    interpolation='bicubic',
                    vmin=-np.pi,
                    vmax=np.pi)
        
        quiv = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y),
                        color='gainsboro',
                        scale=98,
                        headwidth=2,
                        headlength=2,
                        headaxislength=2)
        
        # Initialize scatter plots for defects
        scatter_pos = ax.scatter([], [], c='red', marker='o', s=100, label='+1')
        scatter_neg = ax.scatter([], [], c='blue', marker='o', s=100, label='-1')
        
        title = ax.set_title('Phase Field and Topological Defects')
        ax.legend()
        
        def update(frame):
            # Get gradients for current frame
            scaled_grad_x, scaled_grad_y = self._rescale_gradient(
                grad_x[:, :, frame], 
                grad_y[:, :, frame],
                min_length=0.8,
                max_length=1.9,
                percentile=65,
                sigma=1
            )
            
            # Update phase field
            im.set_array(filtered_matrices[:, :, frame])
            
            # Update gradient field
            quiv.set_UVC(scaled_grad_x, scaled_grad_y)
            
            # Detect defects
            defects = self.detect_topological_defects(
                filtered_matrices[:, :, frame],
                scaled_grad_x,
                scaled_grad_y,
                threshold=threshold
            )
            
            # Update defect positions
            pos_defects = [(x, y) for x, y, c in 
                        zip(defects['x'], defects['y'], defects['charge']) 
                        if c > 0]
            neg_defects = [(x, y) for x, y, c in 
                        zip(defects['x'], defects['y'], defects['charge']) 
                        if c < 0]
            
            if pos_defects:
                pos_x, pos_y = zip(*pos_defects)
                scatter_pos.set_offsets(np.c_[pos_x, pos_y])
            else:
                scatter_pos.set_offsets(np.c_[[], []])
                
            if neg_defects:
                neg_x, neg_y = zip(*neg_defects)
                scatter_neg.set_offsets(np.c_[neg_x, neg_y])
            else:
                scatter_neg.set_offsets(np.c_[[], []])
            
            title.set_text(f'Frame {frame}/{final_frame-initial_frame-1}')
            
            # Return all artists that were updated
            return [im, quiv, scatter_pos, scatter_neg, title]
        
        # Create animation
        anim = FuncAnimation(
            fig, 
            update,
            frames=final_frame-initial_frame,
            interval=1000/fps,
            blit=True
        )
        
        # Save animation
        writer = FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer)
        plt.close()
    '''
    def create_density_animation(self, initial_frame, final_frame, threshold=0.5, 
                           filename='density_animation.mp4', fps=10):
        """
        Create animation of defect density evolution.
        
        Parameters:
        -----------
        initial_frame, final_frame : int
            Start and end frames for animation
        threshold : float
            Threshold for defect detection
        filename : str
            Output filename for animation
        fps : int
            Frames per second for animation
        """
        # Get matrices sequence
        filtered_matrices, grad_x, grad_y = self.create_matrix_sequence(
            'gamma_phase', 
            initial_frame, 
            final_frame, 
            compute_gradients=True
        )
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Initial plot
        density_plot = ax.imshow(
            np.zeros(filtered_matrices.shape[:2]),
            cmap='RdBu',
            interpolation='gaussian',
            vmin=-0.08,  # Adjust these limits based on your data
            vmax=0.08
        )
        
        plt.colorbar(density_plot, label='Defect Density')
        title = ax.set_title('Defect Density')
        
        def update(frame):
            # Get gradients for current frame
            scaled_grad_x, scaled_grad_y = self._rescale_gradient(
                grad_x[:, :, frame], 
                grad_y[:, :, frame],
                min_length=0.8,
                max_length=1.9,
                percentile=65,
                sigma=1
            )
            
            # Detect defects
            defects = self.detect_topological_defects(
                filtered_matrices[:, :, frame],
                scaled_grad_x,
                scaled_grad_y,
                threshold=threshold
            )
            
            # Compute density
            density = self.compute_defect_density(defects, filtered_matrices.shape[:2], sigma=2.0)
            
            # Update plot
            density_plot.set_array(density)
            title.set_text(f'Defect Density (Frame {frame}/{final_frame-initial_frame-1})')
            
            return [density_plot, title]
        
        # Create animation
        anim = FuncAnimation(
            fig, 
            update,
            frames=final_frame-initial_frame,
            interval=1000/fps,
            blit=True
        )
        
        # Save animation
        writer = FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer)
        plt.close()
        
    def _initialize_cuda(self):
        """Initialize CUDA support if available"""
        try:
            import cupy as cp
            self.cuda_available = True
            self.xp = cp  # Use cupy as numpy replacement
            print("CUDA support enabled")
        except ImportError:
            self.cuda_available = False
            self.xp = np
            print("CUDA not available, using CPU")

    def _compute_pte_cuda(self, source_phase, target_phase, embedding_dim, tau, n_bins):
        """
        Compute phase transfer entropy using CUDA if available.
        
        Parameters:
        -----------
        source_phase, target_phase : array-like
            Phase time series
        embedding_dim : int
            Embedding dimension
        tau : int
            Time delay
        n_bins : int
            Number of bins for phase discretization
        """
        # Initialize CUDA if not done already
        if not hasattr(self, 'cuda_available'):
            self._initialize_cuda()
        
        # Convert to appropriate array type (cupy or numpy)
        source_phase = self.xp.asarray(source_phase)
        target_phase = self.xp.asarray(target_phase)
        
        # Create embedded vectors
        source_embedded = self._create_embedding_cuda(source_phase, embedding_dim, tau)
        target_embedded = self._create_embedding_cuda(target_phase, embedding_dim, tau)
        
        # Discretize phase data
        bins = self.xp.linspace(-self.xp.pi, self.xp.pi, n_bins+1)
        source_discrete = self.xp.digitize(source_embedded, bins) - 1
        target_discrete = self.xp.digitize(target_embedded, bins) - 1
        
        # Prepare data for probability computation
        target_future = target_discrete[1:, 0]
        target_past = target_discrete[:-1, 0]
        source_past = source_discrete[:-1, 0]
        
        # Compute probabilities using CUDA-optimized operations
        # P(t+1)
        p_t = self.xp.zeros(n_bins)
        self.xp.add.at(p_t, target_future, 1)
        p_t = p_t / self.xp.sum(p_t)
        
        # P(t+1, t)
        p_tt = self.xp.zeros((n_bins, n_bins))
        self.xp.add.at(p_tt, (target_future, target_past), 1)
        p_tt = p_tt / self.xp.sum(p_tt)
        
        # P(t+1, t, s)
        p_tts = self.xp.zeros((n_bins, n_bins, n_bins))
        self.xp.add.at(p_tts, (target_future, target_past, source_past), 1)
        p_tts = p_tts / self.xp.sum(p_tts)
        
        # Add small constant to avoid log(0)
        epsilon = 1e-10
        p_t = p_t + epsilon
        p_tt = p_tt + epsilon
        p_tts = p_tts + epsilon
        
        # Compute transfer entropy using vectorized operations
        log_term = self.xp.log2(p_tts * p_t[:, None, None] / 
                            (p_tt[:, :, None] * p_tt[:, None, :]))
        te = self.xp.sum(p_tts * log_term)
        
        # Return result as numpy array
        if self.cuda_available:
            return float(te.get())  # Convert from cupy to numpy
        return float(te) 
    def _test_significance_cuda(self, phase_data, connectivity_matrix, 
                          embedding_dim, tau, n_bins, threshold=0.05,
                          n_surrogates=100, method='simple'):
        """CUDA-accelerated significance testing with debug prints"""
        print("1. Starting method...")
        import cupy as cp
        import time
        
        # Force synchronize and clear memory
        print("2. Synchronizing and clearing memory...")
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        
        # Print initial memory status
        print("3. Checking initial memory status...")
        mem_info = cp.cuda.Device().mem_info
        total_mem = mem_info[1] / (1024**3)
        free_mem = mem_info[0] / (1024**3)
        used_mem = total_mem - free_mem
        print(f"Total: {total_mem:.2f} GB")
        print(f"Used: {used_mem:.2f} GB")
        print(f"Free: {free_mem:.2f} GB")
        
        n_neurons = phase_data.shape[0]
        n_timepoints = phase_data.shape[1]
        
        # Calculate optimal batch size
        print("4. Calculating batch size...")
        available_memory = free_mem * 0.7  # Using 70% of free memory
        memory_per_surrogate = (n_neurons * n_neurons * 8) / (1024**3)  # GB
        optimal_batch_size = int(min(available_memory / memory_per_surrogate, 100))
        print(f"Optimized batch size: {optimal_batch_size}")
        
        # Move data to GPU
        print("5. Moving data to GPU...")
        start_time = time.time()
        phase_data = cp.asarray(phase_data)
        connectivity_matrix = cp.asarray(connectivity_matrix)
        cp.cuda.Stream.null.synchronize()
        transfer_time = time.time() - start_time
        print(f"Data transfer time: {transfer_time:.2f}s")
        
        # Time estimation
        print("6. Starting time estimation...")
        test_neurons = min(5, n_neurons)
        test_surrogates = min(10, n_surrogates)
        
        print("7. Generating test data...")
        test_shifts = cp.random.randint(1, n_timepoints, size=(test_surrogates, test_neurons))
        test_source = cp.zeros((test_neurons, test_surrogates, n_timepoints))
        test_target = cp.zeros((test_neurons, test_surrogates, n_timepoints))
        
        print("8. Running test sample...")
        test_start = time.time()
        try:
            for i in range(test_neurons):
                test_source[i] = cp.stack([
                    cp.roll(phase_data[i], int(shift))
                    for shift in test_shifts[:, i]
                ])
            
            for i in range(test_neurons):
                for j in range(test_neurons):
                    if i != j:
                        for k in range(test_surrogates):
                            _ = self._compute_pte_cuda(
                                test_source[i, k],
                                test_source[j, k],
                                embedding_dim, tau, n_bins
                            )
            test_time = time.time() - test_start
            print("9. Test sample completed successfully")
            
            # Calculate estimates
            time_per_neuron_pair = test_time / (test_neurons * (test_neurons - 1) * test_surrogates)
            total_pairs = n_neurons * (n_neurons - 1)
            estimated_total_time = time_per_neuron_pair * total_pairs * n_surrogates
            
            print(f"Estimated total processing time: {estimated_total_time/60:.1f} minutes")
            print(f"Estimated completion time: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_total_time))}")
        except Exception as e:
            print(f"Error during test sample: {str(e)}")
        
        # Clear test data
        print("10. Clearing test data...")
        del test_shifts, test_source, test_target
        cp.get_default_memory_pool().free_all_blocks()
        
        # Main processing
        print("11. Starting main processing...")
        n_batches = (n_surrogates + optimal_batch_size - 1) // optimal_batch_size
        surrogate_values = cp.zeros((n_neurons, n_neurons, n_surrogates))
        
        print(f"Processing {n_surrogates} surrogates in {n_batches} batches")
        print(f"Neurons: {n_neurons}, Timepoints: {n_timepoints}")
        
        total_start_time = time.time()
        
        try:
            for batch in range(n_batches):
                print(f"12. Starting batch {batch+1}/{n_batches}...")
                batch_start_time = time.time()
                batch_start = batch * optimal_batch_size
                batch_end = min((batch + 1) * optimal_batch_size, n_surrogates)
                current_batch_size = batch_end - batch_start
                
                # Generate shifts
                print(f"13. Generating shifts for batch {batch+1}...")
                shifts = cp.random.randint(1, n_timepoints, 
                                        size=(current_batch_size, n_neurons))
                
                # Process neurons
                print(f"14. Processing neurons for batch {batch+1}...")
                processed_pairs = 0
                total_pairs = n_neurons * (n_neurons - 1)  # Total number of neuron pairs
                start_time = time.time()

                for i in range(n_neurons):
                    source_surrogate = cp.array([
                        cp.roll(phase_data[i], int(shift))
                        for shift in shifts[:, i]
                    ])
                    
                    for j in range(n_neurons):
                        if i != j:
                            target_surrogate = cp.array([
                                cp.roll(phase_data[j], int(shift))
                                for shift in shifts[:, j]
                            ])
                            
                            for k in range(current_batch_size):
                                surrogate_values[i, j, batch_start + k] = self._compute_pte_cuda(
                                    source_surrogate[k],
                                    target_surrogate[k],
                                    embedding_dim, tau, n_bins
                                )
                            
                            processed_pairs += 1
                            
                            # Update progress every 100 pairs
                            if processed_pairs % 100 == 0:
                                elapsed_time = time.time() - start_time
                                progress = processed_pairs / total_pairs
                                eta = (elapsed_time / progress) * (1 - progress) if progress > 0 else 0
                                
                                print(f"Processed {processed_pairs}/{total_pairs} pairs "
                                    f"({progress*100:.1f}%) - "
                                    f"ETA: {eta/60:.1f} minutes - "
                                    f"Current pair: ({i},{j})")
                
                print(f"15. Completed batch {batch+1}")
                batch_time = time.time() - batch_start_time
                print(f"Batch time: {batch_time:.2f}s")
                
        except Exception as e:
            print(f"Error during main processing: {str(e)}")
            return None
        
        # Compute significance
        print("16. Computing significance mask...")
        try:
            if method == 'simple':
                p_values = cp.mean(surrogate_values >= connectivity_matrix[:, :, cp.newaxis], 
                                axis=2)
                significance_mask = p_values < threshold
            elif method == 'fdr':
                p_values = cp.mean(surrogate_values >= connectivity_matrix[:, :, cp.newaxis], 
                                axis=2)
                p_values = cp.asnumpy(p_values)
                significance_mask = self._fdr_correction(p_values, threshold)
                significance_mask = cp.asarray(significance_mask)
            else:  # 'bonferroni'
                corrected_threshold = threshold / (n_neurons * (n_neurons - 1))
                p_values = cp.mean(surrogate_values >= connectivity_matrix[:, :, cp.newaxis], 
                                axis=2)
                significance_mask = p_values < corrected_threshold
            
            result = cp.asnumpy(significance_mask)
            print("17. Significance testing completed")
            
        except Exception as e:
            print(f"Error during significance testing: {str(e)}")
            return None
        
        # Final report
        total_time = time.time() - total_start_time
        print("\n18. Final Report")
        print(f"Total processing time: {total_time/60:.1f} minutes")
        print(f"Significant connections found: {np.sum(result)}")
        
        # Clear memory
        print("19. Clearing GPU memory...")
        del surrogate_values, phase_data, connectivity_matrix
        cp.get_default_memory_pool().free_all_blocks()
        
        print("20. Method completed successfully")
        return result
    
    
    def analyze_neural_phase_connectivity(self, neuron_data, data_type='lfp', 
                                    window_start=0, window_length=1000,
                                    embedding_dim=2, tau=1, n_bins=8,
                                    significance_threshold=0.05,
                                    significance_method='simple',
                                    save_path='phase_connectivity.pkl',
                                    use_cuda=True,
                                    n_surrogates=100):
        """
        Analyze phase-based connectivity between neurons using local field potential data.
        """
        print(f"Starting phase connectivity analysis...")
        
        # Debug: Print DataFrame info
        print("\nDataFrame info:")
        print("Columns:", self.data_df.columns.tolist())
        print("Channel column type:", self.data_df['channel'].dtype)
        print("First few channel values:", self.data_df['channel'].head())
        
        # Debug: Print neuron data info
        print("\nNeuron data info:")
        first_neuron = next(iter(neuron_data.values()))
        print("First neuron keys:", first_neuron.keys())
        print("First neuron channel type:", type(first_neuron['channel']))
        print("First neuron channel value:", first_neuron['channel'])
        
        # Get correct number of neurons
        n_neurons = len(neuron_data.keys())
        print(f"\nTotal number of neurons: {n_neurons}")
        
        # 1. Match neurons to their channels
        matching_indices = []
        neuron_indices = []
        neuron_channels = []
        
        for neuron_idx in neuron_data.keys():
            neuron_channel = neuron_data[neuron_idx]['channel']
            
            # Debug: Print channel comparison
            print(f"\nLooking for neuron {neuron_idx}")
            print(f"Neuron channel: {neuron_channel} (type: {type(neuron_channel)})")
            
            # Convert channel to int explicitly
            neuron_channel = int(neuron_channel)
            
            # Find matching channel
            channel_match = self.data_df[self.data_df['channel'] == neuron_channel]
            
            print(f"Found {len(channel_match)} matches")
            if not channel_match.empty:
                matching_indices.append(channel_match.index[0])
                neuron_indices.append(neuron_idx)
                neuron_channels.append(neuron_channel)
                print(f"Matched to index {channel_match.index[0]}")
            else:
                print(f"Available channels: {sorted(self.data_df['channel'].unique())[:5]}...")
        
        if not matching_indices:
            raise ValueError("No matching channels found for any neurons!")
        
        print(f"Found matches for {len(matching_indices)} neurons")
        
        # 2. Extract phase data
        print(f"Extracting {data_type} phase data...")
        window_end = window_start + window_length
        analytical_data = self._get_analytical_data(data_type, window_start, window_end)
        phase_data = analytical_data['phase'][matching_indices, :]
        
        # 3. Compute phase transfer entropy
        print("Computing phase transfer entropy...")
        connectivity_matrix = np.zeros((n_neurons, n_neurons))  # Full size matrix
        
        # Compute PTE only for matched pairs
        for i, idx_i in enumerate(neuron_indices):
            for j, idx_j in enumerate(neuron_indices):
                if idx_i != idx_j:
                    pte = self._compute_pte_cuda(phase_data[i], phase_data[j],
                                            embedding_dim, tau, n_bins)
                    connectivity_matrix[idx_i, idx_j] = pte
        
        # Perform statistical testing using surrogate data
        print(f"Performing statistical testing using {significance_method} method...")
        if self.cuda_available:
            print("Using CUDA for significance testing")
            significance_mask = self._test_significance_cuda(  # Use CUDA version
                phase_data, 
                connectivity_matrix[np.ix_(neuron_indices, neuron_indices)],
                embedding_dim, tau, n_bins,
                threshold=significance_threshold,
                method=significance_method,
                n_surrogates=n_surrogates
            )
        else:
            print("Using CPU for significance testing")
            significance_mask = self._test_significance(
                phase_data, 
                connectivity_matrix[np.ix_(neuron_indices, neuron_indices)],
                embedding_dim, tau, n_bins,
                threshold=significance_threshold,
                method=significance_method,
                n_surrogates=n_surrogates
            )
        
        # Apply significance mask
        for i, idx_i in enumerate(neuron_indices):
            for j, idx_j in enumerate(neuron_indices):
                if not significance_mask[i, j]:
                    connectivity_matrix[idx_i, idx_j] = 0
        
        # Report number of significant connections
        n_significant = np.sum(connectivity_matrix > 0)
        print(f"Found {n_significant} significant connections out of {len(neuron_indices) * (len(neuron_indices)-1)} possible connections")
        
        # 4. Create directed graph
        print("Creating network graph...")
        G = nx.DiGraph()
        
        # Add all neurons as nodes
        for neuron_idx in neuron_data.keys():
            G.add_node(neuron_idx, 
                    channel=neuron_data[neuron_idx]['channel'],
                    position=neuron_data[neuron_idx]['position'])
        
        # Add edges with weights only for significant connections
        for i in range(n_neurons):
            for j in range(n_neurons):
                if connectivity_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=connectivity_matrix[i, j])
        
        # 5. Save results
        results = {
            'connectivity_matrix': connectivity_matrix,
            'neuron_channels': neuron_channels,
            'matching_indices': matching_indices,
            'neuron_indices': neuron_indices,
            'parameters': {
                'data_type': data_type,
                'window_start': window_start,
                'window_length': window_length,
                'embedding_dim': embedding_dim,
                'tau': tau,
                'n_bins': n_bins,
                'significance_threshold': significance_threshold,
                'significance_method': significance_method,
                'n_surrogates': n_surrogates,
                'significant_connections': n_significant
            }
        }
        
        print(f"Saving results to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Visualize the network
        self._plot_connectivity_network(G, connectivity_matrix, neuron_data)
        
        return connectivity_matrix, G
    
    
    
    
        
        # 2. Extract phase data
        print(f"Extracting {data_type} phase data...")
        window_end = window_start + window_length
        analytical_data = self._get_analytical_data(data_type, window_start, window_end)
        phase_data = analytical_data['phase'][matching_indices, :]
        
        # 3. Compute phase transfer entropy
        print("Computing phase transfer entropy...")
        connectivity_matrix = np.zeros((n_neurons, n_neurons))  # Full size matrix
        
        # Compute PTE only for matched pairs
        for i, idx_i in enumerate(neuron_indices):
            for j, idx_j in enumerate(neuron_indices):
                if idx_i != idx_j:
                    pte = self._compute_pte_cuda(phase_data[i], phase_data[j],
                                            embedding_dim, tau, n_bins)
                    connectivity_matrix[idx_i, idx_j] = pte
        
        # Perform statistical testing using surrogate data
        print(f"Performing statistical testing using {significance_method} method...")
        if self.cuda_available:
            print("Using CUDA for significance testing")
            significance_mask = self._test_significance_cuda(  # Use CUDA version
                phase_data, 
                connectivity_matrix[np.ix_(neuron_indices, neuron_indices)],
                embedding_dim, tau, n_bins,
                threshold=significance_threshold,
                method=significance_method
            )
        else:
            print("Using CPU for significance testing")
            significance_mask = self._test_significance(
                phase_data, 
                connectivity_matrix[np.ix_(neuron_indices, neuron_indices)],
                embedding_dim, tau, n_bins,
                threshold=significance_threshold,
                method=significance_method
            )
        
        # Apply significance mask
        for i, idx_i in enumerate(neuron_indices):
            for j, idx_j in enumerate(neuron_indices):
                if not significance_mask[i, j]:
                    connectivity_matrix[idx_i, idx_j] = 0
        
        # Report number of significant connections
        n_significant = np.sum(connectivity_matrix > 0)
        print(f"Found {n_significant} significant connections out of {len(neuron_indices) * (len(neuron_indices)-1)} possible connections")
        
        # 4. Create directed graph
        print("Creating network graph...")
        G = nx.DiGraph()
        
        # Add all neurons as nodes
        for neuron_idx in neuron_data.keys():
            G.add_node(neuron_idx, 
                    channel=neuron_data[neuron_idx]['channel'],
                    position=neuron_data[neuron_idx]['position'])
        
        # Add edges with weights only for significant connections
        for i in range(n_neurons):
            for j in range(n_neurons):
                if connectivity_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=connectivity_matrix[i, j])
        
        # 5. Save results
        results = {
            'connectivity_matrix': connectivity_matrix,
            'neuron_channels': neuron_channels,
            'matching_indices': matching_indices,
            'neuron_indices': neuron_indices,
            'parameters': {
                'data_type': data_type,
                'window_start': window_start,
                'window_length': window_length,
                'embedding_dim': embedding_dim,
                'tau': tau,
                'n_bins': n_bins,
                'significance_threshold': significance_threshold,
                'significance_method': significance_method,
                'significant_connections': n_significant
            }
        }
        
        print(f"Saving results to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Visualize the network
        self._plot_connectivity_network(G, connectivity_matrix, neuron_data)
        
        return connectivity_matrix, G
    
    

    

    def _compute_pte(self, source_phase, target_phase, embedding_dim, tau, n_bins):
        """
        Compute phase transfer entropy between two phase time series.
        
        Parameters:
        -----------
        source_phase : array-like
            Source phase time series
        target_phase : array-like
            Target phase time series
        embedding_dim : int
            Embedding dimension
        tau : int
            Time delay
        n_bins : int
            Number of bins for phase discretization
        
        Returns:
        --------
        float
            Phase transfer entropy value
        """
        # Ensure inputs are numpy arrays
        source_phase = np.asarray(source_phase)
        target_phase = np.asarray(target_phase)
        
        # Create embedded vectors
        source_embedded = self._create_embedding(source_phase, embedding_dim, tau)
        target_embedded = self._create_embedding(target_phase, embedding_dim, tau)
        
        # Discretize phase data
        bins = np.linspace(-np.pi, np.pi, n_bins+1)
        source_discrete = np.digitize(source_embedded, bins) - 1  # -1 to start from 0
        target_discrete = np.digitize(target_embedded, bins) - 1
        
        # Prepare data for histogram
        target_future = target_discrete[1:, 0]  # t+1
        target_past = target_discrete[:-1, 0]   # t
        source_past = source_discrete[:-1, 0]   # t
        
        # Stack data for 3D histogram
        data_for_hist = np.vstack([target_future, target_past, source_past]).T
        
        # Compute probabilities
        # P(t+1)
        p_t = np.zeros(n_bins)
        unique, counts = np.unique(target_future, return_counts=True)
        p_t[unique] = counts
        p_t = p_t / np.sum(p_t)
        
        # P(t+1, t)
        p_tt = np.zeros((n_bins, n_bins))
        for tf, tp in zip(target_future, target_past):
            p_tt[tf, tp] += 1
        p_tt = p_tt / np.sum(p_tt)
        
        # P(t+1, t, s)
        p_tts = np.zeros((n_bins, n_bins, n_bins))
        for tf, tp, sp in zip(target_future, target_past, source_past):
            p_tts[tf, tp, sp] += 1
        p_tts = p_tts / np.sum(p_tts)
        
        # Add small constant to avoid log(0)
        epsilon = 1e-10
        p_t = p_t + epsilon
        p_tt = p_tt + epsilon
        p_tts = p_tts + epsilon
        
        # Normalize
        p_t /= np.sum(p_t)
        p_tt /= np.sum(p_tt)
        p_tts /= np.sum(p_tts)
        
        # Compute transfer entropy
        te = 0
        for i in range(n_bins):
            for j in range(n_bins):
                for k in range(n_bins):
                    if p_tts[i, j, k] > 0:
                        te += p_tts[i, j, k] * np.log2(p_tts[i, j, k] * p_t[i] / 
                                                    (p_tt[i, j] * p_tt[i, k]))
        
        return max(0, te)

    
    def _create_embedding(self, phase_series, dim, tau):
        """
        Create time-delay embedding of phase data.
        
        Parameters:
        -----------
        phase_series : array-like
            Input phase time series
        dim : int
            Embedding dimension
        tau : int
            Time delay
        
        Returns:
        --------
        numpy.ndarray
            Embedded phase data
        """
        # Ensure input is numpy array
        phase_series = np.asarray(phase_series)
        
        n = len(phase_series) - (dim-1)*tau
        embedding = np.zeros((n, dim))
        
        for i in range(dim):
            embedding[:, i] = phase_series[i*tau:i*tau + n]
        
        return embedding
    
    def _create_embedding_cuda(self, phase_series, dim, tau):
        """
        Create time-delay embedding of phase data with CUDA support.
        
        Parameters:
        -----------
        phase_series : array-like
            Input phase time series
        dim : int
            Embedding dimension
        tau : int
            Time delay
        
        Returns:
        --------
        array-like
            Embedded phase data (using either CuPy or NumPy array)
        """
        # Ensure input is using the correct array type (cupy or numpy)
        phase_series = self.xp.asarray(phase_series)
        
        # Calculate embedding size
        n = len(phase_series) - (dim-1)*tau
        
        # Initialize embedding matrix
        embedding = self.xp.zeros((n, dim))
        
        # Fill embedding matrix
        for i in range(dim):
            embedding[:, i] = phase_series[i*tau:i*tau + n]
        
        return embedding

    def _generate_iaaft_surrogate(self, data, max_iter=1000):
        """
        Generate surrogate data using the IAAFT algorithm.
        Preserves both the amplitude distribution and power spectrum.
        """
        x = data.copy()
        amp = np.sort(x)
        fft = np.fft.fft(x)
        power = np.abs(fft)
        
        # Start with random permutation
        y = np.random.permutation(x)
        
        for _ in range(max_iter):
            previous_y = y.copy()
            
            # Match power spectrum
            fft_y = np.fft.fft(y)
            phase_y = fft_y / np.abs(fft_y)
            y = np.real(np.fft.ifft(power * phase_y))
            
            # Match amplitude distribution
            rank = np.argsort(np.argsort(y))
            y = amp[rank]
            
            # Check convergence
            if np.allclose(y, previous_y, rtol=1e-8):
                break
        
        return y

    def _fdr_correction(self, p_values, threshold):
        """
        Apply False Discovery Rate correction to p-values.
        """
        mask = ~np.isnan(p_values)
        p_values_flat = p_values[mask].flatten()
        
        if len(p_values_flat) == 0:
            return np.zeros_like(p_values, dtype=bool)
        
        # Sort p-values
        sorted_p_indices = np.argsort(p_values_flat)
        sorted_p_values = p_values_flat[sorted_p_indices]
        n_tests = len(sorted_p_values)
        
        # Find threshold
        thresh_line = np.arange(1, n_tests + 1) * threshold / n_tests
        above_thresh = sorted_p_values <= thresh_line
        
        if not np.any(above_thresh):
            return np.zeros_like(p_values, dtype=bool)
        
        # Create mask
        significant = np.zeros_like(p_values_flat, dtype=bool)
        max_significant = np.max(np.where(above_thresh)[0])
        significant[sorted_p_indices[:max_significant + 1]] = True
        
        # Reshape back to original shape
        result = np.zeros_like(p_values, dtype=bool)
        result[mask] = significant
        
        return result

    def _plot_connectivity_network(self, G, connectivity_matrix, neuron_data):
        """
        Plot the neural connectivity network using actual neuron positions.
        
        Parameters:
        -----------
        G : networkx.DiGraph
            Graph object containing connectivity information
        connectivity_matrix : numpy.ndarray
            Matrix of connectivity weights
        neuron_data : dict
            Dictionary containing neuron information including positions
        """
        plt.figure(figsize=(12, 8))
        
        # Create position dictionary using neuron positions from spike sorting
        pos = {}
        for i in range(len(neuron_data)):
            pos[i] = neuron_data[i]['position']  # Using actual computed positions
        
        # Draw the network
        edges = G.edges()
        
        if len(edges) == 0:
            # Handle case with no significant connections
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                node_size=500, alpha=0.6)
            # Add channel numbers as labels
            labels = {i: f"Ch{neuron_data[i]['channel']}" for i in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels)
            plt.title('Neural Phase Connectivity Network\n(No significant connections found)')
        else:
            # Draw network with edges if they exist
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights)
            normalized_weights = [w/max_weight for w in weights]
            
            # Draw nodes (neurons)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                node_size=500, alpha=0.6)
            
            # Draw edges with weights
            nx.draw_networkx_edges(G, pos, edge_color='gray',
                                width=[w*3 for w in normalized_weights],
                                arrowsize=20, alpha=0.6)
            
            # Add channel numbers as labels
            labels = {i: f"Ch{neuron_data[i]['channel']}" for i in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels)
            
            plt.title('Neural Phase Connectivity Network')
            
            # Add colorbar for edge weights
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, 
                                    norm=plt.Normalize(vmin=0, vmax=max_weight))
            plt.colorbar(sm, label='Connection Strength')
        
        # Set equal aspect ratio to preserve spatial relationships
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Add axis labels
        plt.xlabel('X Position (µm)')
        plt.ylabel('Y Position (µm)')
        
        plt.tight_layout()
        plt.show()

    
    def cleanup(self):
        """Release memory explicitly"""
        plt.close('all')  # Close all figures
        self.waves.clear()  # Clear wave data
        self.data_df = None  # Clear DataFrame 
         
    def compute_instantaneous_energy(self, data_type, window_start=0, window_length=None, downsample_factor=1):
        """
        Compute instantaneous energy using Hilbert transform for specified data type.
        
        Parameters:
        -----------
        data_type : str
            Type of data to analyze (e.g., 'lfp', 'gamma', etc.)
        window_start : int
            Start frame of analysis window
        window_length : int, optional
            Length of analysis window in frames. If None, uses all available data
        downsample_factor : int
            Factor by which to downsample the time series
        
        Returns:
        --------
        tuple
            Time points, mean energy, standard deviation of energy
        """
        if window_length is None:
            window_length = self.waves[data_type].shape[1] - window_start
        
        window_end = window_start + window_length
        time_points = np.arange(window_start, window_end, downsample_factor) / self.fs
        
        # Get analytical signal
        analytical_data = self._get_analytical_data(data_type, window_start, window_end)
        
        # Compute instantaneous energy (squared amplitude)
        instantaneous_energy = np.abs(analytical_data['analytical']) ** 2
        
        # Downsample for visualization
        downsampled_energy = instantaneous_energy[:, ::downsample_factor]
        
        # Compute statistics across channels
        mean_energy = np.mean(downsampled_energy, axis=0)
        std_energy = np.std(downsampled_energy, axis=0)
        
        return time_points, mean_energy, std_energy

    def plot_average_energy(self, data_types=['lfp', 'gamma'], window_start=0, window_length=None, 
                        downsample_factor=1, figure_size=(12, 8), dpi=300, normalize=True):
            """
            Create publication-quality plot of average instantaneous energy with standard deviation.
            
            Parameters:
            -----------
            data_types : list
                List of data types to analyze
            window_start : int
                Start frame of analysis window
            window_length : int, optional
                Length of analysis window in frames
            downsample_factor : int
                Factor by which to downsample the time series
            figure_size : tuple
                Size of the figure in inches
            dpi : int
                Resolution of the figure
            normalize : bool
                Whether to normalize energy values. If True, each signal type is normalized 
                to its maximum value. If False, raw energy values are plotted.
            """
            plt.figure(figsize=figure_size, dpi=dpi)
            
            # Extended scientific color scheme (12 colors)
            colors = [
                '#80A0A0',  # stary1
                '#4060A0',  # stary2
                '#200060',  # stary3
                '#F39B7F',  # coral
                '#A22900',  # hot_metal
                '#FDDC5C',  # stary4_(sun)
                '#91D1C2',  # Mint
                '#DC0000',  # Bright red
                '#7E6148',  # Brown
                '#B09C85',  # Taupe
                '#6A3D9A',  # Purple
                '#33A02C'   # Dark green
            ]
            
            for i, data_type in enumerate(data_types):
                # Compute energy metrics
                time_points, mean_energy, std_energy = self.compute_instantaneous_energy(
                    data_type, window_start, window_length, downsample_factor
                )
                
                # Normalize if requested
                if normalize:
                    max_energy = np.max(mean_energy)
                    mean_energy = mean_energy / max_energy
                    std_energy = std_energy / max_energy
                
                # Plot mean and standard deviation
                plt.plot(time_points, mean_energy, label=data_type.upper(), 
                        color=colors[i % len(colors)], linewidth=1.5, zorder=3)
                plt.fill_between(time_points, mean_energy - std_energy, mean_energy + std_energy,
                                color=colors[i % len(colors)], alpha=0.2, zorder=2)
            
            # Configure plot aesthetics
            plt.grid(True, linestyle='--', alpha=0.3, zorder=1)
            plt.xlabel('Time (s)', fontsize=10)
            
            # Adjust y-label based on normalization
            if normalize:
                plt.ylabel('Normalized Instantaneous Amplitude', fontsize=10)
            else:
                plt.ylabel(r'Instantaneous Amplitude ($\mu V^2$)', fontsize=10)
            
            # Add legend with custom style
            legend = plt.legend(frameon=True, framealpha=0.9, edgecolor='none',
                            fontsize=9, loc='upper right')
            legend.get_frame().set_facecolor('white')
            
            # Set title with analysis parameters
            plt.title('Average Instantaneous Amplitude Across Channels\n' +
                    f'Window: {window_start/self.fs:.2f}s - {(window_start+window_length)/self.fs:.2f}s',
                    fontsize=11, pad=10)
            
            # Customize axis appearance
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tick_params(labelsize=9)
            
            # Add text box with analysis parameters
            info_text = (
                f'Sampling rate: {self.fs} Hz\n'
                f'Channels: {self.waves[data_types[0]].shape[0]}\n'
                f'Downsample factor: {downsample_factor}\n'
                f'Normalization: {"On" if normalize else "Off"}'
            )
            #plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                    #fontsize=8, verticalalignment='top',
                    #bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            plt.tight_layout()
            return plt.gcf()
    
    def plot_power_spectral_density(self, data_types=['lfp'], t_start=0, t_end=None,
                                log_x=True, log_y=True, figure_size=(10, 6), dpi=300):
        """
        Create publication-quality power spectral density plot with time window selection.
        
        Parameters:
        -----------
        data_types : list
            List of data types to analyze
        t_start : float
            Start time in seconds
        t_end : float or None
            End time in seconds. If None, uses entire recording
        log_x : bool
            Use logarithmic x-axis
        log_y : bool
            Use logarithmic y-axis
        figure_size : tuple
            Size of the figure in inches
        dpi : int
            Resolution of the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure object
        """
        # Convert time to frames
        window_start = int(t_start * self.fs)
        if t_end is None:
            window_length = None
        else:
            window_length = int((t_end - t_start) * self.fs)
        
        if window_length is None:
            window_length = self.waves[data_types[0]].shape[1] - window_start
        
        window_end = window_start + window_length
        
        # Validate time window
        if t_start < 0:
            raise ValueError("Start time cannot be negative")
        if t_end is not None:
            if t_end <= t_start:
                raise ValueError("End time must be greater than start time")
            if t_end * self.fs > self.waves[data_types[0]].shape[1]:
                raise ValueError("End time exceeds recording duration")
        
        # Set up the figure with publication-quality settings
        plt.figure(figsize=figure_size, dpi=dpi)
        
        # Scientific color scheme
        colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
        
        for i, data_type in enumerate(data_types):
            # Get data segment
            data = self.waves[data_type][:, window_start:window_end]
            
            # Compute PSD for each channel using Welch's method
            freqs, psds = [], []
            for channel in range(data.shape[0]):
                f, psd = signal.welch(data[channel], fs=self.fs, 
                                    nperseg=min(4096, data.shape[1]),
                                    noverlap=None,
                                    detrend='constant',
                                    scaling='density')
                freqs.append(f)
                psds.append(psd)
            
            # Convert to numpy arrays
            freqs = np.array(freqs)
            psds = np.array(psds)
            
            # Compute mean and std across channels
            mean_psd = np.mean(psds, axis=0)
            std_psd = np.std(psds, axis=0)
            
            # Plot with appropriate scale
            plt.plot(freqs[0], mean_psd, label=data_type.upper(), 
                    color=colors[i], linewidth=1.5, zorder=3)
            plt.fill_between(freqs[0], mean_psd - std_psd, mean_psd + std_psd,
                            color=colors[i], alpha=0.2, zorder=2)
        
        # Configure axes scales
        if log_x:
            plt.xscale('log')
        if log_y:
            plt.yscale('log')
        
        # Set axis labels and title
        plt.xlabel('Frequency (Hz)', fontsize=10)
        plt.ylabel('Power Spectral Density (V²/Hz)', fontsize=10)
        plt.title('Power Spectral Density Analysis\n' +
                f'Window: {t_start:.2f}s - {t_end:.2f}s',
                fontsize=11, pad=10)
        
        # Customize grid appearance based on scale
        if log_x and log_y:
            plt.grid(True, which="both", ls="-", alpha=0.2)
        else:
            plt.grid(True, linestyle='--', alpha=0.3, zorder=1)
        
        # Customize axis appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tick_params(labelsize=9)
        
        # Add legend with custom style
        legend = plt.legend(frameon=True, framealpha=0.9, edgecolor='none',
                        fontsize=9, loc='upper right')
        legend.get_frame().set_facecolor('white')
        
        # Add analysis parameters text box
        info_text = (
            f'Sampling rate: {self.fs} Hz\n'
            f'Channels: {self.waves[data_types[0]].shape[0]}\n'
            f'Window length: {(window_length/self.fs):.2f}s'
        )
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        return plt.gcf()

    def compute_band_power(self, data_type, freq_band, t_start=0, t_end=None):
        """
        Compute power in specific frequency band.
        
        Parameters:
        -----------
        data_type : str
            Type of data to analyze
        freq_band : tuple
            (low_freq, high_freq) defining the frequency band
        t_start : float
            Start time in seconds
        t_end : float or None
            End time in seconds
            
        Returns:
        --------
        tuple
            Mean and standard deviation of band power across channels
        """
        # Convert time to frames
        window_start = int(t_start * self.fs)
        if t_end is None:
            window_length = None
        else:
            window_length = int((t_end - t_start) * self.fs)
        
        if window_length is None:
            window_length = self.waves[data_type].shape[1] - window_start
        
        window_end = window_start + window_length
        data = self.waves[data_type][:, window_start:window_end]
        
        # Compute power for each channel
        powers = []
        for channel in range(data.shape[0]):
            f, psd = signal.welch(data[channel], fs=self.fs, 
                                nperseg=min(4096, data.shape[1]))
            # Find indices for frequency band
            idx_band = np.logical_and(f >= freq_band[0], f <= freq_band[1])
            # Integrate power in band
            power = np.trapz(psd[idx_band], f[idx_band])
            powers.append(power)
        
        return np.mean(powers), np.std(powers) 
    def compute_wavelet_entropy(self, data_type, t_start=0, t_end=None, n_wavelets=50):
        """
        Compute wavelet entropy over time using continuous wavelet transform.
        """
        # Convert time to frames
        window_start = int(t_start * self.fs)
        if t_end is None:
            window_length = None
        else:
            window_length = int((t_end - t_start) * self.fs)
        
        if window_length is None:
            window_length = self.waves[data_type].shape[1] - window_start
            
        window_end = window_start + window_length
        
        # Get data segment
        data = self.waves[data_type][:, window_start:window_end]
        
        # Define frequencies for wavelets (log-spaced)
        frequencies = np.logspace(np.log10(1), np.log10(self.fs/2), n_wavelets)
        
        # Initialize arrays for results
        entropies = np.zeros((data.shape[0], data.shape[1]))
        
        for channel in range(data.shape[0]):
            # Compute CWT
            wavelet = signal.morlet2
            scales = self.fs / (2 * frequencies)
            cwtm = signal.cwt(data[channel], wavelet, scales)
            
            # Compute normalized power for each time point
            power = np.abs(cwtm) ** 2
            total_power = np.sum(power, axis=0)
            normalized_power = power / total_power
            
            # Compute entropy
            with np.errstate(divide='ignore', invalid='ignore'):
                entropy = -np.sum(normalized_power * np.log2(normalized_power), axis=0)
                entropy[np.isnan(entropy)] = 0
            
            entropies[channel] = entropy
        
        time_points = np.linspace(t_start, t_end if t_end else window_end/self.fs, 
                                entropies.shape[1])
        mean_entropy = np.mean(entropies, axis=0)
        std_entropy = np.std(entropies, axis=0)
        
        return time_points, mean_entropy, std_entropy

    def plot_wavelet_entropy(self, data_types=['lfp'], t_start=0, t_end=None, 
                        figure_size=(10, 6), dpi=300):
        """
        Create publication-quality wavelet entropy plot.
        """
        plt.figure(figsize=figure_size, dpi=dpi)
        
        colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
        
        for i, data_type in enumerate(data_types):
            time_points, mean_entropy, std_entropy = self.compute_wavelet_entropy(
                data_type, t_start, t_end)
            
            plt.plot(time_points, mean_entropy, label=data_type.upper(),
                    color=colors[i], linewidth=1.5, zorder=3)
            plt.fill_between(time_points, mean_entropy - std_entropy,
                            mean_entropy + std_entropy,
                            color=colors[i], alpha=0.2, zorder=2)
        
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Wavelet Entropy (bits)', fontsize=10)
        plt.title('Time-Resolved Wavelet Entropy Analysis\n' +
                f'Window: {t_start:.2f}s - {t_end:.2f}s',
                fontsize=11, pad=10)
        
        plt.grid(True, linestyle='--', alpha=0.3, zorder=1)
        
        # Customize axis appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tick_params(labelsize=9)
        
        # Add legend
        legend = plt.legend(frameon=True, framealpha=0.9, edgecolor='none',
                        fontsize=9, loc='upper right')
        legend.get_frame().set_facecolor('white')
        
        # Add analysis parameters
        info_text = (
            f'Sampling rate: {self.fs} Hz\n'
            f'Channels: {self.waves[data_types[0]].shape[0]}'
        )
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        return plt.gcf()
    
    def analyze_system_instability(self, data_type='lfp', window_start=0, window_length=None,
                             embedding_dim=3, tau=None, n_windows=50, overlap=0.5):
        """
        Analyze system instability through time using attractor reconstruction.
        
        Parameters:
        -----------
        data_type : str
            Type of data to analyze ('lfp', 'theta', etc.)
        window_start : int
            Start frame for analysis
        window_length : int, optional
            Length of each analysis window (default: automatically determined)
        embedding_dim : int
            Embedding dimension for attractor reconstruction
        tau : int, optional
            Time delay (default: determined using mutual information)
        n_windows : int
            Number of windows to analyze
        overlap : float
            Overlap fraction between consecutive windows (0-1)
        
        Returns:
        --------
        tuple
            (instability_scores, window_times, attractor_metrics)
        """
        import tqdm.auto as tqdm
        import time
        
        print("\n=== Starting System Instability Analysis ===")
        print(f"Data type: {data_type}")
        
        # Get the LFP data
        print("Loading data...")
        data = self.waves[data_type]
        fs = self.fs
        print(f"Loaded data shape: {data.shape}")
        
        # If window_length not provided, determine based on data characteristics
        if window_length is None:
            window_length = int(2 * fs)  # 2 second windows by default
            print(f"Using default window length: {window_length} samples ({window_length/fs:.2f} seconds)")
        
        # If tau not provided, estimate using mutual information
        if tau is None:
            print("Estimating optimal time delay...")
            start_time = time.time()
            tau = self._estimate_time_delay(data)
            print(f"Estimated optimal time delay: {tau} samples ({time.time() - start_time:.2f} seconds)")
        
        # Calculate window parameters
        step_size = int(window_length * (1 - overlap))
        total_frames = data.shape[1]
        window_starts = np.arange(window_start, 
                                window_start + total_frames - window_length,
                                step_size)
        
        # Initialize storage for results
        instability_scores = []
        attractor_metrics = []
        window_times = []
        
        print("\nProcessing windows...")
        start_time = time.time()
        
        # Process each window with progress bar
        for win_start in tqdm.tqdm(window_starts[:n_windows], desc="Analyzing windows", ncols=100):
            win_end = win_start + window_length
            window_data = data[:, win_start:win_end]
            
            # Reconstruct attractor for each channel
            attractors = []
            for channel_data in window_data:
                attractor = self._reconstruct_attractor(channel_data, 
                                                    embedding_dim, 
                                                    tau)
                attractors.append(attractor)
            
            # Compute metrics for this window
            metrics = self._compute_attractor_metrics(attractors)
            attractor_metrics.append(metrics)
            
            # Compute instability score
            instability = self._compute_instability_score(metrics)
            instability_scores.append(instability)
            
            # Store window time
            window_times.append(win_start / fs)
        
        # Convert to arrays
        instability_scores = np.array(instability_scores)
        window_times = np.array(window_times)
        
        total_time = time.time() - start_time
        print(f"\nAnalysis completed in {total_time:.2f} seconds")
        print(f"Processed {n_windows} windows of {window_length} samples each")
        print(f"Average processing time per window: {total_time/n_windows:.2f} seconds")
        
        # Plot results
        print("\nGenerating visualization...")
        self._plot_system_instability(instability_scores, window_times, 
                                    attractor_metrics)
        
        return instability_scores, window_times, attractor_metrics

    def _estimate_time_delay(self, data, max_delay=100):
        """
        Estimate optimal time delay using mutual information.
        """
        from sklearn.metrics import mutual_info_score
        
        # Use first channel as representative
        signal = data[0]
        mi_scores = []
        
        for delay in tqdm.tqdm(range(1, max_delay), desc="Calculating mutual information", ncols=100):
            x1 = signal[:-delay]
            x2 = signal[delay:]
            
            # Compute mutual information
            hist_2d, _, _ = np.histogram2d(x1, x2, bins=50)
            mi = mutual_info_score(None, None, contingency=hist_2d)
            mi_scores.append(mi)
        
        # Find first local minimum
        min_idx = np.argmin(mi_scores)
        return min_idx + 1

    def _reconstruct_attractor(self, signal, embedding_dim, tau):
        """
        Reconstruct attractor using time delay embedding.
        """
        n_points = len(signal) - (embedding_dim - 1) * tau
        attractor = np.zeros((n_points, embedding_dim))
        
        for i in range(embedding_dim):
            attractor[:, i] = signal[i * tau : i * tau + n_points]
        
        return attractor

    def _compute_attractor_metrics(self, attractors):
        """
        Compute various metrics for the reconstructed attractors.
        """
        metrics = {
            'lyapunov_exponents': [],
            'correlation_dim': [],
            'recurrence_rate': [],
            'determinism': []
        }
        
        for i, attractor in enumerate(tqdm.tqdm(attractors, desc="Computing metrics", leave=False, ncols=100)):
            # Estimate largest Lyapunov exponent
            lyap = self._estimate_lyapunov(attractor)
            metrics['lyapunov_exponents'].append(lyap)
            
            # Estimate correlation dimension
            corr_dim = self._estimate_correlation_dimension(attractor)
            metrics['correlation_dim'].append(corr_dim)
            
            # Compute recurrence metrics
            rr, det = self._compute_recurrence_metrics(attractor)
            metrics['recurrence_rate'].append(rr)
            metrics['determinism'].append(det)
        
        return metrics

    def _estimate_lyapunov(self, attractor, n_neighbors=5):
        """
        Estimate largest Lyapunov exponent using nearest neighbors.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(attractor)
        distances, indices = nbrs.kneighbors(attractor)
        
        # Compute divergence rates
        divergence_rates = []
        for i in range(len(attractor)-1):
            for j in range(1, n_neighbors+1):
                neighbor_idx = indices[i, j]
                if neighbor_idx < len(attractor)-1:
                    d0 = distances[i, j]
                    d1 = np.linalg.norm(attractor[i+1] - attractor[neighbor_idx+1])
                    if d0 > 0:
                        rate = np.log(d1/d0)
                        divergence_rates.append(rate)
        
        return np.mean(divergence_rates)

    def _estimate_correlation_dimension(self, attractor, max_r=None):
        """
        Estimate correlation dimension using Grassberger-Procaccia algorithm.
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Compute pairwise distances
        distances = euclidean_distances(attractor)
        
        if max_r is None:
            max_r = np.median(distances)
        
        # Compute correlation sum for different r
        r_values = np.logspace(-3, np.log10(max_r), 20)
        c_values = []
        
        for r in r_values:
            c = np.sum(distances < r) / (len(attractor) * (len(attractor) - 1))
            c_values.append(c)
        
        # Estimate dimension from slope
        slope = np.diff(np.log(c_values)) / np.diff(np.log(r_values))
        return np.median(slope)

    def _compute_recurrence_metrics(self, attractor, threshold=None):
        """
        Compute recurrence rate and determinism from recurrence plot.
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Compute distance matrix
        distances = euclidean_distances(attractor)
        
        # Set threshold if not provided
        if threshold is None:
            threshold = np.median(distances) * 0.1
        
        # Create recurrence matrix
        recurrence = distances < threshold
        
        # Compute recurrence rate
        rr = np.sum(recurrence) / (len(attractor) ** 2)
        
        # Compute determinism (ratio of recurrent points forming diagonal lines)
        min_line = 2
        diagonals = []
        for i in range(-len(attractor)+min_line, len(attractor)-min_line):
            diagonal = np.diagonal(recurrence, offset=i)
            current_line = 0
            for point in diagonal:
                if point:
                    current_line += 1
                elif current_line >= min_line:
                    diagonals.append(current_line)
                    current_line = 0
                else:
                    current_line = 0
        
        det = sum(diagonals) / max(np.sum(recurrence), 1)
        
        return rr, det

    def _compute_instability_score(self, metrics):
        """
        Compute overall instability score from attractor metrics.
        """
        # Normalize and combine metrics
        lyap_norm = np.mean(metrics['lyapunov_exponents'])
        dim_norm = np.mean(metrics['correlation_dim'])
        rec_norm = 1 - np.mean(metrics['recurrence_rate'])
        det_norm = 1 - np.mean(metrics['determinism'])
        
        # Weight and combine metrics
        weights = [0.4, 0.2, 0.2, 0.2]  # Can be adjusted based on importance
        score = (weights[0] * lyap_norm + 
                weights[1] * dim_norm + 
                weights[2] * rec_norm +
                weights[3] * det_norm)
        
        return score

    def _plot_system_instability(self, instability_scores, window_times, metrics):
        """
        Create comprehensive visualization of system instability analysis.
        """
        plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Plot 1: Instability Score
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(window_times, instability_scores, 'b-', linewidth=2)
        ax1.fill_between(window_times, 0, instability_scores, alpha=0.2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Instability Score')
        ax1.set_title('System Instability Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Lyapunov Exponents
        ax2 = plt.subplot(gs[1, 0])
        lyap = [m['lyapunov_exponents'] for m in metrics]
        ax2.boxplot(lyap)
        ax2.set_xlabel('Time Window')
        ax2.set_ylabel('Lyapunov Exponent')
        ax2.set_title('Distribution of Lyapunov Exponents')
        
        # Plot 3: Phase Space Statistics
        ax3 = plt.subplot(gs[1, 1])
        corr_dim = [np.mean(m['correlation_dim']) for m in metrics]
        det = [np.mean(m['determinism']) for m in metrics]
        ax3.scatter(corr_dim, det, c=instability_scores, cmap='viridis')
        ax3.set_xlabel('Correlation Dimension')
        ax3.set_ylabel('Determinism')
        ax3.set_title('Phase Space Characteristics')
        plt.colorbar(ax3.collections[0], label='Instability Score')
        
        plt.tight_layout()
        
        # Add summary statistics
        summary_text = f"""Analysis Summary:
        - Time range: {window_times[0]:.1f}s to {window_times[-1]:.1f}s
        - Mean instability: {np.mean(instability_scores):.3f}
        - Peak instability: {np.max(instability_scores):.3f}
        - Peak time: {window_times[np.argmax(instability_scores)]:.1f}s
        """
        plt.figtext(0.02, 0.02, summary_text, fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        print("\nVisualization complete")
        print(summary_text)
        plt.show()

    def analyze_phase_coherence(
        self,
        data_type='theta',
        window_start=0,
        window_length=1000,
        n_bins=100,
        smoothing_sigma=15,
        spike_data=None,
        fr_smoothing_sigma=15,
        figsize=(15, 6),
        fig=None,
        ax=None,
        show_insets=False,
        mi_color='navy'  # New parameter for MI line color
    ):
        """
        Analyzes and visualizes phase coherence across time for the specified data type.
        If spike data object is provided, also plots population firing rate.
        
        Parameters
        ----------
        data_type : str
            Type of data to analyze. Must be one of:
            - 'delta'  : delta band (0.5-4 Hz)
            - 'theta'  : theta band (4-8 Hz)
            - 'alpha'  : alpha band (8-13 Hz)
            - 'beta'   : beta band (13-30 Hz)
            - 'gamma'  : gamma band (30-80 Hz)
            - 'high_gamma' : high gamma band (80-150 Hz)
            - 'lfp'    : raw LFP signal
        window_start : int
            Start frame for analysis window
        window_length : int
            Length of analysis window in frames
        n_bins : int
            Number of bins for phase distribution histograms
        smoothing_sigma : float
            Standard deviation for Gaussian smoothing kernel for MI
        spike_data : SpikeData object, optional
            Spike data object containing spike times and methods for analysis
        fr_smoothing_sigma : float
            Standard deviation for Gaussian smoothing kernel for firing rate
        figsize : tuple
            Figure size for output plot (width, height)
        fig : matplotlib.figure.Figure, optional
            Existing figure to plot on
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on
        show_insets : bool, optional
            Whether to show inset polar plots of min/max phase distributions (default: False)
        mi_color : str, optional
            Color for the MI line plot (default: 'navy'). Can be any valid matplotlib color
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter1d
        
        # Input validation
        valid_types = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'lfp']
        if data_type not in valid_types:
            raise ValueError(f"Invalid data type: {data_type}. Must be one of: {valid_types}")
        
        if data_type not in self.waves:
            raise ValueError(f"Data type {data_type} not found in waves dictionary. Available types: {list(self.waves.keys())}")
        
        window_end = window_start + window_length
        if window_end > self.waves[data_type].shape[1]:
            raise ValueError("Analysis window exceeds data length")
        
        # Get phase data for the specified window
        analytical_data = self._get_analytical_data(data_type, window_start, window_end)
        phase_data = analytical_data['phase']
        
        # Initialize arrays for modulation index calculation
        time_points = np.arange(window_length)
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        modulation_index = np.zeros(window_length)
        
        # Calculate modulation index at each time point
        for t in range(window_length):
            # Compute phase distribution
            hist, _ = np.histogram(phase_data[:, t], bins=bins, density=True)
            hist = hist / hist.sum()  # Normalize
            
            # Calculate modulation index (deviation from uniform distribution)
            uniform_dist = np.ones(n_bins) / n_bins
            modulation_index[t] = np.sqrt(np.sum((hist - uniform_dist) ** 2))
        
        # Apply smoothing
        smoothed_mi = gaussian_filter1d(modulation_index, sigma=smoothing_sigma)
        
        # Create single figure with proper margins and layout
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Clear previous plots if any
        ax.clear()
        
        # Plot modulation index with user-specified color
        time_axis = np.arange(window_start, window_end) / self.fs
        ax.plot(time_axis, smoothed_mi, linewidth=2, color=mi_color, alpha=0.8)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Phase Coherence (MI)', fontsize=12)
        ax.set_xlim(time_axis[0], time_axis[-1])
        
        # If spike data is provided, compute and plot population firing rate
        pop_rate = None
        if spike_data is not None:
            # Get population firing rate using spike data object's binned method
            pop_rate = spike_data.binned(bin_size=1)  # 1ms bins
            pop_fr = gaussian_filter1d(pop_rate.astype(float), sigma=fr_smoothing_sigma)
            
            # Create time bins for firing rate that match the window
            bins_fr = np.linspace(0, spike_data.length, pop_rate.shape[0]) / 1000  # Convert to seconds
            
            # Select only the data within our window
            mask = (bins_fr >= time_axis[0]) & (bins_fr <= time_axis[-1])
            bins_fr = bins_fr[mask]
            pop_fr = pop_fr[mask]
            
            # Create second y-axis for firing rate
            ax_fr = ax.twinx()
            ax_fr.plot(bins_fr, pop_fr, color='firebrick', linewidth=2, alpha=0.75)
            ax_fr.set_ylabel('Population Firing Rate (Hz)', color='firebrick', fontsize=12)
            ax_fr.tick_params(axis='y', labelcolor='firebrick')
            ax_fr.spines['right'].set_color('firebrick')
            ax_fr.spines['right'].set_linewidth(2)
            ax_fr.set_ylim(0, max(pop_fr)*1.2)
            
            # Hide top spine
            ax_fr.spines['top'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if show_insets:
            # Set up inset parameters
            inset_width = 0.1
            inset_height = 0.23
            left_position = 0.02
            top_position = 0.95
            spacing = 0.05
            
            # Calculate angles for polar plots
            angles = bins[:-1] + np.pi/n_bins
            angles = np.append(angles, angles[0])
            
            # Find min/max points
            min_idx = np.argmin(smoothed_mi)
            max_idx = np.argmax(smoothed_mi)
            
            # Plot min MI distribution (left)
            axins1 = ax.inset_axes([left_position, top_position-inset_height, 
                                inset_width, inset_height], projection='polar')
            hist_min, _ = np.histogram(phase_data[:, min_idx], bins=bins, density=True)
            hist_min = hist_min/hist_min.sum()  # Normalize
            hist_min = np.append(hist_min, hist_min[0])
            
            # Use a lighter version of the main MI color for the min distribution
            axins1.plot(angles, hist_min, color=mi_color, linewidth=2, alpha=0.4)
            axins1.fill(angles, hist_min, color=mi_color, alpha=0.2)
            axins1.set_xticks([0, np.pi/2, np.pi, -np.pi/2])
            axins1.set_xticklabels(['0', 'π/2', 'π', '-π/2'], fontsize=8)
            axins1.set_yticks([])
            axins1.grid(True, alpha=0.3, color='gray')
            axins1.spines['polar'].set_visible(True)
            axins1.spines['polar'].set_linewidth(0.5)
            axins1.spines['polar'].set_color('gray')
            axins1.set_theta_zero_location('N')
            axins1.set_theta_direction(-1)
            axins1.set_thetalim(-np.pi, np.pi)
            ax.text(left_position + 0.5*inset_width, top_position-1.15*inset_height,
                    f'Min MI = {smoothed_mi[min_idx]:.3f}\nt = {time_axis[min_idx]:.2f}s',
                    ha='center', va='top', fontsize=8, transform=ax.transAxes)
            
            # Plot max MI distribution (right)
            axins2 = ax.inset_axes([left_position + inset_width + spacing, top_position-inset_height, 
                                inset_width, inset_height], projection='polar')
            hist_max, _ = np.histogram(phase_data[:, max_idx], bins=bins, density=True)
            hist_max = hist_max/hist_max.sum()  # Normalize
            hist_max = np.append(hist_max, hist_max[0])
            
            axins2.plot(angles, hist_max, color=mi_color, linewidth=2, alpha=0.8)
            axins2.fill(angles, hist_max, color=mi_color, alpha=0.3)
            axins2.set_xticks([0, np.pi/2, np.pi, -np.pi/2])
            axins2.set_xticklabels(['0', 'π/2', 'π', '-π/2'], fontsize=8)
            axins2.set_yticks([])
            axins2.grid(True, alpha=0.3, color='gray')
            axins2.spines['polar'].set_visible(True)
            axins2.spines['polar'].set_linewidth(0.5)
            axins2.spines['polar'].set_color('gray')
            axins2.set_theta_zero_location('N')
            axins2.set_theta_direction(-1)
            axins2.set_thetalim(-np.pi, np.pi)
            ax.text(left_position + 1.5*inset_width + spacing, top_position-1.15*inset_height,
                    f'Max MI = {smoothed_mi[max_idx]:.3f}\nt = {time_axis[max_idx]:.2f}s',
                    ha='center', va='top', fontsize=8, transform=ax.transAxes)
        
        # Add title
        ax.set_title(f'Phase Coherence Analysis: {data_type}', fontsize=14)
        
        # Final layout adjustment
        plt.tight_layout()
        
        return modulation_index, smoothed_mi, pop_rate, fig, ax
    
    def plot_multi_band_coherence(
        self,
        window_start=0,
        window_length=1000,
        n_bins=1000,
        smoothing_sigma=15,
        spike_data=None,
        fr_smoothing_sigma=15,
        figsize=(15, 6),
        data_types=None,
        mi_colors=None
    ):
        """
        Creates a two-panel plot showing phase coherence analysis across multiple frequency bands.
        Left panel shows LFP MI with firing rate, right panel shows multiple frequency bands.
        
        Parameters
        ----------
        window_start : int
            Start frame for analysis window
        window_length : int
            Length of analysis window in frames
        n_bins : int
            Number of bins for phase distribution histograms
        smoothing_sigma : float
            Standard deviation for Gaussian smoothing kernel for MI
        spike_data : SpikeData object, optional
            Spike data object containing spike times and methods for analysis
        fr_smoothing_sigma : float
            Standard deviation for Gaussian smoothing kernel for firing rate
        figsize : tuple
            Figure size for output plot (width, height)
        data_types : list of str, optional
            List of frequency bands to plot. If None, defaults to ['THETA', 'DELTA', 'HIGH_GAMMA']
        mi_colors : dict, optional
            Dictionary mapping data types to colors. If None, uses default colors.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter1d
        
        # Default frequency bands if none provided
        if data_types is None:
            data_types = ['THETA', 'DELTA', 'HIGH_GAMMA']
        
        # Convert all band names to uppercase
        data_types = [dt.upper() for dt in data_types]
        
        # Default scientific color scheme
        default_colors = {
            'LFP':   '#FDDC5C',# stary1
            'DELTA':  '#80A0A0',  # stary2
            'THETA': '#4060A0',  # stary3
            'ALPHA': '#200060',  # stary4
            'BETA': '#F39B7F',   # Coral
            'GAMMA': '#8491B4',  # Purple-gray
            'HIGH_GAMMA': '#200060'  # Mint
        }
        
        # If custom colors provided, convert keys to uppercase
        if mi_colors is not None:
            mi_colors = {k.upper(): v for k, v in mi_colors.items()}
            # Fill in any missing colors from default
            for k, v in default_colors.items():
                if k not in mi_colors:
                    mi_colors[k] = v
        else:
            mi_colors = default_colors
        
        # Create figure with two subplots and set style
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
        
        # Configure grid style
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.color'] = '#cccccc'
        plt.rcParams['axes.axisbelow'] = True
        
        # Time axis in seconds
        time_axis = np.arange(window_start, window_start + window_length) / self.fs
        
        # Left panel - LFP MI with firing rate
        ax1 = fig.add_subplot(gs[0])
        
        # Compute LFP MI
        analytical_data = self._get_analytical_data('theta', window_start, window_start + window_length)
        phase_data = analytical_data['phase']
        
        # Calculate MI
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        modulation_index = np.zeros(window_length)
        for t in range(window_length):
            hist, _ = np.histogram(phase_data[:, t], bins=bins, density=True)
            hist = hist / hist.sum()
            uniform_dist = np.ones(n_bins) / n_bins
            modulation_index[t] = np.sqrt(np.sum((hist - uniform_dist) ** 2))
        
        # Smooth MI
        smoothed_mi = gaussian_filter1d(modulation_index, sigma=smoothing_sigma)
        
        # Plot LFP MI - always black in left panel
        ax1.plot(time_axis, smoothed_mi, linewidth=3, color='#b17f89', label='LFP (theta)', alpha=0.8)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Phase Coherence (MI)', fontsize=12)
        
        # Add firing rate if provided
        if spike_data is not None:
            ax1_fr = ax1.twinx()
            pop_rate = spike_data.binned(bin_size=1)
            pop_fr = gaussian_filter1d(pop_rate.astype(float), sigma=fr_smoothing_sigma)
            
            # Create time bins for firing rate
            bins_fr = np.linspace(0, spike_data.length, pop_rate.shape[0]) / 1000
            
            # Select data within window
            mask = (bins_fr >= time_axis[0]) & (bins_fr <= time_axis[-1])
            bins_fr = bins_fr[mask]
            pop_fr = pop_fr[mask]
            
            ax1_fr.plot(bins_fr, pop_fr, color='#ccb364', linewidth=3, alpha=0.7)
            ax1_fr.set_ylabel('Population Firing Rate (Hz)', color='#ccb364', fontsize=12)
            ax1_fr.tick_params(axis='y', labelcolor='#ccb364')
            ax1_fr.spines['right'].set_color('#ccb364')
            ax1_fr.spines['right'].set_linewidth(2)
            # Dilate y-axis scale for firing rate to reduce overlap
            max_fr = max(pop_fr)
            ax1_fr.set_ylim(0, 3)  # Increased from 1.8 to 2.5
            
            # Hide grid for firing rate axis
            ax1_fr.grid(False)
            
            # Hide top spine
            ax1_fr.spines['top'].set_visible(False)
        
        ax1.spines['top'].set_visible(False)
        ax1.set_title('LFP (theta) Phase Coherence and Firing Rate', fontsize=12)
        
        # Right panel - Multiple frequency bands
        ax2 = fig.add_subplot(gs[1])
        
        # Compute and plot MI for each band
        for band in data_types:
            analytical_data = self._get_analytical_data(band.lower(), window_start, window_start + window_length)
            phase_data = analytical_data['phase']
            
            # Calculate MI
            modulation_index = np.zeros(window_length)
            for t in range(window_length):
                hist, _ = np.histogram(phase_data[:, t], bins=bins, density=True)
                hist = hist / hist.sum()
                uniform_dist = np.ones(n_bins) / n_bins
                modulation_index[t] = np.sqrt(np.sum((hist - uniform_dist) ** 2))
            
            # Smooth MI
            smoothed_mi = gaussian_filter1d(modulation_index, sigma=smoothing_sigma)
            
            # Plot band MI
            ax2.plot(time_axis, smoothed_mi, linewidth=2, color=mi_colors[band], label=band, alpha=0.8)
        
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Phase Coherence (MI)', fontsize=12)
        ax2.spines['top'].set_visible(False)
        
        # Configure legend without box
        ax2.legend(fontsize=10, loc='upper right', frameon=False)
        ax2.set_title('Multi-band Phase Coherence', fontsize=12)
        
        # Set same time limits for both panels
        ax1.set_xlim(time_axis[0], time_axis[-1])
        ax2.set_xlim(time_axis[0], time_axis[-1])
        
        # Add overall title
        fig.suptitle('Phase Coherence Analysis Across Frequency Bands', 
                    fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        return fig, (ax1, ax2)
    
    def plot_stacked_band_coherence(
        self,
        window_start=0,
        window_length=1000,
        n_bins=100,
        smoothing_sigma=15,
        spike_data=None,
        fr_smoothing_sigma=15,
        figsize=(15, 6),
        data_types=None,
        mi_colors=None,
        lfp_color='#00a0c0',
        fr_color='#c00000'
    ):
        """
        Creates a publication-style plot with two main panels: left panel shows LFP MI with firing rate,
        right panel shows stacked subplots for different frequency bands.
        
        Parameters
        ----------
        window_start : int
            Start frame for analysis window
        window_length : int
            Length of analysis window in frames
        n_bins : int
            Number of bins for phase distribution histograms
        smoothing_sigma : float
            Standard deviation for Gaussian smoothing kernel for MI
        spike_data : SpikeData object, optional
            Spike data object containing spike times and methods for analysis
        fr_smoothing_sigma : float
            Standard deviation for Gaussian smoothing kernel for firing rate
        figsize : tuple
            Figure size for output plot (width, height)
        data_types : list of str, optional
            List of frequency bands to plot. If None, defaults to ['delta', 'theta', 'gamma']
        mi_colors : dict, optional
            Dictionary mapping data types to colors. If None, uses default colors.
        lfp_color : str, optional
            Color for LFP MI line in left panel. Default is turquoise '#00a0c0'.
        fr_color : str, optional
            Color for firing rate line in left panel. Default is red '#c00000'.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from scipy.ndimage import gaussian_filter1d
        
        # Default frequency bands if none provided
        if data_types is None:
            data_types = ['delta', 'theta', 'gamma']
        
        # Ensure data types are lowercase for consistency with waves dictionary
        data_types = [dt.lower() for dt in data_types]
        
        # Default colors for frequency bands
        default_colors = {
            'delta': '#0060c0',    # Light blue
            'theta': '#0000a0',    # Dark blue
            'alpha': '#400080',    # Purple-blue
            'beta': '#800060',     # Purple
            'gamma': '#a000a0',    # Bright purple
            'high_gamma': '#c000c0' # Magenta
        }
        
        # If custom colors provided, use them, otherwise use defaults
        if mi_colors is not None:
            mi_colors = {k.lower(): v for k, v in mi_colors.items()}
            # Fill in missing colors from defaults
            for k, v in default_colors.items():
                if k not in mi_colors:
                    mi_colors[k] = v
        else:
            mi_colors = default_colors
        
        # Create figure with two main panels
        fig = plt.figure(figsize=figsize)
        
        # Use gridspec for precise control over layout
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15, figure=fig)
        
        # Left panel is a single plot
        ax_left = fig.add_subplot(gs[0])
        
        # Right panel will be subdivided into stacked subplots
        gs_right = gridspec.GridSpecFromSubplotSpec(len(data_types), 1, 
                                                subplot_spec=gs[1], 
                                                hspace=0.05)
        
        # Time axis in seconds
        time_axis = np.arange(window_start, window_start + window_length) / self.fs
        
        # Compute and plot LFP MI for left panel
        analytical_data = self._get_analytical_data('lfp', window_start, window_start + window_length)
        phase_data = analytical_data['phase']
        
        # Calculate MI
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        modulation_index = np.zeros(window_length)
        for t in range(window_length):
            hist, _ = np.histogram(phase_data[:, t], bins=bins, density=True)
            hist = hist / hist.sum()
            uniform_dist = np.ones(n_bins) / n_bins
            modulation_index[t] = np.sqrt(np.sum((hist - uniform_dist) ** 2))
        
        # Smooth MI
        lfp_mi = gaussian_filter1d(modulation_index, sigma=smoothing_sigma)
        
        # Plot LFP MI
        ax_left.plot(time_axis, lfp_mi, linewidth=2, color=lfp_color, alpha=0.8)
        ax_left.set_ylabel('MI', fontsize=12, color=lfp_color, rotation=0, labelpad=15)
        
        # Configure left y-axis to match LFP MI color
        ax_left.tick_params(axis='y', labelcolor=lfp_color)
        ax_left.spines['left'].set_color(lfp_color)
        
        # Add firing rate if provided
        if spike_data is not None:
            ax_fr = ax_left.twinx()
            pop_rate = spike_data.binned(bin_size=1)
            pop_fr = gaussian_filter1d(pop_rate.astype(float), sigma=fr_smoothing_sigma)
            
            # Create time bins for firing rate
            bins_fr = np.linspace(0, spike_data.length, pop_rate.shape[0]) / 1000
            
            # Select data within window
            mask = (bins_fr >= time_axis[0]) & (bins_fr <= time_axis[-1])
            bins_fr = bins_fr[mask]
            pop_fr = pop_fr[mask]
            
            # Plot firing rate
            ax_fr.plot(bins_fr, pop_fr, linewidth=2, color=fr_color, alpha=0.8)
            ax_fr.set_ylabel('Firing Rate', fontsize=12, color=fr_color, rotation=270, labelpad=20)
            
            # Configure right y-axis to match firing rate color
            ax_fr.tick_params(axis='y', labelcolor=fr_color)
            ax_fr.spines['right'].set_color(fr_color)
            ax_fr.grid(False)
            
            # Dilate y-axis scale for firing rate to reduce overlap
            max_fr = max(pop_fr)
            ax_fr.set_ylim(0, max_fr*2.5)
        
        # Hide top spine and add minimal grid
        ax_left.spines['top'].set_visible(False)
        ax_left.grid(True, alpha=0.3, axis='x')
        
        # Add text label for LFP in upper left corner
        ax_left.text(0.05, 0.95, 'LFP', transform=ax_left.transAxes, 
                    fontsize=12, color=lfp_color, fontweight='bold',
                    verticalalignment='top', horizontalalignment='left')
        
        # Only show x-axis label for left panel
        ax_left.set_xlabel('time (s)', fontsize=12)
        
        # Create subplots for each frequency band in right panel
        axes_right = []
        
        for i, band in enumerate(data_types):
            # Create subplot
            ax = fig.add_subplot(gs_right[i])
            axes_right.append(ax)
            
            # Compute band MI
            analytical_data = self._get_analytical_data(band, window_start, window_start + window_length)
            phase_data = analytical_data['phase']
            
            # Calculate MI
            modulation_index = np.zeros(window_length)
            for t in range(window_length):
                hist, _ = np.histogram(phase_data[:, t], bins=bins, density=True)
                hist = hist / hist.sum()
                uniform_dist = np.ones(n_bins) / n_bins
                modulation_index[t] = np.sqrt(np.sum((hist - uniform_dist) ** 2))
            
            # Smooth MI
            band_mi = gaussian_filter1d(modulation_index, sigma=smoothing_sigma)
            
            # Plot band MI
            ax.plot(time_axis, band_mi, linewidth=2, color=mi_colors[band], alpha=0.8)
            
            # Add band name as text label in upper right
            ax.text(0.95, 0.8, band.capitalize(), transform=ax.transAxes,
                    fontsize=12, color=mi_colors[band], fontweight='bold',
                    verticalalignment='top', horizontalalignment='right')
            
            # Configure axes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set minimal grid
            ax.grid(True, alpha=0.3, axis='x')
            
            # Hide x-axis labels for all but the bottom subplot
            if i < len(data_types) - 1:
                ax.set_xticklabels([])
                ax.set_xlabel('')
            else:
                ax.set_xlabel('time (s)', fontsize=12)
            
            # Ensure all subplots have the same x-axis limits
            ax.set_xlim(time_axis[0], time_axis[-1])
        
        # Make sure left panel has same x-axis limits
        ax_left.set_xlim(time_axis[0], time_axis[-1])
        
        # Final layout adjustments
        plt.tight_layout()
        
        return fig, (ax_left, axes_right)
    
    def compute_comodulogram(self, phase_bands, amp_bands, window_start=0, window_length=None,
                            n_bins=20, surrogate_tests=200, p_threshold=0.05):
        """
        Compute PAC between multiple frequency bands to create a comodulogram.
        
        Parameters
        ----------
        phase_bands : list of str
            Names of frequency bands to use for phase
        amp_bands : list of str
            Names of frequency bands to use for amplitude
        window_start : int
            Start frame for analysis
        window_length : int, optional
            Length of analysis window in frames
        n_bins : int
            Number of phase bins for MI calculation
        surrogate_tests : int
            Number of surrogate tests for significance
        p_threshold : float
            P-value threshold for significance
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'comodulogram': 2D array of MI values
            - 'significance_mask': 2D boolean array of significant values
            - 'p_values': 2D array of p-values
        """
        try:
            # Initialize results matrices
            comod = np.zeros((len(phase_bands), len(amp_bands)))
            p_vals = np.zeros_like(comod)
            sig_mask = np.zeros_like(comod, dtype=bool)
            
            # Compute PAC for each band combination
            for i, phase_band in enumerate(phase_bands):
                for j, amp_band in enumerate(amp_bands):
                    results = self.compute_pac(
                        phase_band, amp_band,
                        window_start, window_length,
                        n_bins, surrogate_tests,
                        p_threshold
                    )
                    
                    if results is not None:
                        comod[i, j] = results['mi']
                        p_vals[i, j] = results['p_value']
                        sig_mask[i, j] = results['is_significant']
                        
            return {
                'comodulogram': comod,
                'significance_mask': sig_mask,
                'p_values': p_vals
            }
            
        except Exception as e:
            print(f"Error in compute_comodulogram: {str(e)}")
            return None

    def plot_comodulogram(self, results, phase_bands, amp_bands, save_path=None):
        """
        Plot comodulogram results.
        
        Parameters
        ----------
        results : dict
            Results from compute_comodulogram
        phase_bands : list of str
            Names of phase frequency bands
        amp_bands : list of str
            Names of amplitude frequency bands
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if results is None:
            print("No results to plot")
            return None
            
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot comodulogram
            im = ax.imshow(results['comodulogram'], 
                        aspect='auto', 
                        interpolation='nearest',
                        origin='lower')
            
            # Add significance markers
            sig_y, sig_x = np.where(results['significance_mask'])
            ax.plot(sig_x, sig_y, 'k.', markersize=10)
            
            # Configure axes
            ax.set_xticks(np.arange(len(amp_bands)))
            ax.set_yticks(np.arange(len(phase_bands)))
            ax.set_xticklabels(amp_bands)
            ax.set_yticklabels(phase_bands)
            
            plt.xlabel('Amplitude Frequency Band')
            plt.ylabel('Phase Frequency Band')
            plt.title('Cross-Frequency Phase-Amplitude Coupling')
            
            # Add colorbar
            plt.colorbar(im, label='Modulation Index')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            print(f"Error in plot_comodulogram: {str(e)}")
            return None
        
    def detect_ieds(self, detection_threshold=8, artifact_threshold=20, window_size=1.0, 
              min_prominence=None, min_distance_ms=250, return_waveforms=True,
              spatial_analysis=True, target_fs=400, downsample=True):
        """
        Detect Interictal Epileptiform Discharges (IEDs) based on Elliot Smith's approach.
        
        This method assumes the data has already been filtered in the 20-40 Hz range
        and is available in the waves dictionary under the key 'IED'.
        
        Parameters:
        -----------
        detection_threshold : float
            Detection threshold as a multiple of the median absolute deviation (MAD),
            similar to Elliot Smith's SNR threshold. Default: 8
        artifact_threshold : float
            Threshold for artifact rejection as a multiple of MAD. Default: 20
        window_size : float
            Size of the window to extract around each detection in seconds. Default: 1.0
        min_prominence : float or None
            Minimum prominence for peak detection. If None, it's set based on the MAD.
        min_distance_ms : float
            Minimum distance between detections in milliseconds. Default: 250
        return_waveforms : bool
            Whether to return the waveforms of detected IEDs. Default: True
        spatial_analysis : bool
            Whether to perform spatial analysis to calculate propagation speed/direction. Default: True
        target_fs : int
            Target sampling frequency for downsampling. Default: 400 Hz (matching Smith's approach)
        downsample : bool
            Whether to downsample the data to target_fs before detection. Default: True
        
        Returns:
        --------
        ied_data : dict
            Dictionary containing the detected IEDs with the following keys:
            - 'detections': list of dict for each channel with detected IEDs
            - 'parameters': processing parameters
            - 'measurements': signal statistics by channel
            - 'found_peaks': peak information by channel
            If return_waveforms is True:
            - 'full_waveforms': extracted windows around each IED (all channels)
            If spatial_analysis is True:
            - 'spatial_properties': propagation speed, direction, and other spatial metrics
        """
        import numpy as np
        from scipy import signal
        from scipy.ndimage import gaussian_filter
        
        print(f"Starting IED detection with threshold: {detection_threshold}")
        
        # Check if IED band data exists
        if 'narrowRipples' not in self.waves:
            raise ValueError("IED band data not found in waves dictionary. Make sure to filter data in 20-40 Hz range.")
        
        # Get sampling rate and IED data
        fs = self.fs  # Access fs directly from self.fs instead of self.waves
        ied_data = self.waves['narrowRipples']
        n_channels, n_samples = ied_data.shape
        
        print(f"Processing {n_channels} channels with {n_samples} samples at {fs} Hz")
        
        # Downsample if requested (matching Smith's approach)
        if downsample and fs > target_fs:
            print(f"Downsampling from {fs} Hz to {target_fs} Hz...")
            # Calculate downsampling factor
            downsample_factor = int(fs / target_fs)
            
            # Use scipy's resample to downsample
            downsampled_data = np.zeros((n_channels, n_samples // downsample_factor))
            for ch in range(n_channels):
                downsampled_data[ch] = signal.resample(ied_data[ch], n_samples // downsample_factor)
                
            # Update variables to use downsampled data
            ied_data = downsampled_data
            fs = target_fs
            n_samples = ied_data.shape[1]
            print(f"Downsampled to {n_samples} samples at {fs} Hz")
        
        # Initialize output structure
        result = {
            'parameters': {
                'detection_threshold': detection_threshold,
                'artifact_threshold': artifact_threshold,
                'window_size': window_size,
                'sampling_rate': fs,
                'min_distance_ms': min_distance_ms,
                'original_fs': self.fs,
                'downsampled': downsample,
                'target_fs': target_fs if downsample else None
            },
            'measurements': [],
            'found_peaks': [],
            'detections': []
        }
        
        # Calculate time windows in samples
        window_samples = int(window_size * fs)
        min_distance_samples = int(min_distance_ms * fs / 1000)
        half_window = window_samples // 2
        
        # Initialize for all channels
        for ch in range(n_channels):
            result['measurements'].append({})
            result['found_peaks'].append({})
            result['detections'].append({'times': [], 'amplitudes': []})
        
        # Create windows matrix for found IEDs if required
        if return_waveforms:
            result['full_waveforms'] = []
        
        # Process each channel
        print("Processing channels...")
        all_detection_times = []
        
        for ch in range(n_channels):
            # Get data for this channel and smooth it
            signal_ch = ied_data[ch]
            
            # Calculate statistics for this channel
            # Using MAD for robust statistics, similar to Smith's approach
            mad = np.median(np.abs(signal_ch - np.median(signal_ch))) / 0.6745  # Scale factor for Gaussian equivalent
            snr_threshold = detection_threshold * mad
            artifact_thresh = artifact_threshold * mad
            
            # Store measurements
            result['measurements'][ch]['SNR'] = snr_threshold
            result['measurements'][ch]['ArtifactThresh'] = artifact_thresh
            
            # Apply smoothing (similar to Smith's 'smooth' function)
            smoothed_signal = gaussian_filter(np.abs(signal_ch), fs/50)
            
            # Set prominence if not provided
            if min_prominence is None:
                prominence = mad
            else:
                prominence = min_prominence
            
            # Find peaks in the smoothed signal (Smith looks for peaks in the absolute value)
            peaks, properties = signal.find_peaks(
                smoothed_signal, 
                height=snr_threshold,
                distance=min_distance_samples,
                prominence=prominence
            )
            
            # Store peak information
            result['found_peaks'][ch]['peaks'] = properties['peak_heights']
            result['found_peaks'][ch]['locs'] = peaks
            result['found_peaks'][ch]['prominences'] = properties.get('prominences', np.zeros_like(peaks))
            
            # Filter out artifacts and store valid detection times
            valid_peaks = properties['peak_heights'] < artifact_thresh
            valid_locs = peaks[valid_peaks]
            
            # Store valid detection times
            result['detections'][ch]['times'] = valid_locs
            result['detections'][ch]['amplitudes'] = properties['peak_heights'][valid_peaks]
            
            # Collect all detection times for overall analysis
            all_detection_times.extend(valid_locs)
            
            print(f"Channel {ch}: Found {len(peaks)} peaks, {len(valid_locs)} valid after artifact rejection")
        
        # Sort all detection times
        all_detection_times = np.sort(all_detection_times)
        
        # Identify co-occurring IEDs across channels (Smith's "retained detections")
        print("Identifying co-occurring IEDs across channels...")
        bin_size = fs // 2  # 0.5 second bins
        n_bins = int(np.ceil(n_samples / bin_size))
        
        # Create histogram of detections
        detection_histogram, bin_edges = np.histogram(all_detection_times, bins=n_bins)
        
        # Find bins with significant multi-channel activity
        multi_channel_threshold = detection_threshold // 2  # Lower threshold for multi-channel events
        significant_bins = np.where(detection_histogram > multi_channel_threshold)[0]
        
        # Extract time windows for significant bins
        significant_start_times = bin_edges[significant_bins]
        significant_end_times = bin_edges[significant_bins + 1]
        
        # Find all detections that fall within significant bins
        retained_detection_indices = {}
        for i, (start, end) in enumerate(zip(significant_start_times, significant_end_times)):
            retained_detection_indices[i] = np.where((all_detection_times >= start) & 
                                                    (all_detection_times <= end))[0]
        
        # Store number of significant IED event clusters
        result['n_significant_events'] = len(significant_bins)
        print(f"Found {len(significant_bins)} significant multi-channel IED events")
        
        # Extract waveforms for each significant event if requested
        if return_waveforms:
            result['full_waveforms'] = []
            
            for event_idx, detection_indices in retained_detection_indices.items():
                if len(detection_indices) == 0:
                    continue
                    
                # Use median time as central point for window extraction
                central_time = np.median(all_detection_times[detection_indices])
                
                # Calculate window boundaries
                start_sample = int(max(0, central_time - half_window))
                end_sample = int(min(n_samples, central_time + half_window))
                
                # Extract waveforms for all channels
                event_window = {
                    'central_time': central_time,
                    'time_samples': np.arange(start_sample, end_sample),
                    'waveforms': ied_data[:, start_sample:end_sample].copy(),
                    'channels_with_detections': []
                }
                
                # Track which channels had detections in this window
                for ch in range(n_channels):
                    ch_times = result['detections'][ch]['times']
                    in_window = np.where((ch_times >= start_sample) & (ch_times <= end_sample))[0]
                    if len(in_window) > 0:
                        event_window['channels_with_detections'].append(ch)
                
                result['full_waveforms'].append(event_window)
        
        # Perform spatial analysis for significant events
        if spatial_analysis and len(retained_detection_indices) > 0:
            result['spatial_properties'] = self._analyze_ied_propagation(result, retained_detection_indices, all_detection_times)
        
        return result

    def _analyze_ied_propagation(self, ied_result, event_indices, all_detection_times):
        """
        Analyze the propagation patterns of detected IEDs.
        
        This method implements Smith's approach to analyzing the spatial patterns 
        of IED propagation, including calculating propagation speed and direction.
        
        Parameters:
        -----------
        ied_result : dict
            Result dictionary from detect_ieds
        event_indices : dict
            Dictionary mapping event indices to detection indices
        all_detection_times : array
            Array of all detection times
        
        Returns:
        --------
        spatial_properties : dict
            Dictionary containing spatial analysis results for each significant event
        """
        import numpy as np
        from scipy.spatial import distance
        from scipy.ndimage import gaussian_filter
        
        # Get electrode locations
        locations = self.data_df[['x', 'y']].values  # Original locations in cm
        
        # Initialize results
        spatial_properties = {'events': []}
        
        # Process each significant event
        for event_idx, detection_indices in event_indices.items():
            if len(detection_indices) < 3:  # Need at least 3 points for meaningful analysis
                continue
                
            # Get detection times for this event
            event_times = all_detection_times[detection_indices]
            
            # Find which channels had detections at these times
            event_channels = []
            detection_times_by_channel = {}
            
            for ch in range(len(ied_result['detections'])):
                ch_times = ied_result['detections'][ch]['times']
                in_event = np.isin(ch_times, event_times)
                if np.any(in_event):
                    event_channels.append(ch)
                    detection_times_by_channel[ch] = ch_times[in_event]
            
            # Skip if too few channels involved
            if len(event_channels) < 3:
                continue
                
            # Get median detection time for normalization
            median_time = np.median(event_times)
            
            # Create matrices for spatial linear regression (similar to Smith's SpatialLinearRegression)
            X = []  # Positions
            Y = []  # Times
            
            for ch in event_channels:
                times = detection_times_by_channel[ch]
                for t in times:
                    X.append(locations[ch])
                    Y.append(t - median_time)  # Center times around median
            
            X = np.array(X)
            Y = np.array(Y)
            
            # Skip if not enough points for regression
            if len(X) < 3:
                continue
            
            # Perform multivariate linear regression (simplified version of Smith's approach)
            # X: electrode positions (x, y)
            # Y: detection times
            # We want to solve Y = X * beta + intercept
            
            # Add intercept term
            X_with_intercept = np.column_stack([X, np.ones(X.shape[0])])
            
            try:
                # Check if we have enough data points for reliable regression
                if X_with_intercept.shape[0] <= X_with_intercept.shape[1]:
                    print(f"Warning: Underdetermined system for event {event_idx} - " 
                        f"{X_with_intercept.shape[0]} data points < {X_with_intercept.shape[1]} variables")
                    continue
                    
                # Solve using least squares
                beta, residuals, rank, s = np.linalg.lstsq(X_with_intercept, Y, rcond=None)
                
                # Print debugging information
                print(f"Event {event_idx} linear model: shape of X={X.shape}, Y={Y.shape}, beta={beta.shape}")
                print(f"Beta values: {beta}")
                
                # Calculate propagation velocity (similar to Smith's pinv(IWbetas))
                # First check if beta has enough elements
                if len(beta) >= 2 and (abs(beta[0]) > 1e-10 or abs(beta[1]) > 1e-10):  # Avoid division by zero
                    try:
                        # Use pseudo-inverse to get velocity vector
                        velocity = np.linalg.pinv(beta[:2].reshape(1, 2))[0]
                        
                        # Make sure velocity has at least 2 elements
                        if len(velocity) >= 2:
                            speed = np.sqrt(np.sum(velocity**2))
                            direction = np.arctan2(velocity[1], velocity[0])
                        else:
                            print(f"Warning: velocity vector has wrong shape: {velocity.shape}")
                            speed = np.linalg.norm(velocity)
                            direction = 0.0  # Default direction
                            
                        # Calculate p-value (simplified)
                        # In Smith's code, this is done with F-test or permutation
                        p_value = 0.05  # Placeholder
                        
                        if residuals.size > 0:
                            r_squared = 1 - residuals[0] / np.sum((Y - np.mean(Y))**2)
                        else:
                            r_squared = 0
                    except Exception as e:
                        print(f"Error calculating velocity: {e}")
                        # Provide default values
                        velocity = np.array([0.0, 0.0])
                        speed = 0.0
                        direction = 0.0
                        p_value = 1.0
                        r_squared = 0.0
                    
                    event_properties = {
                        'event_idx': event_idx,
                        'median_time': median_time,
                        'channels': event_channels,
                        'n_channels': len(event_channels),
                        'beta': beta.tolist(),
                        'velocity': velocity.tolist(),
                        'speed': float(speed),
                        'direction': float(direction),
                        'p_value': p_value,
                        'r_squared': float(r_squared)
                    }
                    
                    spatial_properties['events'].append(event_properties)
                
            except np.linalg.LinAlgError:
                # Skip events where linear regression fails
                continue
        
        print(f"Completed spatial analysis for {len(spatial_properties['events'])} events")
        return spatial_properties

    def plot_detected_ieds(self, ied_result, max_events=5, figsize=(15, 10), save_path=None):
        """
        Plot detected IEDs with their spatial properties.
        
        Parameters:
        -----------
        ied_result : dict
            Result dictionary from detect_ieds method
        max_events : int
            Maximum number of events to plot. Default: 5
        figsize : tuple
            Figure size (width, height). Default: (15, 10)
        save_path : str or None
            Path to save the figure. If None, figure is displayed only.
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec
        import matplotlib.colors as mcolors
        
        # Check if we have waveforms to plot
        if 'full_waveforms' not in ied_result or not ied_result['full_waveforms']:
            raise ValueError("No waveforms available in IED result to plot")
        
        # Get sampling rate
        fs = ied_result['parameters']['sampling_rate']
        
        # Select events to plot (up to max_events)
        n_events = min(max_events, len(ied_result['full_waveforms']))
        events_to_plot = sorted(range(len(ied_result['full_waveforms'])), 
                            key=lambda i: len(ied_result['full_waveforms'][i]['channels_with_detections']),
                            reverse=True)[:n_events]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        for i, event_idx in enumerate(events_to_plot):
            event = ied_result['full_waveforms'][event_idx]
            
            # Find spatial properties for this event
            event_spatial = None
            if 'spatial_properties' in ied_result:
                for sp in ied_result['spatial_properties']['events']:
                    if sp['median_time'] == event['central_time']:
                        event_spatial = sp
                        break
            
            # Create subplot grid for this event
            gs = GridSpec(3, 3, figure=fig, left=0.05 + (i % 2) * 0.5, 
                        right=0.45 + (i % 2) * 0.5,
                        bottom=0.7 - (i // 2) * 0.35, 
                        top=0.95 - (i // 2) * 0.35,
                        wspace=0.3, hspace=0.3)
            
            # 1. Plot average waveform
            ax1 = fig.add_subplot(gs[0, 0])
            time_sec = event['time_samples'] / fs
            mean_waveform = np.mean(event['waveforms'], axis=0)
            ax1.plot(time_sec, mean_waveform, 'k-')
            
            # Highlight detection point
            central_idx = np.argmin(np.abs(event['time_samples'] - event['central_time']))
            ax1.axvline(time_sec[central_idx], color='r', linestyle='--')
            
            ax1.set_title(f"Event {event_idx+1}: Mean Waveform")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            
            # 2. Plot multi-channel waveforms
            ax2 = fig.add_subplot(gs[0, 1:])
            channels_with_detections = event['channels_with_detections']
            
            # Choose a subset of channels if too many
            if len(channels_with_detections) > 10:
                plot_channels = np.random.choice(channels_with_detections, 10, replace=False)
            else:
                plot_channels = channels_with_detections
                
            cmap = plt.cm.viridis
            colors = [cmap(i/len(plot_channels)) for i in range(len(plot_channels))]
            
            for j, ch in enumerate(plot_channels):
                ax2.plot(time_sec, event['waveforms'][ch], color=colors[j], 
                        label=f"Ch {ch}", alpha=0.7)
                
            ax2.axvline(time_sec[central_idx], color='r', linestyle='--')
            ax2.set_title("Multi-channel Waveforms")
            ax2.set_xlabel("Time (s)")
            
            # Add legend if not too many channels
            if len(plot_channels) <= 5:
                ax2.legend(loc='upper right', fontsize=8)
            
            # 3. Plot spatial map of electrodes with detections
            ax3 = fig.add_subplot(gs[1:, 0])
            
            # Get electrode positions
            x_pos = self.data_df['x_mod'].values
            y_pos = self.data_df['y_mod'].values
            
            # Plot all electrodes as small gray dots
            ax3.scatter(x_pos, y_pos, s=20, c='gray', alpha=0.3)
            
            # Highlight electrodes with detections
            detection_colors = []
            for ch in channels_with_detections:
                ch_times = ied_result['detections'][ch]['times']
                in_window = np.where((ch_times >= event['time_samples'][0]) & 
                                    (ch_times <= event['time_samples'][-1]))[0]
                if len(in_window) > 0:
                    detection_time = ch_times[in_window[0]] - event['central_time']
                    detection_colors.append(detection_time)
                    
            if channels_with_detections:
                scatter = ax3.scatter(x_pos[channels_with_detections], y_pos[channels_with_detections], 
                                s=80, c=detection_colors, cmap='viridis', 
                                alpha=0.8, edgecolor='k')
                cbar = plt.colorbar(scatter, ax=ax3)
                cbar.set_label('Detection time relative to event (samples)')
            
            ax3.set_title("Spatial Map of Detections")
            ax3.set_xlabel("X position")
            ax3.set_ylabel("Y position")
            ax3.set_aspect('equal')
            
            # 4. Plot propagation vector if available
            if event_spatial:
                # Draw propagation vector
                vector_scale = 15
                center_x = np.mean(x_pos[channels_with_detections])
                center_y = np.mean(y_pos[channels_with_detections])
                velocity = np.array(event_spatial['velocity'])
                
                # Normalize and scale velocity
                vel_norm = velocity / np.linalg.norm(velocity) * vector_scale
                
                ax3.arrow(center_x, center_y, vel_norm[0], vel_norm[1], 
                        head_width=2, head_length=3, fc='r', ec='r', 
                        length_includes_head=True)
                
                # Add speed and direction information
                info_text = (f"Speed: {event_spatial['speed']:.2f} cm/s\n"
                        f"Direction: {np.degrees(event_spatial['direction']):.1f}°\n"
                        f"R²: {event_spatial['r_squared']:.2f}")
                
                ax3.text(0.05, 0.05, info_text, transform=ax3.transAxes,
                    fontsize=8, verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            # 5. Plot timing histogram
            ax4 = fig.add_subplot(gs[1, 1:])
            
            detection_times = []
            for ch in channels_with_detections:
                ch_times = ied_result['detections'][ch]['times']
                in_window = np.where((ch_times >= event['time_samples'][0]) & 
                                    (ch_times <= event['time_samples'][-1]))[0]
                if len(in_window) > 0:
                    detection_times.extend(ch_times[in_window] - event['central_time'])
            
            if detection_times:
                ax4.hist(detection_times, bins=20, color='steelblue', alpha=0.7)
                ax4.axvline(0, color='r', linestyle='--')
                ax4.set_title("Detection Time Distribution")
                ax4.set_xlabel("Time relative to event center (samples)")
                ax4.set_ylabel("Count")
            
            # 6. Plot additional statistics
            ax5 = fig.add_subplot(gs[2, 1:])
            ax5.axis('off')
            
            # Event summary text
            summary_text = (
                f"Event {event_idx+1} Summary:\n"
                f"Time: {event['central_time']/fs:.3f} s\n"
                f"Channels with detections: {len(channels_with_detections)}\n"
            )
            
            if event_spatial:
                summary_text += (
                    f"Propagation speed: {event_spatial['speed']:.2f} cm/s\n"
                    f"Propagation direction: {np.degrees(event_spatial['direction']):.1f}°\n"
                    f"Model p-value: {event_spatial['p_value']:.3f}\n"
                    f"R-squared: {event_spatial['r_squared']:.3f}\n"
                )
            
            ax5.text(0.0, 1.0, summary_text, fontsize=9, verticalalignment='top')
        
        plt.suptitle(f"Detected Interictal Epileptiform Discharges (IEDs)", fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def detect_ripples(self, narrowband_key='narrowRipple', wideband_key='wideRipple', 
                low_threshold=2.0, high_threshold=5.0, 
                min_duration=20, max_duration=100, min_interval=30,
                    window_length=11, restrict=None, time_start=None, time_end=None):
        """
        Detect ripples in LFP signals using Buzsaki's approach.
        
        Parameters:
        -----------
        narrowband_key : str
            Key for the narrowband filtered signal (50-300 Hz) in waves dictionary
        wideband_key : str
            Key for the wideband filtered signal (130-200 Hz) in waves dictionary
        low_threshold : float
            Threshold for ripple beginning/end in standard deviations
        high_threshold : float
            Threshold for ripple peak in standard deviations
        min_duration : float
            Minimum ripple duration in ms
        max_duration : float
            Maximum ripple duration in ms
        min_interval : float
            Minimum inter-ripple interval in ms
        window_length : int
            Window length for smoothing in samples
        restrict : array-like, optional
            Time intervals to restrict the analysis
        time_start : float, optional
            Start time for analysis in seconds. If None, starts from the beginning
        time_end : float, optional
            End time for analysis in seconds. If None, analyzes until the end
            
        Returns:
        --------
        dict
            Dictionary containing ripple events for each channel
        """
        import numpy as np
        from scipy.signal import filtfilt
        import pandas as pd
        
        # Convert durations to samples
        min_duration_samples = int(min_duration / 1000 * self.fs)
        max_duration_samples = int(max_duration / 1000 * self.fs)
        min_interval_samples = int(min_interval / 1000 * self.fs)
        
        # Get filtered signals
        if narrowband_key not in self.waves or wideband_key not in self.waves:
            raise KeyError(f"Keys {narrowband_key} or {wideband_key} not found in waves dictionary")
        
        # Initialize result dictionary to store ripples for each channel
        ripples = {}
        
        # Process each channel separately
        n_channels = self.waves[wideband_key].shape[0]
        total_samples = self.waves[wideband_key].shape[1]
        
        # Convert time window to sample indices
        start_idx = 0 if time_start is None else int(time_start * self.fs)
        end_idx = total_samples if time_end is None else int(time_end * self.fs)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(total_samples, end_idx)
        
        # Create time vector for the selected window
        time = np.arange(start_idx, end_idx) / self.fs
        
        for ch in range(n_channels):
            # Get filtered signal for this channel (sliced to time window)
            wideband_signal = self.waves[wideband_key][ch, start_idx:end_idx]
            narrowband_signal = self.waves[narrowband_key][ch, start_idx:end_idx]
            
            # Square the signal
            squared_signal = wideband_signal ** 2
            
            # Smooth the squared signal (using a moving average filter)
            window = np.ones(window_length) / window_length
            smoothed_squared_signal = np.convolve(squared_signal, window, mode='same')
            
            # Determine which samples to use for normalization
            if restrict is not None:
                keep = np.zeros(len(time), dtype=bool)
                for start, end in restrict:
                    keep = np.logical_or(keep, (time >= start) & (time <= end))
            else:
                keep = np.ones(len(time), dtype=bool)
            
            # Normalize the signal to z-scores
            mean = np.mean(smoothed_squared_signal[keep])
            std = np.std(smoothed_squared_signal[keep])
            normalized_signal = (smoothed_squared_signal - mean) / std
            
            # First pass: Detect periods above low threshold
            thresholded = normalized_signal > low_threshold
            
            # Find transitions (rising and falling edges)
            # Add zeros at start and end to properly detect edges
            padded = np.concatenate(([0], thresholded, [0]))
            rising_edges = np.where(np.diff(padded) > 0)[0]
            falling_edges = np.where(np.diff(padded) < 0)[0] - 1
            
            # Ensure same number of start and end events
            if len(rising_edges) == 0 or len(falling_edges) == 0:
                ripples[ch] = pd.DataFrame(columns=['start_time', 'end_time', 'peak_time', 'peak_normalized_power', 'duration'])
                continue
                
            if rising_edges[0] > falling_edges[0]:
                falling_edges = falling_edges[1:]
            if rising_edges[-1] > falling_edges[-1]:
                rising_edges = rising_edges[:-1]
                
            # Combine into start-stop pairs
            first_pass = np.column_stack((rising_edges, falling_edges))
            
            if len(first_pass) == 0:
                ripples[ch] = pd.DataFrame(columns=['start_time', 'end_time', 'peak_time', 'peak_normalized_power', 'duration'])
                continue
            
            # Second pass: Merge ripples if inter-ripple period is too short
            merged_ripples = []
            current_ripple = first_pass[0]
            
            for i in range(1, len(first_pass)):
                if first_pass[i, 0] - current_ripple[1] < min_interval_samples:
                    # Merge
                    current_ripple[1] = first_pass[i, 1]
                else:
                    merged_ripples.append(current_ripple)
                    current_ripple = first_pass[i]
            
            merged_ripples.append(current_ripple)
            second_pass = np.array(merged_ripples)
            
            # Third pass: Discard ripples with peak power < high_threshold
            third_pass = []
            peak_times = []
            peak_powers = []
            
            for start, end in second_pass:
                ripple_segment = normalized_signal[start:end+1]
                max_index = np.argmax(ripple_segment)
                max_value = ripple_segment[max_index]
                
                if max_value > high_threshold:
                    third_pass.append([start, end])
                    peak_times.append(start + max_index)
                    peak_powers.append(max_value)
            
            if len(third_pass) == 0:
                ripples[ch] = pd.DataFrame(columns=['start_time', 'end_time', 'peak_time', 'peak_normalized_power', 'duration'])
                continue
                
            third_pass = np.array(third_pass)
            peak_times = np.array(peak_times)
            peak_powers = np.array(peak_powers)
            
            # Fourth pass: Discard ripples that are too short or too long
            durations = third_pass[:, 1] - third_pass[:, 0]
            valid_duration = (durations >= min_duration_samples) & (durations <= max_duration_samples)
            
            final_starts = third_pass[valid_duration, 0]
            final_ends = third_pass[valid_duration, 1]
            final_peaks = peak_times[valid_duration]
            final_powers = peak_powers[valid_duration]
            final_durations = durations[valid_duration]
            
            # Convert to time (seconds)
            start_times = time[final_starts]
            end_times = time[final_ends]
            peak_times = time[final_peaks]
            
            # Create DataFrame with results
            ripple_data = pd.DataFrame({
                'start_time': start_times,
                'end_time': end_times,
                'peak_time': peak_times,
                'peak_normalized_power': final_powers,
                'duration': final_durations / self.fs  # in seconds
            })
            
            # Store in the results dictionary
            ripples[ch] = ripple_data
        
        # Add metadata to the ripples dictionary
        ripples['metadata'] = {
            'fs': self.fs,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'min_interval': min_interval,
            'window_length': window_length,
            'narrowband_key': narrowband_key,
            'wideband_key': wideband_key,
            'time_start': time_start if time_start is not None else 0,
            'time_end': time_end if time_end is not None else total_samples / self.fs,
            'sample_start': start_idx,
            'sample_end': end_idx
        }
        
        return ripples

    def visualize_ripples(self, ripples, channel, n_ripples=10, window=0.2, figsize=(15, 10),
                            sort_by='peak_normalized_power'):
        """
        Visualize detected ripples from a specific channel.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        channel : int
            Channel to visualize
        n_ripples : int
            Number of ripples to visualize (if None, all ripples are shown)
        window : float
            Time window around each ripple in seconds
        figsize : tuple
            Figure size
        sort_by : str
            Field to sort ripples by for visualization (e.g., 'peak_normalized_power', 'duration')
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if channel not in ripples:
            raise ValueError(f"Channel {channel} not found in ripples dictionary")
        
        # Get ripple data for this channel
        ripple_df = ripples[channel]
        
        if len(ripple_df) == 0:
            print(f"No ripples detected on channel {channel}")
            return
        
        # Get metadata
        metadata = ripples['metadata']
        fs = metadata['fs']
        narrowband_key = metadata['narrowband_key']
        wideband_key = metadata['wideband_key']
        
        # Get the signals
        narrowband_signal = self.waves[narrowband_key][channel]
        wideband_signal = self.waves[wideband_key][channel]
        raw_signal = self.waves['lfp'][channel] if 'lfp' in self.waves else None
        
        # Create time vector
        time = np.arange(len(wideband_signal)) / fs
        
        # Limit the number of ripples to visualize
        if n_ripples is not None and n_ripples < len(ripple_df):
            # Sort by specified field (default: peak_normalized_power)
            if sort_by in ripple_df.columns:
                ripple_df = ripple_df.sort_values(sort_by, ascending=False).iloc[:n_ripples]
            else:
                print(f"Warning: sort_by field '{sort_by}' not found. Sorting by peak power.")
                ripple_df = ripple_df.sort_values('peak_normalized_power', ascending=False).iloc[:n_ripples]
        
        # Create the figure
        n_cols = min(5, len(ripple_df))
        n_rows = int(np.ceil(len(ripple_df) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        
        # Plot each ripple
        for i, (idx, ripple) in enumerate(ripple_df.iterrows()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Calculate window boundaries
            start_time = ripple['peak_time'] - window / 2
            end_time = ripple['peak_time'] + window / 2
            
            # Find corresponding indices
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(time) - 1, end_idx)
            
            # Extract signal segments
            t_segment = time[start_idx:end_idx]
            wide_segment = wideband_signal[start_idx:end_idx]
            narrow_segment = narrowband_signal[start_idx:end_idx]
            
            # Plot signals
            if raw_signal is not None:
                raw_segment = raw_signal[start_idx:end_idx]
                ax.plot(t_segment, raw_segment, 'k-', alpha=0.5, label='Raw LFP')
            
            ax.plot(t_segment, narrow_segment, 'b-', label='Narrow-band')
            ax.plot(t_segment, wide_segment, 'r-', label='Wide-band')
            
            # Mark ripple boundaries
            ax.axvline(ripple['start_time'], color='g', linestyle='--', label='Start')
            ax.axvline(ripple['peak_time'], color='m', linestyle='-', label='Peak')
            ax.axvline(ripple['end_time'], color='r', linestyle='--', label='End')
            
            # Set title with ripple info
            duration_ms = ripple['duration'] * 1000
            ax.set_title(f"Ripple {idx}\nPower: {ripple['peak_normalized_power']:.2f}\nDuration: {duration_ms:.1f} ms")
            
            # Only add legend to the first subplot
            if i == 0:
                ax.legend(fontsize='small')
            
            # Set x-axis limits
            ax.set_xlim(start_time, end_time)
            
            # Only add x-label to bottom row
            if row == n_rows - 1:
                ax.set_xlabel('Time (s)')
            
            # Only add y-label to first column
            if col == 0:
                ax.set_ylabel('Amplitude')
        
        # Remove empty subplots
        for i in range(len(ripple_df), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        # Add metadata about time window
        metadata = ripples['metadata']
        time_str = f" ({metadata['time_start']:.1f}s - {metadata['time_end']:.1f}s)" if 'time_start' in metadata else ""
        
        plt.suptitle(f"Channel {channel} - Detected Ripples{time_str}", fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def detect_ripples(self, narrowband_key='narrowRipples', wideband_key='broadRipples', 
                lfp_key='lfp', sharp_wave_threshold=2.0,
                low_threshold=2.0, high_threshold=5.0, 
                min_duration=20, max_duration=100, min_interval=30,
                window_length=11, restrict=None, time_start=None, time_end=None,
                require_sharp_wave=True, sharp_wave_window=50, sharp_wave_band=(1, 30),
                neuron_file_path=None, use_neuron_neighbors=False):
        """
        Detect ripples in LFP signals using Buzsaki's approach with additional sharp wave verification.
        
        Parameters:
        -----------
        narrowband_key : str
            Key for the narrowband filtered signal (50-300 Hz) in waves dictionary
        wideband_key : str
            Key for the wideband filtered signal (130-200 Hz) in waves dictionary
        lfp_key : str
            Key for the raw LFP signal in waves dictionary, used for sharp wave detection
        sharp_wave_threshold : float
            Threshold for sharp wave detection in standard deviations (negative deflection)
        low_threshold : float
            Threshold for ripple beginning/end in standard deviations
        high_threshold : float
            Threshold for ripple peak in standard deviations
        min_duration : float
            Minimum ripple duration in ms
        max_duration : float
            Maximum ripple duration in ms
        min_interval : float
            Minimum inter-ripple interval in ms
        window_length : int
            Window length for smoothing in samples
        restrict : array-like, optional
            Time intervals to restrict the analysis
        time_start : float, optional
            Start time for analysis in seconds. If None, starts from the beginning
        time_end : float, optional
            End time for analysis in seconds. If None, analyzes until the end
        require_sharp_wave : bool
            Whether to require coincidence with a sharp wave for valid ripple detection
        sharp_wave_window : float
            Time window in ms around ripple to look for sharp wave
        sharp_wave_band : tuple
            Frequency band for sharp wave detection (low, high) in Hz
        neuron_file_path : str, optional
            Path to the neuron data ZIP file (to extract neighbor channels)
        use_neuron_neighbors : bool, optional
            Whether to only analyze channels that are neighbors to neurons
            
        Returns:
        --------
        dict
            Dictionary containing ripple events for each channel
        """
        import numpy as np
        from scipy.signal import filtfilt
        import pandas as pd
        
        # Convert durations to samples
        min_duration_samples = int(min_duration / 1000 * self.fs)
        max_duration_samples = int(max_duration / 1000 * self.fs)
        min_interval_samples = int(min_interval / 1000 * self.fs)
        
        # Get filtered signals
        if narrowband_key not in self.waves or wideband_key not in self.waves:
            raise KeyError(f"Keys {narrowband_key} or {wideband_key} not found in waves dictionary")
        
        if require_sharp_wave and lfp_key not in self.waves:
            raise KeyError(f"Key {lfp_key} not found in waves dictionary but required for sharp wave detection")
        
        # Initialize result dictionary to store ripples for each channel
        ripples = {}
        
        # Process each channel separately
        n_channels = self.waves[wideband_key].shape[0]
        total_samples = self.waves[wideband_key].shape[1]
        
        # Convert time window to sample indices
        start_idx = 0 if time_start is None else int(time_start * self.fs)
        end_idx = total_samples if time_end is None else int(time_end * self.fs)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(total_samples, end_idx)
        
        # Create time vector for the selected window
        time = np.arange(start_idx, end_idx) / self.fs
        
        # Convert sharp wave window from ms to samples
        sharp_wave_window_samples = int(sharp_wave_window / 1000 * self.fs)
        
        # Filter LFP for sharp wave detection if required
        if require_sharp_wave:
            from scipy.signal import butter, filtfilt
            
            # Design bandpass filter for sharp wave detection
            nyquist = self.fs / 2
            low_cut = sharp_wave_band[0] / nyquist
            high_cut = sharp_wave_band[1] / nyquist
            b, a = butter(3, [low_cut, high_cut], btype='band')
            
            # Process all channels together for efficiency
            raw_lfp = self.waves[lfp_key][:, start_idx:end_idx]
            filtered_sw = np.zeros_like(raw_lfp)
            
            for ch in range(n_channels):
                filtered_sw[ch] = filtfilt(b, a, raw_lfp[ch])
        
        # Get channel list to process - either all channels or only neuron neighbors
        channels_to_process = np.arange(n_channels)
        channel_map = {i: i for i in range(n_channels)}  # Default mapping (no change)
        
        # If using neuron neighbors, filter the channel list
        valid_channels = None  # For metadata
        
        if use_neuron_neighbors and neuron_file_path is not None:
            try:
                import zipfile
                
                # Load neuron data
                with zipfile.ZipFile(neuron_file_path, 'r') as f_zip:
                    with f_zip.open("qm.npz") as qm_file:
                        neuron_data = np.load(qm_file, allow_pickle=True)["neuron_data"].item()
                
                # Collect neuron channels and neighboring channels
                neuron_channels = set()
                neighbor_channels_set = set()
                
                for neuron_id, neuron_info in neuron_data.items():
                    # Add primary channel of each neuron
                    neuron_channels.add(int(neuron_info['channel']))
                    
                    # Add all neighbor channels if present
                    if 'neighbor_channels' in neuron_info:
                        for channel in neuron_info['neighbor_channels']:
                            neighbor_channels_set.add(int(channel))
                
                # Map from channel numbers to indices in the waves array
                # First get the channel numbers from the data_df
                config_channels = self.data_df['channel'].values
                
                # Create mapping from channel number to index in waves array
                channel_to_index = {ch: i for i, ch in enumerate(config_channels)}
                
                # Find channels that are neighbors of neurons and exist in our data
                valid_channels = sorted(neighbor_channels_set.intersection(set(config_channels)))
                
                # Convert to channel indices
                valid_indices = [channel_to_index[ch] for ch in valid_channels if ch in channel_to_index]
                
                # Update channels_to_process and channel_map
                channels_to_process = valid_indices
                channel_map = {i: ch for i, ch in enumerate(valid_indices)}
                
                print(f"Using {len(valid_indices)} channels that are neighbors to neurons.")
                
            except Exception as e:
                print(f"Error loading neuron data: {str(e)}. Using all channels.")
                # Continue with all channels if there's an error
        
        # Process each channel
        for i, ch in enumerate(channels_to_process):
            # Get filtered signal for this channel (sliced to time window)
            wideband_signal = self.waves[wideband_key][ch, start_idx:end_idx]
            narrowband_signal = self.waves[narrowband_key][ch, start_idx:end_idx]
            
            # Square the signal
            squared_signal = wideband_signal ** 2
            
            # Smooth the squared signal (using a moving average filter)
            window = np.ones(window_length) / window_length
            smoothed_squared_signal = np.convolve(squared_signal, window, mode='same')
            
            # Determine which samples to use for normalization
            if restrict is not None:
                keep = np.zeros(len(time), dtype=bool)
                for start, end in restrict:
                    keep = np.logical_or(keep, (time >= start) & (time <= end))
            else:
                keep = np.ones(len(time), dtype=bool)
            
            # Normalize the signal to z-scores
            mean = np.mean(smoothed_squared_signal[keep])
            std = np.std(smoothed_squared_signal[keep])
            normalized_signal = (smoothed_squared_signal - mean) / std
            
            # First pass: Detect periods above low threshold
            thresholded = normalized_signal > low_threshold
            
            # Find transitions (rising and falling edges)
            # Add zeros at start and end to properly detect edges
            padded = np.concatenate(([0], thresholded, [0]))
            rising_edges = np.where(np.diff(padded) > 0)[0]
            falling_edges = np.where(np.diff(padded) < 0)[0] - 1
            
            # Ensure same number of start and end events
            if len(rising_edges) == 0 or len(falling_edges) == 0:
                # Use the original channel index as key for consistent mapping
                ripples[channel_map[i]] = pd.DataFrame(columns=['start_time', 'end_time', 'peak_time', 'peak_normalized_power', 'duration', 'has_sharp_wave'])
                continue
                
            if rising_edges[0] > falling_edges[0]:
                falling_edges = falling_edges[1:]
            if rising_edges[-1] > falling_edges[-1]:
                rising_edges = rising_edges[:-1]
                
            # Combine into start-stop pairs
            first_pass = np.column_stack((rising_edges, falling_edges))
            
            if len(first_pass) == 0:
                ripples[channel_map[i]] = pd.DataFrame(columns=['start_time', 'end_time', 'peak_time', 'peak_normalized_power', 'duration', 'has_sharp_wave'])
                continue
            
            # Second pass: Merge ripples if inter-ripple period is too short
            merged_ripples = []
            current_ripple = first_pass[0]
            
            for i in range(1, len(first_pass)):
                if first_pass[i, 0] - current_ripple[1] < min_interval_samples:
                    # Merge
                    current_ripple[1] = first_pass[i, 1]
                else:
                    merged_ripples.append(current_ripple)
                    current_ripple = first_pass[i]
            
            merged_ripples.append(current_ripple)
            second_pass = np.array(merged_ripples)
            
            # Third pass: Discard ripples with peak power < high_threshold
            third_pass = []
            peak_times = []
            peak_powers = []
            
            for start, end in second_pass:
                ripple_segment = normalized_signal[start:end+1]
                max_index = np.argmax(ripple_segment)
                max_value = ripple_segment[max_index]
                
                if max_value > high_threshold:
                    third_pass.append([start, end])
                    peak_times.append(start + max_index)
                    peak_powers.append(max_value)
            
            if len(third_pass) == 0:
                ripples[ch] = pd.DataFrame(columns=['start_time', 'end_time', 'peak_time', 'peak_normalized_power', 'duration', 'has_sharp_wave'])
                continue
                
            third_pass = np.array(third_pass)
            peak_times = np.array(peak_times)
            peak_powers = np.array(peak_powers)
            
            # Fourth pass: Discard ripples that are too short or too long
            durations = third_pass[:, 1] - third_pass[:, 0]
            valid_duration = (durations >= min_duration_samples) & (durations <= max_duration_samples)
            
            final_starts = third_pass[valid_duration, 0]
            final_ends = third_pass[valid_duration, 1]
            final_peaks = peak_times[valid_duration]
            final_powers = peak_powers[valid_duration]
            final_durations = durations[valid_duration]
            
            # Fifth pass (optional): Verify coincidence with sharp waves
            has_sharp_wave = np.ones(len(final_peaks), dtype=bool)  # Default to True
            
            if require_sharp_wave and len(final_peaks) > 0:
                has_sharp_wave = np.zeros(len(final_peaks), dtype=bool)  # Start with False
                
                # Get the sharp wave filtered signal for this channel
                sw_signal = filtered_sw[ch]
                
                # Calculate sharp wave threshold based on negative deflection
                sw_mean = np.mean(sw_signal)
                sw_std = np.std(sw_signal)
                sw_threshold = sw_mean - sharp_wave_threshold * sw_std
                
                # Check each candidate ripple
                for i, peak_idx in enumerate(final_peaks):
                    # Define window around ripple
                    sw_start = max(0, peak_idx - sharp_wave_window_samples)
                    sw_end = min(len(sw_signal) - 1, peak_idx + sharp_wave_window_samples)
                    
                    # Extract signal in window
                    sw_segment = sw_signal[sw_start:sw_end]
                    
                    # Check if any point in the segment exceeds the negative threshold
                    if np.min(sw_segment) < sw_threshold:
                        has_sharp_wave[i] = True
            
            # Apply sharp wave filter if required
            if require_sharp_wave:
                final_starts = final_starts[has_sharp_wave]
                final_ends = final_ends[has_sharp_wave]
                final_peaks = final_peaks[has_sharp_wave]
                final_powers = final_powers[has_sharp_wave]
                final_durations = final_durations[has_sharp_wave]
                
            # Convert to time (seconds)
            start_times = time[final_starts]
            end_times = time[final_ends]
            peak_times = time[final_peaks]
            
            # Create DataFrame with results
            ripple_data = pd.DataFrame({
                'start_time': start_times,
                'end_time': end_times,
                'peak_time': peak_times,
                'peak_normalized_power': final_powers,
                'duration': final_durations / self.fs,  # in seconds
                'has_sharp_wave': has_sharp_wave[has_sharp_wave] if require_sharp_wave else np.ones(len(start_times), dtype=bool)
            })
            
            # Store in the results dictionary using the original channel index
            ripples[channel_map[i]] = ripple_data
        
        # Add metadata to the ripples dictionary
        ripples['metadata'] = {
            'fs': self.fs,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'min_interval': min_interval,
            'window_length': window_length,
            'narrowband_key': narrowband_key,
            'wideband_key': wideband_key,
            'time_start': time_start if time_start is not None else 0,
            'time_end': time_end if time_end is not None else total_samples / self.fs,
            'sample_start': start_idx,
            'sample_end': end_idx,
            'require_sharp_wave': require_sharp_wave,
            'sharp_wave_threshold': sharp_wave_threshold,
            'sharp_wave_window': sharp_wave_window,
            'sharp_wave_band': sharp_wave_band,
            'use_neuron_neighbors': use_neuron_neighbors,
            'valid_channels': valid_channels
        }
        
        return ripples

    def visualize_ripples(self, ripples, channel, n_ripples=10, window=0.2, figsize=(15, 10),
                            sort_by='peak_normalized_power', show_sharp_wave=True):
        """
        Visualize detected ripples from a specific channel.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        channel : int
            Channel to visualize
        n_ripples : int
            Number of ripples to visualize (if None, all ripples are shown)
        window : float
            Time window around each ripple in seconds
        figsize : tuple
            Figure size
        sort_by : str
            Field to sort ripples by for visualization (e.g., 'peak_normalized_power', 'duration')
        show_sharp_wave : bool
            Whether to show sharp wave component if available
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if channel not in ripples:
            raise ValueError(f"Channel {channel} not found in ripples dictionary")
        
        # Get ripple data for this channel
        ripple_df = ripples[channel]
        
        if len(ripple_df) == 0:
            print(f"No ripples detected on channel {channel}")
            return
        
        # Get metadata
        metadata = ripples['metadata']
        fs = metadata['fs']
        narrowband_key = metadata['narrowband_key']
        wideband_key = metadata['wideband_key']
        
        # Get the signals
        narrowband_signal = self.waves[narrowband_key][channel]
        wideband_signal = self.waves[wideband_key][channel]
        raw_signal = self.waves['lfp'][channel] if 'lfp' in self.waves else None
        
        # Get sharp wave filtered signal if available and requested
        show_sw = show_sharp_wave and 'require_sharp_wave' in metadata and metadata['require_sharp_wave']
        sw_signal = None
        
        if show_sw and raw_signal is not None:
            from scipy.signal import butter, filtfilt
            
            # Design bandpass filter for sharp wave detection
            nyquist = fs / 2
            low_cut = metadata['sharp_wave_band'][0] / nyquist
            high_cut = metadata['sharp_wave_band'][1] / nyquist
            b, a = butter(3, [low_cut, high_cut], btype='band')
            
            # Filter the signal
            sw_signal = filtfilt(b, a, raw_signal)
        
        # Create time vector
        time = np.arange(len(wideband_signal)) / fs
        
        # Limit the number of ripples to visualize
        if n_ripples is not None and n_ripples < len(ripple_df):
            # Sort by specified field (default: peak_normalized_power)
            if sort_by in ripple_df.columns:
                ripple_df = ripple_df.sort_values(sort_by, ascending=False).iloc[:n_ripples]
            else:
                print(f"Warning: sort_by field '{sort_by}' not found. Sorting by peak power.")
                ripple_df = ripple_df.sort_values('peak_normalized_power', ascending=False).iloc[:n_ripples]
        
        # Create the figure
        n_cols = min(5, len(ripple_df))
        n_rows = int(np.ceil(len(ripple_df) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        
        # Plot each ripple
        for i, (idx, ripple) in enumerate(ripple_df.iterrows()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Calculate window boundaries
            start_time = ripple['peak_time'] - window / 2
            end_time = ripple['peak_time'] + window / 2
            
            # Find corresponding indices
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(time) - 1, end_idx)
            
            # Extract signal segments
            t_segment = time[start_idx:end_idx]
            wide_segment = wideband_signal[start_idx:end_idx]
            narrow_segment = narrowband_signal[start_idx:end_idx]
            
            # Plot signals
            if raw_signal is not None:
                raw_segment = raw_signal[start_idx:end_idx]
                ax.plot(t_segment, raw_segment, 'k-', alpha=0.5, label='Raw LFP')
            
            ax.plot(t_segment, narrow_segment, 'b-', label='Narrow-band')
            ax.plot(t_segment, wide_segment, 'r-', label='Wide-band')
            
            # Add sharp wave trace if available
            if show_sw and sw_signal is not None:
                sw_segment = sw_signal[start_idx:end_idx]
                ax.plot(t_segment, sw_segment, 'g-', alpha=0.7, linewidth=1.5, label='Sharp Wave')
                
                # Add sharp wave threshold line if available
                if 'sharp_wave_threshold' in metadata:
                    sw_mean = np.mean(sw_signal)
                    sw_std = np.std(sw_signal)
                    sw_threshold = sw_mean - metadata['sharp_wave_threshold'] * sw_std
                    ax.axhline(y=sw_threshold, color='g', linestyle='--', alpha=0.5, label='SW Threshold')
            
            # Mark ripple boundaries
            ax.axvline(ripple['start_time'], color='g', linestyle='--', label='Start')
            ax.axvline(ripple['peak_time'], color='m', linestyle='-', label='Peak')
            ax.axvline(ripple['end_time'], color='r', linestyle='--', label='End')
            
            # Set title with ripple info
            duration_ms = ripple['duration'] * 1000
            ax.set_title(f"Ripple {idx}\nPower: {ripple['peak_normalized_power']:.2f}\nDuration: {duration_ms:.1f} ms")
            
            # Only add legend to the first subplot
            if i == 0:
                ax.legend(fontsize='small')
            
            # Set x-axis limits
            ax.set_xlim(start_time, end_time)
            
            # Only add x-label to bottom row
            if row == n_rows - 1:
                ax.set_xlabel('Time (s)')
            
            # Only add y-label to first column
            if col == 0:
                ax.set_ylabel('Amplitude')
        
        # Remove empty subplots
        for i in range(len(ripple_df), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        # Add metadata about time window
        metadata = ripples['metadata']
        time_str = f" ({metadata['time_start']:.1f}s - {metadata['time_end']:.1f}s)" if 'time_start' in metadata else ""
        
        plt.suptitle(f"Channel {channel} - Detected Ripples{time_str}", fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        return fig

    def plot_ripple_rate_heatmap(self, ripples, figsize=(10, 8), cmap='viridis', 
                            exclude_empty=True, interpolation='bicubic',
                            show_labels=True, show_stats=True):
        """
        Create a heatmap of ripple detection rates across the electrode array,
        using the downsample factor and proper spatial representation.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        figsize : tuple
            Figure size
        cmap : str
            Colormap name
        exclude_empty : bool
            Whether to exclude channels with no ripples from color scaling
        interpolation : str
            Interpolation method for the heatmap display
        show_labels : bool
            Whether to show text labels with channel numbers and rates
        show_stats : bool
            Whether to show detection statistics in the corner
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the heatmap
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Calculate grid dimensions (similar to create_matrix_sequence)
        num_rows = int(self.data_df['y_mod'].max() + 1)
        num_cols = int(self.data_df['x_mod'].max() + 1)
        
        # Calculate downsampled dimensions
        ds_rows = num_rows // self.downsample_factor + 1
        ds_cols = num_cols // self.downsample_factor + 1
        
        # Initialize ripple rate matrix with downsampled dimensions
        ripple_rate = np.zeros((ds_rows, ds_cols), dtype=np.float32)
        count_matrix = np.zeros((ds_rows, ds_cols), dtype=np.int32)  # To track how many electrodes in each bin
        
        # Extract time window from metadata
        metadata = ripples['metadata']
        duration = (metadata['time_end'] - metadata['time_start'])  # in seconds
        
        # Count ripples for each channel
        channels = self.data_df['channel'].values
        for ch_idx, channel in enumerate(channels):
            if ch_idx in ripples and isinstance(ripples[ch_idx], pd.DataFrame):
                n_ripples = len(ripples[ch_idx])
                
                # Get the downsampled coordinates
                x_mod = self.data_df.iloc[ch_idx]['x_mod']
                y_mod = self.data_df.iloc[ch_idx]['y_mod']
                
                # Convert to downsampled coordinates
                x_ds = x_mod // self.downsample_factor
                y_ds = y_mod // self.downsample_factor
                
                # Ensure coordinates are within bounds
                if 0 <= x_ds < ds_cols and 0 <= y_ds < ds_rows:
                    # Add ripple rate (Hz) to the matrix
                    ripple_rate[y_ds, x_ds] += n_ripples / duration if duration > 0 else 0
                    count_matrix[y_ds, x_ds] += 1
        
        # Average by count (for bins with multiple electrodes)
        with np.errstate(divide='ignore', invalid='ignore'):
            ripple_rate = np.divide(ripple_rate, count_matrix, 
                                where=count_matrix > 0, 
                                out=np.zeros_like(ripple_rate))
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set color limits based on non-zero values if requested
        if exclude_empty:
            non_zero = ripple_rate[ripple_rate > 0]
            vmin = np.min(non_zero) if len(non_zero) > 0 else 0
            vmax = np.max(ripple_rate)
        else:
            vmin = 0
            vmax = np.max(ripple_rate)
        
        # Plot the heatmap
        im = ax.imshow(ripple_rate, cmap=cmap, interpolation=interpolation, 
                    origin='lower', vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Ripple Rate (Hz)')
        
        # Set labels and title
        ax.set_xlabel('X-coordinate (downsampled)')
        ax.set_ylabel('Y-coordinate (downsampled)')
        
        # Add title with time window information
        time_str = f" ({metadata['time_start']:.1f}s - {metadata['time_end']:.1f}s)" if 'time_start' in metadata else ""
        ax.set_title(f'Ripple Detection Rate Across Electrode Array{time_str}')
        
        # Add grid for visualization
        ax.grid(False)
        
        # Add coordinate labels for cells with ripples
        for y_ds in range(ds_rows):
            for x_ds in range(ds_cols):
                if ripple_rate[y_ds, x_ds] > 0:
                    # Find channels in this downsampled bin
                    bin_channels = []
                    for ch_idx, channel in enumerate(channels):
                        x_mod = self.data_df.iloc[ch_idx]['x_mod']
                        y_mod = self.data_df.iloc[ch_idx]['y_mod']
                        if (x_mod // self.downsample_factor == x_ds and 
                            y_mod // self.downsample_factor == y_ds):
                            bin_channels.append(channel)
                    
                    # Add text label
                    if bin_channels:
                        if len(bin_channels) == 1:
                            text = f"Ch{bin_channels[0]}\n{ripple_rate[y_ds, x_ds]:.2f}Hz"
                        else:
                            text = f"{len(bin_channels)} chs\n{ripple_rate[y_ds, x_ds]:.2f}Hz"
                        
                        ax.text(x_ds, y_ds, text, 
                            ha="center", va="center", 
                            color="white" if ripple_rate[y_ds, x_ds] > vmax/2 else "black",
                            fontsize=8)
        
        # Add a summary of detection parameters
        summary = (
            f"Detection Parameters:\n"
            f"Low Threshold: {metadata['low_threshold']}σ\n"
            f"High Threshold: {metadata['high_threshold']}σ\n"
            f"Duration: {metadata['min_duration']}-{metadata['max_duration']} ms\n"
            f"Analysis Window: {metadata['time_start']:.1f}s - {metadata['time_end']:.1f}s\n"
            f"Downsample Factor: {self.downsample_factor}"
        )
        plt.figtext(0.02, 0.02, summary, fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        return fig

    def detect_and_visualize_ripples(self, narrowband_key='narrowRipple', wideband_key='wideRipple',
                            lfp_key='lfp', sharp_wave_threshold=2.0,
                            low_threshold=2.0, high_threshold=5.0,
                            min_duration=20, max_duration=100, min_interval=30,
                            window_length=11, restrict=None, time_start=None, time_end=None,
                            require_sharp_wave=True, sharp_wave_window=50, sharp_wave_band=(1, 30),
                            neuron_file_path=None, use_neuron_neighbors=False,
                            channel_to_plot=0, n_ripples=5, window=0.2,
                            plot_heatmap=True, plot_timecourse=False, 
                            show_heatmap_labels=True, figsize=(12, 10)):
        """
        Detect ripples and visualize the results in a single call.
        
        Parameters:
        -----------
        narrowband_key : str
            Key for the narrowband filtered signal (50-300 Hz) in waves dictionary
        wideband_key : str
            Key for the wideband filtered signal (130-200 Hz) in waves dictionary
        lfp_key : str
            Key for the raw LFP signal in waves dictionary, used for sharp wave detection
        sharp_wave_threshold : float
            Threshold for sharp wave detection in standard deviations (negative deflection)
        low_threshold : float
            Threshold for ripple beginning/end in standard deviations
        high_threshold : float
            Threshold for ripple peak in standard deviations
        min_duration : float
            Minimum ripple duration in ms
        max_duration : float
            Maximum ripple duration in ms
        min_interval : float
            Minimum inter-ripple interval in ms
        window_length : int
            Window length for smoothing in samples
        restrict : array-like, optional
            Time intervals to restrict the analysis
        time_start : float, optional
            Start time for analysis in seconds. If None, starts from the beginning
        time_end : float, optional
            End time for analysis in seconds. If None, analyzes until the end
        require_sharp_wave : bool
            Whether to require coincidence with a sharp wave for valid ripple detection
        sharp_wave_window : float
            Time window in ms around ripple to look for sharp wave
        sharp_wave_band : tuple
            Frequency band for sharp wave detection (low, high) in Hz
        neuron_file_path : str, optional
            Path to the neuron data ZIP file (to extract neighbor channels)
        use_neuron_neighbors : bool, optional
            Whether to only analyze channels that are neighbors to neurons
        channel_to_plot : int
            Channel to visualize ripples from
        n_ripples : int
            Number of ripples to visualize
        window : float
            Time window around each ripple in seconds for visualization
        plot_heatmap : bool
            Whether to plot the ripple rate heatmap
        plot_timecourse : bool
            Whether to plot the ripple timecourse
        show_heatmap_labels : bool
            Whether to show labels on the heatmap
        figsize : tuple
            Figure size for plots
        
        Returns:
        --------
        tuple
            (ripples, figures) - Ripple detection results and generated figures
        """
        # Detect ripples
        ripples = self.detect_ripples(
            narrowband_key=narrowband_key,
            wideband_key=wideband_key,
            lfp_key=lfp_key,
            sharp_wave_threshold=sharp_wave_threshold,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            min_duration=min_duration,
            max_duration=max_duration,
            min_interval=min_interval,
            window_length=window_length,
            restrict=restrict,
            time_start=time_start,
            time_end=time_end,
            require_sharp_wave=require_sharp_wave,
            sharp_wave_window=sharp_wave_window,
            sharp_wave_band=sharp_wave_band,
            neuron_file_path=neuron_file_path,
            use_neuron_neighbors=use_neuron_neighbors
        )
        
        # Create figures
        figures = []
        
        # Visualize ripples for specified channel
        if channel_to_plot is not None:
            fig_ripples = self.visualize_ripples(
                ripples, 
                channel=channel_to_plot,
                n_ripples=n_ripples,
                window=window,
                figsize=figsize
            )
            figures.append(fig_ripples)
        
        # Plot ripple rate heatmap
        if plot_heatmap:
            fig_heatmap = self.plot_ripple_rate_heatmap(
                ripples,
                figsize=figsize,
                show_labels=show_heatmap_labels
            )
            figures.append(fig_heatmap)
        
        # Plot ripple timecourse if requested
        if plot_timecourse:
            fig_timecourse = self.plot_average_ripple_rate(
                ripples,
                bin_size_seconds=5,
                smoothing_sigma=2,
                figsize=figsize,
                time_start=time_start,
                time_end=time_end
            )
            figures.append(fig_timecourse)
        
        # Calculate overall statistics
        total_ripples = sum(len(ripples[ch]) for ch in ripples if isinstance(ch, (int, np.integer)))
        n_channels = sum(1 for ch in ripples if isinstance(ch, (int, np.integer)) and len(ripples[ch]) > 0)
        
        if n_channels > 0:
            print(f"Detected {total_ripples} ripples across {n_channels} channels")
            print(f"Average {total_ripples/n_channels:.1f} ripples per active channel")
            
            # Report on sharp wave coincidence if that feature was used
            if 'require_sharp_wave' in ripples['metadata'] and ripples['metadata']['require_sharp_wave']:
                sw_count = 0
                total_count = 0
                for ch in ripples:
                    if isinstance(ch, (int, np.integer)) and len(ripples[ch]) > 0:
                        if 'has_sharp_wave' in ripples[ch].columns:
                            sw_count += ripples[ch]['has_sharp_wave'].sum()
                            total_count += len(ripples[ch])
                
                if total_count > 0:
                    sw_percent = sw_count / total_count * 100
                    print(f"Sharp wave coincidence: {sw_count}/{total_count} ripples ({sw_percent:.1f}%)")
            
            # Report on neuron neighbors if that feature was used
            if 'use_neuron_neighbors' in ripples['metadata'] and ripples['metadata']['use_neuron_neighbors']:
                if 'valid_channels' in ripples['metadata'] and ripples['metadata']['valid_channels'] is not None:
                    valid_channels = ripples['metadata']['valid_channels']
                    print(f"Using {len(valid_channels)} neuron neighbor channels")
        else:
            print("No ripples detected on any channel")
        
        return ripples, figures
    
    def plot_average_ripple_rate(self, ripples, bin_size_seconds=5, smoothing_sigma=None, 
                            figsize=(14, 6), include_percentiles=True, 
                            time_start=None, time_end=None, channel_subset=None):
        """
        Plot the average ripple rate across time, showing the mean rate per electrode.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        bin_size_seconds : float
            Size of time bins in seconds
        smoothing_sigma : float or None
            Standard deviation for Gaussian smoothing (in bins), None for no smoothing
        figsize : tuple
            Figure size
        include_percentiles : bool
            Whether to include 25th and 75th percentiles to show variability
        time_start : float or None
            Start time for plot display in seconds (None to start from beginning)
        time_end : float or None
            End time for plot display in seconds (None to end at recording end)
        channel_subset : list or None
            List of channels to include in the plot (None to use all channels)
                
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the average ripple rate plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        
        # Get metadata
        metadata = ripples['metadata']
        fs = metadata['fs']
        
        # Get recording duration
        if wideband_key := metadata.get('wideband_key'):
            duration = self.waves[wideband_key].shape[1] / fs
        else:
            # If wideband key is not available, try to find the maximum time from ripples
            max_time = 0
            for ch in ripples:
                if isinstance(ch, (int, np.integer)):
                    if len(ripples[ch]) > 0:
                        ch_max = ripples[ch]['end_time'].max()
                        max_time = max(max_time, ch_max)
            duration = max_time + 10  # Add 10 seconds buffer
        
        # Create time bins for the entire recording
        n_bins = int(np.ceil(duration / bin_size_seconds))
        bins = np.linspace(0, n_bins * bin_size_seconds, n_bins + 1)
        bin_centers = bins[:-1] + bin_size_seconds / 2
        
        # Initialize matrix to store ripple counts per channel
        channel_filter = channel_subset if channel_subset is not None else ripples.keys()
        valid_channels = []
        
        # First pass: identify valid channels with ripples
        for ch in ripples:
            if isinstance(ch, (int, np.integer)) and ch in channel_filter:
                ch_ripples = ripples[ch]
                if len(ch_ripples) > 0:
                    valid_channels.append(ch)
        
        # Check if we have any valid channels
        if not valid_channels:
            print("No valid channels with ripples found")
            return None
        
        # Create matrix for ripple counts
        counts_matrix = np.zeros((len(valid_channels), n_bins))
        
        # Second pass: fill the matrix with ripple counts per bin per channel
        for i, ch in enumerate(valid_channels):
            ch_ripples = ripples[ch]
            peak_times = ch_ripples['peak_time'].values
            hist, _ = np.histogram(peak_times, bins=bins)
            counts_matrix[i, :] = hist
        
        # Convert to rate (ripples/second/electrode)
        rate_matrix = counts_matrix / bin_size_seconds
        
        # Calculate statistics across channels
        mean_rate = np.mean(rate_matrix, axis=0)
        percentile_25 = np.percentile(rate_matrix, 25, axis=0)
        percentile_75 = np.percentile(rate_matrix, 75, axis=0)
        
        # Apply smoothing if requested
        if smoothing_sigma is not None:
            mean_rate = gaussian_filter1d(mean_rate, smoothing_sigma)
            percentile_25 = gaussian_filter1d(percentile_25, smoothing_sigma)
            percentile_75 = gaussian_filter1d(percentile_75, smoothing_sigma)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot mean ripple rate
        ax.plot(bin_centers, mean_rate, 'b-', linewidth=2, label='Mean Rate')
        
        # Plot percentiles if requested
        if include_percentiles:
            ax.fill_between(bin_centers, percentile_25, percentile_75, alpha=0.3, color='blue',
                        label='25th-75th Percentile')
        
        # Add labels and set title
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel(f'Ripple Rate (ripples/second/electrode)', fontsize=12)
        
        # Set title based on whether we're viewing a time window or entire recording
        if time_start is not None or time_end is not None:
            title = 'Average Ripple Rate Across Electrodes'
            if time_start is not None and time_end is not None:
                title += f' ({time_start:.1f}s to {time_end:.1f}s)'
            elif time_start is not None:
                title += f' (from {time_start:.1f}s)'
            elif time_end is not None:
                title += f' (until {time_end:.1f}s)'
        else:
            title = 'Average Ripple Rate Across Electrodes (Entire Recording)'
        
        ax.set_title(title, fontsize=14)
        
        # Set x-axis limits (full duration or specified time window)
        if time_start is not None and time_end is not None:
            ax.set_xlim(time_start, time_end)
        elif time_start is not None:
            ax.set_xlim(time_start, duration)
        elif time_end is not None:
            ax.set_xlim(0, time_end)
        else:
            ax.set_xlim(0, duration)
        
        # Add legend
        ax.legend(fontsize=10, loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Calculate overall stats
        overall_mean_rate = np.mean(mean_rate)
        peak_rate = np.max(mean_rate)
        peak_time = bin_centers[np.argmax(mean_rate)]
        
        # Calculate stats for the visible window
        x_min, x_max = ax.get_xlim()
        visible_bin_indices = (bin_centers >= x_min) & (bin_centers <= x_max)
        visible_mean_rate = np.mean(mean_rate[visible_bin_indices]) if np.any(visible_bin_indices) else 0
        
        # Add text with statistics
        stats_text = (
            f"Active channels: {len(valid_channels)}\n"
            f"Overall mean rate: {overall_mean_rate:.3f} ripples/s/electrode\n"
            f"Rate in view: {visible_mean_rate:.3f} ripples/s/electrode\n"
            f"Peak rate: {peak_rate:.3f} ripples/s/electrode at {peak_time:.1f}s\n"
            f"Bin size: {bin_size_seconds}s"
        )
        
        if smoothing_sigma is not None:
            stats_text += f"\nSmoothing: {smoothing_sigma} bins"
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        return fig

    def plot_ripple_time_distribution(self, ripples, bin_size_seconds=5, smoothing_sigma=None, 
                                figsize=(14, 6), include_channels=True, max_channels=10,
                                time_start=None, time_end=None, channel_subset=None):
        """
        Plot the distribution of ripples across time for the entire recording.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        bin_size_seconds : float
            Size of time bins in seconds
        smoothing_sigma : float or None
            Standard deviation for Gaussian smoothing (in bins), None for no smoothing
        figsize : tuple
            Figure size
        include_channels : bool
            Whether to include individual channel ripple counts
        max_channels : int
            Maximum number of channels to show individually
        time_start : float or None
            Start time for plot display in seconds (None to start from beginning)
        time_end : float or None
            End time for plot display in seconds (None to end at recording end)
        channel_subset : list or None
            List of channels to include in the plot (None to use all channels)
                
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the time distribution plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        
        # Get metadata
        metadata = ripples['metadata']
        fs = metadata['fs']
        
        # Get recording duration
        if wideband_key := metadata.get('wideband_key'):
            duration = self.waves[wideband_key].shape[1] / fs
        else:
            # If wideband key is not available, try to find the maximum time from ripples
            max_time = 0
            for ch in ripples:
                if isinstance(ch, (int, np.integer)):
                    if len(ripples[ch]) > 0:
                        ch_max = ripples[ch]['end_time'].max()
                        max_time = max(max_time, ch_max)
            duration = max_time + 10  # Add 10 seconds buffer
        
        # Create time bins for the entire recording
        n_bins = int(np.ceil(duration / bin_size_seconds))
        bins = np.linspace(0, n_bins * bin_size_seconds, n_bins + 1)
        bin_centers = bins[:-1] + bin_size_seconds / 2
        
        # Initialize counts for all channels combined
        total_counts = np.zeros(n_bins)
        
        # Initialize dictionary to store channel-specific counts
        channel_counts = {}
        
        # Filter channels if a subset is specified
        channel_filter = channel_subset if channel_subset is not None else ripples.keys()
        
        # Count ripples per bin for each channel
        valid_channels = []
        
        for ch in ripples:
            if isinstance(ch, (int, np.integer)) and ch in channel_filter:
                ch_ripples = ripples[ch]
                if len(ch_ripples) > 0:
                    valid_channels.append(ch)
                    
                    # Get ripple peak times
                    peak_times = ch_ripples['peak_time'].values
                    
                    # Count ripples per bin
                    hist, _ = np.histogram(peak_times, bins=bins)
                    
                    # Store counts
                    channel_counts[ch] = hist
                    
                    # Add to total
                    total_counts += hist
        
        # Apply smoothing if requested
        if smoothing_sigma is not None:
            for ch in channel_counts:
                channel_counts[ch] = gaussian_filter1d(channel_counts[ch], smoothing_sigma)
            total_counts_smooth = gaussian_filter1d(total_counts, smoothing_sigma)
        else:
            total_counts_smooth = total_counts
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot total ripple count
        ax.plot(bin_centers, total_counts_smooth, 'k-', linewidth=2, label='All Channels')
        ax.fill_between(bin_centers, 0, total_counts_smooth, alpha=0.2, color='gray')
        
        # Plot individual channels (if requested and at least one channel has ripples)
        if include_channels and valid_channels:
            # Sort channels by total ripple count
            channel_totals = {ch: np.sum(channel_counts[ch]) for ch in channel_counts}
            sorted_channels = sorted(channel_totals.keys(), key=lambda x: channel_totals[x], reverse=True)
            
            # Limit the number of channels plotted
            plot_channels = sorted_channels[:max_channels]
            
            # Create a colormap
            import matplotlib.cm as cm
            colors = cm.tab10(np.linspace(0, 1, len(plot_channels)))
            
            # Plot each channel
            for i, ch in enumerate(plot_channels):
                ch_counts = channel_counts[ch]
                if smoothing_sigma is not None:
                    ch_counts = gaussian_filter1d(ch_counts, smoothing_sigma)
                
                ax.plot(bin_centers, ch_counts, color=colors[i], alpha=0.7, 
                    linewidth=1.5, label=f'Channel {ch}')
        
        # Add labels and legend
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel(f'Ripple Count (per {bin_size_seconds}s bin)', fontsize=12)
        
        # Set title based on whether we're viewing a time window or entire recording
        if time_start is not None or time_end is not None:
            title = 'Ripple Occurrence Over Time'
            if time_start is not None and time_end is not None:
                title += f' ({time_start:.1f}s to {time_end:.1f}s)'
            elif time_start is not None:
                title += f' (from {time_start:.1f}s)'
            elif time_end is not None:
                title += f' (until {time_end:.1f}s)'
        else:
            title = 'Ripple Occurrence Over Time (Entire Recording)'
        
        ax.set_title(title, fontsize=14)
        
        # Set x-axis limits (full duration or specified time window)
        if time_start is not None and time_end is not None:
            ax.set_xlim(time_start, time_end)
        elif time_start is not None:
            ax.set_xlim(time_start, duration)
        elif time_end is not None:
            ax.set_xlim(0, time_end)
        else:
            ax.set_xlim(0, duration)
        
        # Add legend with reasonable size
        if include_channels and valid_channels:
            ax.legend(fontsize=8, loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add summary stats
        total_ripples = np.sum(total_counts)
        active_channels = len(valid_channels)
        ripple_rate = total_ripples / duration
        
        # Calculate stats for the visible window
        x_min, x_max = ax.get_xlim()
        visible_bin_indices = (bin_centers >= x_min) & (bin_centers <= x_max)
        visible_ripples = np.sum(total_counts[visible_bin_indices])
        visible_duration = x_max - x_min
        visible_rate = visible_ripples / visible_duration if visible_duration > 0 else 0
        
        stats_text = (
            f"Total ripples: {total_ripples} "
            f"({visible_ripples} in view, {visible_ripples/total_ripples*100:.1f}%)\n"
            f"Active channels: {active_channels}\n"
            f"Overall rate: {ripple_rate:.2f} ripples/second\n"
            f"Rate in view: {visible_rate:.2f} ripples/second\n"
            f"Bin size: {bin_size_seconds}s"
        )
        
        if smoothing_sigma is not None:
            stats_text += f"\nSmoothing: {smoothing_sigma} bins"
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        return fig
    
    def plot_aligned_ripples(self, ripples, channel, window=0.2, min_ripples=5, 
                        use_lfp=True, smooth_sigma=0.001, downsample_factor=1,
                        ripple_alpha=0.15, mean_color='red', mean_linewidth=2.5,
                        figsize=(12, 8), dpi=300, save_path=None,
                        plot_ci=True, ci_level=0.95, title=None):
        """
        Plot all sharp wave ripples from a single channel, aligned at their peaks, 
        with the mean ripple waveform overlaid.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        channel : int
            Channel to visualize
        window : float
            Time window (in seconds) around each ripple peak
        min_ripples : int
            Minimum number of ripples required to create plot
        use_lfp : bool
            Whether to use raw LFP (True) or ripple-band filtered signal (False)
        smooth_sigma : float
            Standard deviation for Gaussian smoothing (in seconds), 0 for no smoothing
        downsample_factor : int
            Factor for downsampling the signal before plotting (1 for no downsampling)
        ripple_alpha : float
            Opacity for individual ripple traces (0-1)
        mean_color : str
            Color for the mean ripple waveform
        mean_linewidth : float
            Line width for the mean ripple waveform
        figsize : tuple
            Figure size
        dpi : int
            DPI for saving figure
        save_path : str or None
            Path to save the figure (None for no saving)
        plot_ci : bool
            Whether to plot confidence intervals around the mean
        ci_level : float
            Confidence interval level (0-1)
        title : str or None
            Custom title for the figure (None for default title)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the aligned ripples plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 9,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8
        })
        
        # Check if channel exists in ripples
        if channel not in ripples:
            raise ValueError(f"Channel {channel} not found in ripples dictionary")
        
        # Get metadata
        metadata = ripples['metadata']
        fs = metadata['fs']
        
        # Get ripples for this channel
        ch_ripples = ripples[channel]
        
        # Check if we have enough ripples
        if len(ch_ripples) < min_ripples:
            print(f"Not enough ripples on channel {channel} ({len(ch_ripples)} found, {min_ripples} required)")
            return None
        
        # Get signal to use (raw LFP or ripple-band)
        signal_key = 'lfp' if use_lfp else metadata['wideband_key']
        if signal_key not in self.waves:
            raise ValueError(f"Signal key '{signal_key}' not found in waves dictionary")
        signal = self.waves[signal_key][channel]
        
        # Calculate samples for window
        half_window_samples = int(window * fs / 2)
        
        # Calculate Gaussian sigma in samples (if smoothing requested)
        smooth_sigma_samples = 0
        if smooth_sigma > 0:
            smooth_sigma_samples = int(smooth_sigma * fs)
        
        # Create time vector for the window
        time = np.arange(-half_window_samples, half_window_samples + 1) / fs
        
        # Store aligned ripples
        aligned_ripples = []
        excluded_count = 0
        
        # Process each ripple
        for _, ripple in ch_ripples.iterrows():
            # Get peak time and convert to sample index
            peak_time = ripple['peak_time']
            peak_idx = int(peak_time * fs)
            
            # Calculate window boundaries
            start_idx = peak_idx - half_window_samples
            end_idx = peak_idx + half_window_samples + 1  # +1 to include the end point
            
            # Check if window is within signal bounds
            if start_idx < 0 or end_idx >= len(signal):
                excluded_count += 1
                continue
            
            # Extract signal segment
            segment = signal[start_idx:end_idx]
            
            # Apply smoothing if requested
            if smooth_sigma_samples > 0:
                segment = gaussian_filter1d(segment, sigma=smooth_sigma_samples)
            
            # Apply downsampling if requested
            if downsample_factor > 1:
                # Ensure consistent length after downsampling
                target_length = len(time)
                original_indices = np.linspace(0, len(segment) - 1, target_length)
                segment = np.interp(original_indices, np.arange(len(segment)), segment)
            
            # Save aligned segment
            aligned_ripples.append(segment)
        
        # Convert to numpy array for easier computation
        aligned_ripples = np.array(aligned_ripples)
        
        # Print info about processed ripples
        included_count = len(aligned_ripples)
        print(f"Processed {included_count} ripples from channel {channel}")
        if excluded_count > 0:
            print(f"Excluded {excluded_count} ripples that didn't fit in the time window")
        
        # Calculate mean ripple
        mean_ripple = np.mean(aligned_ripples, axis=0)
        
        # Calculate confidence intervals if requested
        if plot_ci:
            # Calculate standard error of the mean
            sem = np.std(aligned_ripples, axis=0) / np.sqrt(len(aligned_ripples))
            
            # Calculate z-score for the desired confidence level
            from scipy.stats import norm
            z = norm.ppf((1 + ci_level) / 2)
            
            # Calculate confidence interval
            ci_lower = mean_ripple - z * sem
            ci_upper = mean_ripple + z * sem
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Downsample time vector if needed
        if downsample_factor > 1:
            plot_time = np.linspace(time[0], time[-1], len(mean_ripple))
        else:
            plot_time = time
        
        # Plot individual ripples with transparency
        for ripple in aligned_ripples:
            ax.plot(plot_time, ripple, 'k-', alpha=ripple_alpha, linewidth=0.5)
        
        # Plot confidence interval if requested
        if plot_ci:
            ax.fill_between(plot_time, ci_lower, ci_upper, 
                        color=mean_color, alpha=0.2,
                        label=f'{int(ci_level*100)}% Confidence Interval')
        
        # Plot mean ripple with solid line
        ax.plot(plot_time, mean_ripple, color=mean_color, linewidth=mean_linewidth, 
            label='Mean Ripple')
        
        # Add vertical line at peak (time = 0)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # Set labels
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Amplitude (mV)', fontsize=10)
        
        # Create title
        if title is None:
            signal_label = "LFP" if use_lfp else "Ripple-band"
            title = f"Channel {channel}: {included_count} Aligned {signal_label} Sharp Wave Ripples"
        
        ax.set_title(title, fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=9, loc='upper right')
        
        # Add metrics text box
        ripple_max = np.max(np.abs(mean_ripple))
        metrics_text = (
            f"Channel: {channel}\n"
            f"Ripples: {included_count}\n"
            f"Window: ±{window:.3f}s\n"
            f"Peak Amplitude: {ripple_max:.3f} mV"
        )
        
        if smooth_sigma > 0:
            metrics_text += f"\nSmoothing: {smooth_sigma*1000:.1f} ms"
        
        ax.text(0.02, 0.98, metrics_text, 
            transform=ax.transAxes, 
            va='top', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        return fig

    def plot_multichannel_aligned_ripples(self, ripples, channels=None, min_ripples_per_channel=10,
                                    window=0.2, use_lfp=True, smooth_sigma=0.001,
                                    ripple_alpha=0.15, mean_color='red',
                                    figsize=(16, 10), dpi=300, save_path=None,
                                    plot_ci=True, ci_level=0.95, max_channels=9):
        """
        Generate aligned ripple plots for multiple channels on a single figure.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        channels : list or None
            List of channels to plot (None to auto-select top channels by ripple count)
        min_ripples_per_channel : int
            Minimum number of ripples required for a channel to be included
        window : float
            Time window (in seconds) around each ripple peak
        use_lfp : bool
            Whether to use raw LFP (True) or ripple-band filtered signal (False)
        smooth_sigma : float
            Standard deviation for Gaussian smoothing (in seconds), 0 for no smoothing
        ripple_alpha : float
            Opacity for individual ripple traces (0-1)
        mean_color : str
            Color for the mean ripple waveform
        figsize : tuple
            Figure size
        dpi : int
            DPI for saving figure
        save_path : str or None
            Path to save the figure (None for no saving)
        plot_ci : bool
            Whether to plot confidence intervals around the mean
        ci_level : float
            Confidence interval level (0-1)
        max_channels : int
            Maximum number of channels to plot (will select top channels by ripple count)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the aligned ripples plots
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import math
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 9,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8
        })
        
        # Get active channels with ripples
        channel_counts = {}
        for ch in ripples:
            if isinstance(ch, (int, np.integer)):
                ripple_count = len(ripples[ch])
                if ripple_count >= min_ripples_per_channel:
                    channel_counts[ch] = ripple_count
        
        if not channel_counts:
            print(f"No channels with at least {min_ripples_per_channel} ripples found")
            return None
            
        # If no channels specified, take top channels by ripple count
        if channels is None:
            sorted_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)
            channels = [ch for ch, count in sorted_channels[:max_channels]]
        else:
            # Filter provided channels that don't meet the minimum ripple count
            channels = [ch for ch in channels if ch in channel_counts]
            
            # Limit to max channels
            channels = channels[:max_channels]
        
        if not channels:
            print(f"No channels with sufficient ripples after filtering")
            return None
        
        print(f"Creating aligned ripple plots for {len(channels)} channels")
        
        # Calculate grid dimensions
        n_plots = len(channels)
        n_cols = min(3, n_plots)
        n_rows = math.ceil(n_plots / n_cols)
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                                sharex=True, sharey=True, 
                                squeeze=False)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Process each channel
        for i, channel in enumerate(channels):
            if i >= len(axes_flat):
                break
                
            ax = axes_flat[i]
            
            # Call the single channel function but extract the plotting logic
            ripple_data = self._prepare_aligned_ripples(
                ripples, channel, window=window, use_lfp=use_lfp, 
                smooth_sigma=smooth_sigma
            )
            
            if ripple_data is None:
                # Skip this subplot if data preparation failed
                ax.text(0.5, 0.5, f"Channel {channel}:\nInsufficient data", 
                    ha='center', va='center', transform=ax.transAxes)
                continue
                
            aligned_ripples, mean_ripple, time, included_count, peak_amplitude = ripple_data
            
            # Plot individual ripples with transparency
            for ripple in aligned_ripples:
                ax.plot(time, ripple, 'k-', alpha=ripple_alpha, linewidth=0.5)
            
            # Plot confidence interval if requested
            if plot_ci and len(aligned_ripples) > 1:
                # Calculate standard error of the mean
                sem = np.std(aligned_ripples, axis=0) / np.sqrt(len(aligned_ripples))
                
                # Calculate z-score for the desired confidence level
                from scipy.stats import norm
                z = norm.ppf((1 + ci_level) / 2)
                
                # Calculate confidence interval
                ci_lower = mean_ripple - z * sem
                ci_upper = mean_ripple + z * sem
                
                ax.fill_between(time, ci_lower, ci_upper, 
                            color=mean_color, alpha=0.2)
            
            # Plot mean ripple with solid line
            ax.plot(time, mean_ripple, color=mean_color, linewidth=2)
            
            # Add vertical line at peak (time = 0)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            
            # Set title for this subplot
            signal_label = "LFP" if use_lfp else "Ripple-band"
            ax.set_title(f"Ch {channel}: {included_count} ripples", fontsize=10)
            
            # Add peak amplitude in the corner
            ax.text(0.02, 0.02, f"Peak: {peak_amplitude:.3f} mV", 
                transform=ax.transAxes, fontsize=8, va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(channels), len(axes_flat)):
            fig.delaxes(axes_flat[i])
        
        # Add common axis labels
        fig.text(0.5, 0.01, 'Time (s)', ha='center', fontsize=11)
        fig.text(0.01, 0.5, 'Amplitude (mV)', va='center', rotation='vertical', fontsize=11)
        
        # Add super title
        signal_type = "LFP" if use_lfp else "Ripple-band"
        plt.suptitle(f"Aligned {signal_type} Sharp Wave Ripples Across Channels", 
                fontsize=14, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        return fig

    def _prepare_aligned_ripples(self, ripples, channel, window=0.2, use_lfp=True, smooth_sigma=0.001):
        """
        Helper function to prepare aligned ripple data without plotting.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        channel : int
            Channel to process
        window : float
            Time window (in seconds) around each ripple peak
        use_lfp : bool
            Whether to use raw LFP (True) or ripple-band filtered signal (False)
        smooth_sigma : float
            Standard deviation for Gaussian smoothing (in seconds), 0 for no smoothing
            
        Returns:
        --------
        tuple or None
            (aligned_ripples, mean_ripple, time, included_count, peak_amplitude) or None if processing fails
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        
        # Check if channel exists in ripples
        if channel not in ripples:
            print(f"Channel {channel} not found in ripples dictionary")
            return None
        
        # Get metadata
        metadata = ripples['metadata']
        fs = metadata['fs']
        
        # Get ripples for this channel
        ch_ripples = ripples[channel]
        
        # Check if we have enough ripples
        if len(ch_ripples) < 5:  # Minimum requirement for basic processing
            print(f"Not enough ripples on channel {channel} ({len(ch_ripples)} found)")
            return None
        
        # Get signal to use (raw LFP or ripple-band)
        signal_key = 'lfp' if use_lfp else metadata['wideband_key']
        if signal_key not in self.waves:
            print(f"Signal key '{signal_key}' not found in waves dictionary")
            return None
        signal = self.waves[signal_key][channel]
        
        # Calculate samples for window
        half_window_samples = int(window * fs / 2)
        
        # Calculate Gaussian sigma in samples (if smoothing requested)
        smooth_sigma_samples = 0
        if smooth_sigma > 0:
            smooth_sigma_samples = int(smooth_sigma * fs)
        
        # Create time vector for the window
        time = np.arange(-half_window_samples, half_window_samples + 1) / fs
        
        # Store aligned ripples
        aligned_ripples = []
        
        # Process each ripple
        for _, ripple in ch_ripples.iterrows():
            # Get peak time and convert to sample index
            peak_time = ripple['peak_time']
            peak_idx = int(peak_time * fs)
            
            # Calculate window boundaries
            start_idx = peak_idx - half_window_samples
            end_idx = peak_idx + half_window_samples + 1  # +1 to include the end point
            
            # Check if window is within signal bounds
            if start_idx < 0 or end_idx >= len(signal):
                continue
            
            # Extract signal segment
            segment = signal[start_idx:end_idx]
            
            # Apply smoothing if requested
            if smooth_sigma_samples > 0:
                segment = gaussian_filter1d(segment, sigma=smooth_sigma_samples)
            
            # Save aligned segment
            aligned_ripples.append(segment)
        
        # Convert to numpy array for easier computation
        aligned_ripples = np.array(aligned_ripples)
        
        # Calculate mean ripple
        mean_ripple = np.mean(aligned_ripples, axis=0)
        
        # Calculate peak amplitude
        peak_amplitude = np.max(np.abs(mean_ripple))
        
        # Return prepared data
        return aligned_ripples, mean_ripple, time, len(aligned_ripples), peak_amplitude
    
    def plot_ripple_rate_over_time(self, ripples, bin_size_seconds=5, smoothing_sigma=None, 
                             top_n_channels=3, include_total=True, channel_subset=None,
                             figsize=(12, 8), dpi=300, save_path=None, normalize=False,
                             time_start=None, time_end=None, color_total='black', 
                             channel_colors=None, show_events=False, event_times=None):
        """
        Plot the rate of sharp wave ripples over time, showing the total rate and
        the individual rates for the top channels with the most ripples.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        bin_size_seconds : float
            Size of time bins in seconds
        smoothing_sigma : float or None
            Standard deviation for Gaussian smoothing (in bins), None for no smoothing
        top_n_channels : int
            Number of top channels to display individually
        include_total : bool
            Whether to include the total ripple rate across all channels
        channel_subset : list or None
            List of channels to include (None to use all channels)
        figsize : tuple
            Figure size
        dpi : int
            DPI for saving figure
        save_path : str or None
            Path to save the figure (None for no saving)
        normalize : bool
            Whether to normalize rates to peak values (for comparing patterns)
        time_start : float or None
            Start time for plot display in seconds (None to start from beginning)
        time_end : float or None
            End time for plot display in seconds (None to end at recording end)
        color_total : str
            Color for the total ripple rate line
        channel_colors : list or None
            List of colors for individual channels (None for automatic colors)
        show_events : bool
            Whether to add markers for specific events (e.g., stimulations)
        event_times : list or None
            List of event times in seconds to mark on the plot
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the ripple rate plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        import matplotlib.dates as mdates
        from matplotlib.ticker import MaxNLocator
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8
        })
        
        # Get metadata
        metadata = ripples['metadata']
        fs = metadata['fs']
        
        # Determine recording duration from metadata or max ripple time
        if 'time_end' in metadata and metadata['time_end'] is not None:
            recording_duration = metadata['time_end']
        else:
            # Find max ripple time
            max_time = 0
            for ch in ripples:
                if isinstance(ch, (int, np.integer)) and len(ripples[ch]) > 0:
                    ch_max = ripples[ch]['end_time'].max()
                    max_time = max(max_time, ch_max)
            recording_duration = max_time + 10  # Add buffer
        
        # Adjust time window if specified
        if time_start is None:
            time_start = 0
        if time_end is None:
            time_end = recording_duration
        
        # Create time bins
        n_bins = int(np.ceil((time_end - time_start) / bin_size_seconds))
        bins = np.linspace(time_start, time_start + n_bins * bin_size_seconds, n_bins + 1)
        bin_centers = bins[:-1] + bin_size_seconds / 2
        
        # Collect channel statistics
        channel_stats = {}
        valid_channels = []
        
        # Filter channels if subset is provided
        if channel_subset is not None:
            valid_channel_set = set(channel_subset)
        else:
            valid_channel_set = {ch for ch in ripples if isinstance(ch, (int, np.integer))}
        
        # Count ripples per channel
        for ch in valid_channel_set:
            if ch in ripples and isinstance(ch, (int, np.integer)):
                ripple_count = len(ripples[ch])
                if ripple_count > 0:
                    channel_stats[ch] = {'count': ripple_count}
                    valid_channels.append(ch)
        
        # Sort channels by ripple count (descending)
        top_channels = sorted(valid_channels, key=lambda ch: channel_stats[ch]['count'], reverse=True)
        
        # Limit to top N channels
        if top_n_channels > 0:
            top_channels = top_channels[:min(top_n_channels, len(top_channels))]
        
        # Initialize bins for each channel
        for ch in top_channels:
            channel_stats[ch]['bins'] = np.zeros(n_bins)
            # Sort ripples by time
            ch_ripples = ripples[ch].sort_values('peak_time')
            # Count ripples per bin
            for _, ripple in ch_ripples.iterrows():
                peak_time = ripple['peak_time']
                # Skip ripples outside the time window
                if peak_time < time_start or peak_time >= time_end:
                    continue
                bin_idx = int((peak_time - time_start) / bin_size_seconds)
                if 0 <= bin_idx < n_bins:
                    channel_stats[ch]['bins'][bin_idx] += 1
            
            # Convert to rate (ripples per second)
            channel_stats[ch]['rate'] = channel_stats[ch]['bins'] / bin_size_seconds
            
            # Apply smoothing if requested
            if smoothing_sigma is not None:
                channel_stats[ch]['rate_smooth'] = gaussian_filter1d(
                    channel_stats[ch]['rate'], sigma=smoothing_sigma, mode='reflect')
            else:
                channel_stats[ch]['rate_smooth'] = channel_stats[ch]['rate']
        
        # Calculate total rate across all channels
        total_rate = np.zeros(n_bins)
        
        # Accumulate ripples from all valid channels
        for ch in valid_channels:
            ch_ripples = ripples[ch]
            for _, ripple in ch_ripples.iterrows():
                peak_time = ripple['peak_time']
                # Skip ripples outside the time window
                if peak_time < time_start or peak_time >= time_end:
                    continue
                bin_idx = int((peak_time - time_start) / bin_size_seconds)
                if 0 <= bin_idx < n_bins:
                    total_rate[bin_idx] += 1
        
        # Convert to rate (ripples per second)
        total_rate = total_rate / bin_size_seconds
        
        # Apply smoothing if requested
        if smoothing_sigma is not None:
            total_rate_smooth = gaussian_filter1d(total_rate, sigma=smoothing_sigma, mode='reflect')
        else:
            total_rate_smooth = total_rate
        
        # Normalize if requested
        if normalize:
            # Normalize total rate
            max_total = np.max(total_rate_smooth)
            if max_total > 0:
                total_rate_smooth = total_rate_smooth / max_total
            
            # Normalize individual channel rates
            for ch in top_channels:
                max_rate = np.max(channel_stats[ch]['rate_smooth'])
                if max_rate > 0:
                    channel_stats[ch]['rate_smooth'] = channel_stats[ch]['rate_smooth'] / max_rate
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Default channel colors if not provided
        if channel_colors is None:
            channel_colors = plt.cm.tab10(np.linspace(0, 1, len(top_channels)))
        
        # Plot total rate if requested
        if include_total:
            ax.plot(bin_centers, total_rate_smooth, color=color_total, linewidth=2.5, 
                    label='Total (All Channels)', zorder=10)
            
            # Add light fill under the curve
            ax.fill_between(bin_centers, 0, total_rate_smooth, color=color_total, 
                            alpha=0.1, zorder=5)
        
        # Plot individual channel rates
        for i, ch in enumerate(top_channels):
            color = channel_colors[i % len(channel_colors)]
            ch_rate = channel_stats[ch]['rate_smooth']
            ripple_count = channel_stats[ch]['count']
            
            # Plot line with some transparency
            ax.plot(bin_centers, ch_rate, color=color, linewidth=1.8, alpha=0.85,
                    label=f'Channel {ch} ({ripple_count} ripples)', zorder=9-i)
        
        # Add event markers if requested
        if show_events and event_times is not None:
            y_max = ax.get_ylim()[1]
            for event_time in event_times:
                if time_start <= event_time <= time_end:
                    ax.axvline(x=event_time, color='gray', linestyle='--', alpha=0.5, zorder=1)
                    ax.plot(event_time, y_max*0.05, 'v', color='red', markersize=8, alpha=0.7, zorder=15)
        
        # Set labels
        if normalize:
            ax.set_ylabel('Normalized Ripple Rate', fontsize=11)
        else:
            ax.set_ylabel('Ripple Rate (ripples/second)', fontsize=11)
        
        ax.set_xlabel('Time (seconds)', fontsize=11)
        
        # Set title
        if include_total:
            title = f"Sharp Wave Ripple Rate Over Time"
        else:
            title = f"Sharp Wave Ripple Rate Over Time: Top {len(top_channels)} Channels"
        
        ax.set_title(title, fontsize=12)
        
        # Add time window info to title if specified
        if time_start > 0 or time_end < recording_duration:
            ax.set_title(f"{title}\n(Time window: {time_start:.1f}s - {time_end:.1f}s)", fontsize=12)
        
        # Set x-axis limits
        ax.set_xlim(time_start, time_end)
        
        # Force y-axis to start at 0
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1])
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=9, loc='upper right')
        
        # Add text with statistics
        total_ripples = sum(len(ripples[ch]) for ch in valid_channels)
        active_channels = len(valid_channels)
        
        # Calculate stats for visible window
        window_ripples = 0
        for ch in valid_channels:
            ch_ripples = ripples[ch]
            for _, ripple in ch_ripples.iterrows():
                if time_start <= ripple['peak_time'] < time_end:
                    window_ripples += 1
        
        window_duration = time_end - time_start
        
        if window_duration > 0:
            stats_text = (
                f"Total ripples: {total_ripples} in {active_channels} channels\n"
                f"Window ripples: {window_ripples} ({window_ripples/total_ripples*100:.1f}%)\n"
                f"Avg. rate: {window_ripples/window_duration:.3f} ripples/second\n"
                f"Bin size: {bin_size_seconds}s"
            )
            
            if smoothing_sigma is not None:
                stats_text += f"\nSmoothing: {smoothing_sigma} bins"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_ripple_rate_heatmap_over_time(self, ripples, bin_size_seconds=5, smoothing_sigma=None,
                                    figsize=(14, 8), dpi=300, save_path=None,
                                    time_start=None, time_end=None, min_ripples=5,
                                    channel_subset=None, show_events=False, event_times=None,
                                    colormap='viridis', sort_by_activity=True):
        """
        Create a heatmap visualizing ripple rates over time for multiple channels.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method
        bin_size_seconds : float
            Size of time bins in seconds
        smoothing_sigma : float or None
            Standard deviation for Gaussian smoothing (in bins), None for no smoothing
        figsize : tuple
            Figure size
        dpi : int
            DPI for saving figure
        save_path : str or None
            Path to save the figure (None for no saving)
        time_start : float or None
            Start time for plot display in seconds (None to start from beginning)
        time_end : float or None
            End time for plot display in seconds (None to end at recording end)
        min_ripples : int
            Minimum number of ripples for a channel to be included
        channel_subset : list or None
            List of channels to include (None to use all channels)
        show_events : bool
            Whether to add markers for specific events (e.g., stimulations)
        event_times : list or None
            List of event times in seconds to mark on the plot
        colormap : str
            Colormap name for the heatmap
        sort_by_activity : bool
            Whether to sort channels by total activity (True) or channel number (False)
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the ripple rate heatmap
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        import matplotlib.dates as mdates
        from matplotlib.ticker import MaxNLocator
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8
        })
        
        # Get metadata
        metadata = ripples['metadata']
        fs = metadata['fs']
        
        # Determine recording duration from metadata or max ripple time
        if 'time_end' in metadata and metadata['time_end'] is not None:
            recording_duration = metadata['time_end']
        else:
            # Find max ripple time
            max_time = 0
            for ch in ripples:
                if isinstance(ch, (int, np.integer)) and len(ripples[ch]) > 0:
                    ch_max = ripples[ch]['end_time'].max()
                    max_time = max(max_time, ch_max)
            recording_duration = max_time + 10  # Add buffer
        
        # Adjust time window if specified
        if time_start is None:
            time_start = 0
        if time_end is None:
            time_end = recording_duration
        
        # Create time bins
        n_bins = int(np.ceil((time_end - time_start) / bin_size_seconds))
        bins = np.linspace(time_start, time_start + n_bins * bin_size_seconds, n_bins + 1)
        bin_centers = bins[:-1] + bin_size_seconds / 2
        
        # Collect channel ripple rates
        channel_data = {}
        
        # Filter channels if subset is provided
        if channel_subset is not None:
            valid_channel_set = set(channel_subset)
        else:
            valid_channel_set = {ch for ch in ripples if isinstance(ch, (int, np.integer))}
        
        # Process each channel
        valid_channels = []
        for ch in valid_channel_set:
            if ch in ripples and isinstance(ch, (int, np.integer)):
                ch_ripples = ripples[ch]
                
                # Only include channels with enough ripples
                if len(ch_ripples) >= min_ripples:
                    # Initialize bins for this channel
                    channel_bins = np.zeros(n_bins)
                    
                    # Count ripples per bin
                    for _, ripple in ch_ripples.iterrows():
                        peak_time = ripple['peak_time']
                        # Skip ripples outside the time window
                        if peak_time < time_start or peak_time >= time_end:
                            continue
                        bin_idx = int((peak_time - time_start) / bin_size_seconds)
                        if 0 <= bin_idx < n_bins:
                            channel_bins[bin_idx] += 1
                    
                    # Convert to rate (ripples per second)
                    channel_rate = channel_bins / bin_size_seconds
                    
                    # Apply smoothing if requested
                    if smoothing_sigma is not None:
                        channel_rate_smooth = gaussian_filter1d(
                            channel_rate, sigma=smoothing_sigma, mode='reflect')
                    else:
                        channel_rate_smooth = channel_rate
                    
                    # Store data
                    channel_data[ch] = {
                        'rate': channel_rate_smooth,
                        'total': np.sum(channel_bins),
                        'max_rate': np.max(channel_rate_smooth)
                    }
                    valid_channels.append(ch)
        
        # Skip plotting if no valid channels
        if not valid_channels:
            print("No channels with sufficient ripples to plot.")
            return None
        
        # Sort channels
        if sort_by_activity:
            # Sort by total activity (descending)
            sorted_channels = sorted(valid_channels, 
                                key=lambda ch: channel_data[ch]['total'], 
                                reverse=True)
        else:
            # Sort by channel number
            sorted_channels = sorted(valid_channels)
        
        # Create matrix for heatmap (channels x time)
        n_channels = len(sorted_channels)
        heatmap_data = np.zeros((n_channels, n_bins))
        
        # Fill heatmap data
        for i, ch in enumerate(sorted_channels):
            heatmap_data[i, :] = channel_data[ch]['rate']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap=colormap,
                    extent=[time_start, time_end, n_channels-0.5, -0.5],
                    interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Ripple Rate (ripples/second)', fontsize=10)
        
        # Add event markers if requested
        if show_events and event_times is not None:
            for event_time in event_times:
                if time_start <= event_time <= time_end:
                    ax.axvline(x=event_time, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Set labels
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Channel', fontsize=11)
        
        # Add custom y-ticks with channel numbers
        ytick_positions = np.arange(n_channels)
        ytick_labels = [str(ch) for ch in sorted_channels]
        
        # Only show a reasonable number of y-ticks if there are many channels
        if n_channels > 20:
            # Create sensible ticks (e.g. every 5th channel)
            step = max(1, n_channels // (figsize[1] * 2))
            ytick_positions = ytick_positions[::step]
            ytick_labels = ytick_labels[::step]
        
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)
        
        # Set title
        title = "Sharp Wave Ripple Activity Across Channels Over Time"
        
        # Add time window info to title if specified
        if time_start > 0 or time_end < recording_duration:
            ax.set_title(f"{title}\n(Time window: {time_start:.1f}s - {time_end:.1f}s)", fontsize=12)
        else:
            ax.set_title(title, fontsize=12)
        
        # Add text with statistics
        total_ripples = sum(channel_data[ch]['total'] for ch in valid_channels)
        max_rate = max(channel_data[ch]['max_rate'] for ch in valid_channels)
        
        stats_text = (
            f"Channels: {n_channels}\n"
            f"Total ripples: {int(total_ripples)}\n"
            f"Max rate: {max_rate:.2f} ripples/s\n"
            f"Bin size: {bin_size_seconds}s"
        )
        
        if smoothing_sigma is not None:
            stats_text += f"\nSmoothing: {smoothing_sigma} bins"
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        return fig
    
    def create_sharp_wave_ripple_rate_animation(self, ripples, time_windows=None, window_size_seconds=1.0, 
                                    overlap=0.5, smoothing_factor=2, grid_interpolation='bicubic',
                                    filename='ripple_rate_animation.mp4', fps=10, figsize=(12, 10),
                                    colormap='viridis', min_ripples_per_channel=5, 
                                    min_channels=5, channel_subset=None, title=None,
                                    show_timestamps=True, show_colorbar=True, vmin=None, vmax=None,
                                    include_text_stats=True):
        """
        Create an animated spatial heatmap of sharp wave ripple detection rates across the electrode array.
        
        This method visualizes how ripple rates vary spatially across the array over time,
        allowing for analysis of ripple generation zones and propagation patterns.
        
        Parameters:
        -----------
        ripples : dict
            Output from detect_ripples method containing ripple events for each channel
        time_windows : list of tuples or None
            List of (start_time, end_time) tuples in seconds to define animation frames.
            If None, windows are created automatically using window_size_seconds and overlap.
        window_size_seconds : float
            Size of each time window in seconds when automatically creating windows
        overlap : float
            Overlap fraction between consecutive windows (0-1) when automatically creating windows
        smoothing_factor : float
            Spatial smoothing factor for the ripple rate map
        grid_interpolation : str
            Interpolation method for display ('bicubic', 'nearest', 'bilinear', etc.)
        filename : str
            Output filename for the animation
        fps : int
            Frames per second for the animation
        figsize : tuple
            Figure size
        colormap : str
            Colormap for the heatmap
        min_ripples_per_channel : int
            Minimum number of ripples required for a channel to be included
        min_channels : int
            Minimum number of valid channels required to create the animation
        channel_subset : list or None
            Optional list of channels to include (None for all channels)
        title : str or None
            Custom title for the animation (None for default title)
        show_timestamps : bool
            Whether to display timestamps on each frame
        show_colorbar : bool
            Whether to display the colorbar
        vmin, vmax : float or None
            Fixed minimum and maximum values for colormap (None for auto-scaling)
        include_text_stats : bool
            Whether to include text with statistics in each frame
            
        Returns:
        --------
        bool
            True if animation was successfully created, False otherwise
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        from scipy.ndimage import gaussian_filter
        import os
        import time
        
        # Get metadata from ripples
        metadata = ripples['metadata']
        fs = metadata.get('fs', 1000.0)  # Default to 1000 Hz if not found
        
        # Determine recording duration
        if 'time_end' in metadata and metadata['time_end'] is not None:
            recording_duration = metadata['time_end']
        else:
            # Find max ripple time
            max_time = 0
            for ch in ripples:
                if isinstance(ch, (int, np.integer)) and len(ripples[ch]) > 0:
                    ch_max = ripples[ch]['end_time'].max()
                    max_time = max(max_time, ch_max)
            recording_duration = max_time + 10  # Add buffer
        
        # Get valid channels with enough ripples
        valid_channels = []
        channel_positions = {}
        
        # Filter channels if subset is provided
        if channel_subset is not None:
            valid_channel_set = set(channel_subset)
        else:
            valid_channel_set = {ch for ch in ripples if isinstance(ch, (int, np.integer))}
        
        # Count ripples per channel and get positions
        for ch in valid_channel_set:
            if ch in ripples and isinstance(ch, (int, np.integer)):
                ch_ripples = ripples[ch]
                if len(ch_ripples) >= min_ripples_per_channel:
                    valid_channels.append(ch)
                    
                    # Find channel position in the data_df
                    ch_row = self.data_df[self.data_df['channel'] == ch]
                    if not ch_row.empty:
                        x_pos = ch_row['x_mod'].values[0]
                        y_pos = ch_row['y_mod'].values[0]
                        channel_positions[ch] = (x_pos, y_pos)
        
        # Check if we have enough channels
        if len(valid_channels) < min_channels:
            print(f"Only {len(valid_channels)} channels with sufficient ripples found. Minimum required: {min_channels}")
            return False
        
        print(f"Using {len(valid_channels)} channels for ripple rate animation")
        
        # Calculate grid dimensions
        grid_size_x = int(self.data_df['x_mod'].max()) + 1
        grid_size_y = int(self.data_df['y_mod'].max()) + 1
        
        # Calculate downsampled dimensions (similar to create_matrix_sequence)
        ds_rows = grid_size_y // self.downsample_factor + 1
        ds_cols = grid_size_x // self.downsample_factor + 1
        
        # Create time windows if not provided
        if time_windows is None:
            # Calculate number of windows based on duration and overlap
            effective_step = window_size_seconds * (1 - overlap)
            n_windows = max(2, int((recording_duration - window_size_seconds) / effective_step) + 1)
            
            time_windows = []
            for i in range(n_windows):
                start_time = i * effective_step
                end_time = start_time + window_size_seconds
                if end_time <= recording_duration:
                    time_windows.append((start_time, end_time))
        
        print(f"Creating animation with {len(time_windows)} time windows")
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initialize colormap limits if not provided
        if vmin is None or vmax is None:
            # Calculate sample ripple rates to determine appropriate limits
            sample_rates = []
            for start_time, end_time in time_windows[:min(5, len(time_windows))]:
                # Create empty rate grid
                rate_grid = np.zeros((ds_rows, ds_cols))
                count_grid = np.zeros((ds_rows, ds_cols))
                
                # Calculate rates for this window
                for ch in valid_channels:
                    if ch in channel_positions:
                        x, y = channel_positions[ch]
                        x_ds = int(x) // self.downsample_factor
                        y_ds = int(y) // self.downsample_factor
                        
                        ch_ripples = ripples[ch]
                        # Count ripples in this time window
                        window_ripples = ch_ripples[
                            (ch_ripples['peak_time'] >= start_time) & 
                            (ch_ripples['peak_time'] < end_time)
                        ]
                        
                        n_ripples = len(window_ripples)
                        rate = n_ripples / (end_time - start_time)
                        
                        # Add to rate grid
                        rate_grid[y_ds, x_ds] += rate
                        count_grid[y_ds, x_ds] += 1
                
                # Calculate average rate per position
                with np.errstate(divide='ignore', invalid='ignore'):
                    avg_rate_grid = np.divide(rate_grid, count_grid, 
                                            where=count_grid > 0, 
                                            out=np.zeros_like(rate_grid))
                
                # Apply smoothing
                if smoothing_factor > 0:
                    smoothed_grid = gaussian_filter(avg_rate_grid, sigma=smoothing_factor)
                else:
                    smoothed_grid = avg_rate_grid
                
                # Save non-zero rates for limit calculation
                sample_rates.extend(smoothed_grid[smoothed_grid > 0].flatten())
            
            # Set limits based on percentiles to handle outliers
            if sample_rates:
                if vmin is None:
                    vmin = max(0, np.percentile(sample_rates, 5))
                if vmax is None:
                    vmax = np.percentile(sample_rates, 95)
            else:
                vmin = 0
                vmax = 1  # Default if no rates available
        
        # Create initial empty plot
        im = ax.imshow(np.zeros((ds_rows, ds_cols)), 
                    cmap=colormap, interpolation=grid_interpolation,
                    vmin=vmin, vmax=vmax)
        
        # Add colorbar if requested
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Ripple Rate (ripples/second)')
        
        # Add title
        if title is None:
            title = "Sharp Wave Ripple Rate Spatial Distribution"
        
        title_obj = ax.set_title(title)
        
        # Initialize text object for timestamp and stats
        if show_timestamps or include_text_stats:
            text_obj = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9,
                            verticalalignment='bottom', horizontalalignment='left',
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        else:
            text_obj = None
        
        def update(frame_idx):
            """Update function for animation"""
            start_time, end_time = time_windows[frame_idx]
            window_duration = end_time - start_time
            
            # Create empty rate grid
            rate_grid = np.zeros((ds_rows, ds_cols))
            count_grid = np.zeros((ds_rows, ds_cols))
            
            # Calculate total ripples in this window
            total_ripples = 0
            active_channels = 0
            
            # Calculate rates for this window
            for ch in valid_channels:
                if ch in channel_positions:
                    x, y = channel_positions[ch]
                    x_ds = int(x) // self.downsample_factor
                    y_ds = int(y) // self.downsample_factor
                    
                    ch_ripples = ripples[ch]
                    # Count ripples in this time window
                    window_ripples = ch_ripples[
                        (ch_ripples['peak_time'] >= start_time) & 
                        (ch_ripples['peak_time'] < end_time)
                    ]
                    
                    n_ripples = len(window_ripples)
                    rate = n_ripples / window_duration
                    
                    # Add to rate grid
                    if 0 <= x_ds < ds_cols and 0 <= y_ds < ds_rows:
                        rate_grid[y_ds, x_ds] += rate
                        count_grid[y_ds, x_ds] += 1
                    
                    total_ripples += n_ripples
                    if n_ripples > 0:
                        active_channels += 1
            
            # Calculate average rate per position
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_rate_grid = np.divide(rate_grid, count_grid, 
                                        where=count_grid > 0, 
                                        out=np.zeros_like(rate_grid))
            
            # Apply smoothing
            if smoothing_factor > 0:
                smoothed_grid = gaussian_filter(avg_rate_grid, sigma=smoothing_factor)
            else:
                smoothed_grid = avg_rate_grid
            
            # Update plot
            im.set_array(smoothed_grid)
            
            # Update timestamp and stats if needed
            if show_timestamps or include_text_stats:
                text_content = ""
                if show_timestamps:
                    text_content += f"Time window: {start_time:.1f}s - {end_time:.1f}s"
                
                if include_text_stats:
                    if show_timestamps:
                        text_content += "\n"
                    text_content += f"Ripples: {total_ripples} in {active_channels}/{len(valid_channels)} channels\n"
                    avg_rate = total_ripples / window_duration if window_duration > 0 else 0
                    text_content += f"Avg rate: {avg_rate:.2f} ripples/second"
                
                text_obj.set_text(text_content)
            
            # Update title if needed
            if show_timestamps and title:
                title_obj.set_text(f"{title}\nWindow: {start_time:.1f}s - {end_time:.1f}s")
            
            return [im, title_obj] + ([text_obj] if text_obj else [])
        
        # Create animation
        frames = len(time_windows)
        print(f"Generating animation with {frames} frames...")
        
        anim = FuncAnimation(fig, update, frames=frames, blit=True)
        
        # Save animation
        print(f"Saving animation to {filename}...")
        writer = FFMpegWriter(fps=fps)
        anim.save(os.path.abspath(filename), writer=writer)
        plt.close(fig)
        
        print(f"Animation saved successfully!")
        return True
    
    def plot_phase_distributions(self, data_type, timestamps, n_bins=36, 
                           radius_offset=0.1, save_path=None, figsize=(18, 10), dpi=300,
                           cmap='viridis', separate_plots=True, show_legend=True):
        """
        Plot polar histograms showing phase distributions across channels at specified timestamps.
        
        Parameters:
        -----------
        data_type : str
            Type of data to analyze (e.g., 'theta', 'gamma', etc.)
        timestamps : list or array
            List of timestamps (in seconds) to analyze
        n_bins : int
            Number of bins for the phase histogram
        radius_offset : float
            Minimum radius offset for the polar histogram bars (0.0-1.0).
            Using a non-zero value creates a "donut" shape that can improve
            visual interpretation of angular distributions.
        save_path : str or None
            Path to save the figure. If None, figure is displayed only.
        figsize : tuple
            Figure size for the plots
        dpi : int
            DPI for saved figure
        cmap : str or matplotlib colormap
            Colormap for time progression
        separate_plots : bool
            If True, shows both separate histograms and combined histogram.
            If False, shows only the combined histogram.
        show_legend : bool
            Whether to show the legend in the combined plot
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the phase distribution plots
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        import math
        
        # Ensure data_type exists
        if data_type not in self.waves:
            raise ValueError(f"Data type '{data_type}' not found in waves dictionary")
        
        # Validate radius_offset
        if not 0.0 <= radius_offset < 1.0:
            print(f"Warning: radius_offset should be between 0.0 and 1.0. Setting to 0.1.")
            radius_offset = 0.1
        
        # First, compute the phase data for the entire time series
        print(f"Computing phase data for {data_type}...")
        # Get the whole data array for this band
        data = self.waves[data_type]
        
        # Compute analytical signal for the entire recording at once
        # This uses the _get_analytical_data method but with the entire time range
        analytical_data = self._get_analytical_data(data_type, 0, data.shape[1])
        
        # Store the phase data for easy access
        phase_data = analytical_data['phase']
        
        print(f"Phase data shape: {phase_data.shape}")
        
        # Convert timestamps to sample indices
        fs = self.fs
        if isinstance(timestamps, (int, float)):
            timestamps = [timestamps]  # Convert single timestamp to list
        
        sample_indices = [int(t * fs) for t in timestamps]
        n_times = len(timestamps)
        
        if n_times == 0:
            raise ValueError("No timestamps provided")
        
        # Set up figure based on separate_plots option
        if separate_plots:
            # Calculate grid dimensions for separate plots
            n_cols = min(5, n_times)
            n_rows = math.ceil(n_times / n_cols) + 1  # +1 for combined plot
            fig = plt.figure(figsize=figsize, dpi=dpi)
            
            # Create gridspec for layout
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(n_rows, n_cols, figure=fig)
            
            # Combined plot at the bottom spans all columns
            ax_combined = fig.add_subplot(gs[-1, :], projection='polar')
        else:
            # Just create a single polar plot
            fig = plt.figure(figsize=(figsize[0]//2, figsize[1]), dpi=dpi)
            ax_combined = fig.add_subplot(111, projection='polar')
        
        # Get colormap for time progression
        colormap = get_cmap(cmap, max(n_times, 2))  # Ensure at least 2 colors
        
        # Initialize storage for all phase distributions and valid timestamps
        all_phases = []
        valid_timestamps = []
        valid_indices = []
        max_hist_value = 0  # Track maximum histogram value for consistent scaling
        
        # Process each timestamp
        for i, (t, idx) in enumerate(zip(timestamps, sample_indices)):
            # Skip if index is out of bounds
            if idx >= phase_data.shape[1]:
                print(f"Warning: timestamp {t}s out of bounds. Skipping.")
                continue
                
            # Get phase data at this specific timestamp
            try:
                # Get phases across all electrodes at this timestamp
                phases = phase_data[:, idx]
                    
                # Store valid timestamps and phases
                valid_timestamps.append(t)
                valid_indices.append(i)
                all_phases.append(phases)
                
                # Calculate histogram for this set of phases
                hist, bin_edges = np.histogram(phases, bins=n_bins, range=(-np.pi, np.pi))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Normalize histogram
                hist = hist / np.sum(hist)
                
                # Update maximum histogram value
                max_hist_value = max(max_hist_value, np.max(hist))
                
                if separate_plots:
                    # Create individual polar histogram
                    ax = fig.add_subplot(gs[i // n_cols, i % n_cols], projection='polar')
                    
                    # Plot polar histogram with radius offset
                    bars = ax.bar(bin_centers, hist, width=(2*np.pi/n_bins), 
                                bottom=radius_offset, alpha=0.7, color=colormap(i/n_times))
                    
                    # Set title with radians
                    ax.set_title(f"t = {t:.2f}s", fontsize=10)
                    
                    # Add polar grid and angle labels in radians
                    ax.set_rlabel_position(90)  # Move radial labels away from plotted line
                    ax.set_xticks(np.array([0, np.pi/2, np.pi, 3*np.pi/2]))
                    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2'], fontsize=8)
                    
                    # Remove y-ticks for cleaner appearance
                    ax.set_yticks([])
                    
                    # Set r-limits to include the offset
                    ax.set_ylim(0, max(radius_offset + max_hist_value * 1.2, radius_offset * 2))
                    
            except Exception as e:
                print(f"Error processing timestamp {t}s: {str(e)}")
                continue
        
        # Check if we have any valid data
        if not all_phases:
            raise ValueError("No valid phase data could be extracted for the given timestamps")
        
        # After processing all timestamps, we know the maximum histogram value
        # Use this to set consistent y-limits for all plots
        if separate_plots:
            y_max = max(radius_offset + max_hist_value * 1.2, radius_offset * 2)
            for i in range(min(len(valid_indices), n_rows * n_cols - n_cols)):
                ax = fig.axes[i]
                ax.set_ylim(0, y_max)
        
        # Create combined polar histogram with time-colored distributions
        for i, (phases, t, orig_idx) in enumerate(zip(all_phases, valid_timestamps, valid_indices)):
            hist, bin_edges = np.histogram(phases, bins=n_bins, range=(-np.pi, np.pi))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Normalize histogram
            hist = hist / np.sum(hist)
            
            # Plot with color based on original time index for consistent coloring
            color_idx = orig_idx / max(1, n_times-1)  # Avoid division by zero
            bars = ax_combined.bar(bin_centers, hist, width=(2*np.pi/n_bins), 
                                bottom=radius_offset, alpha=0.5, color=colormap(color_idx),
                                label=f"Time {orig_idx+1}: {t:.2f}s")
        
        # Configure combined plot
        y_max = max(radius_offset + max_hist_value * 1.2, radius_offset * 2)
        ax_combined.set_ylim(0, y_max)
        ax_combined.set_rlabel_position(90)
        
        # Set tick labels in radians
        ax_combined.set_xticks(np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 
                                        5*np.pi/4, 3*np.pi/2, 7*np.pi/4]))
        ax_combined.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', 
                                    '5π/4', '3π/2', '7π/4'])
        
        # Add radial grid and labels that include the offset
        ytick_max = max_hist_value
        yticks = np.linspace(radius_offset, radius_offset + ytick_max, 4)  # 4 radial ticks
        ax_combined.set_yticks(yticks)
        
        # Format y-tick labels to show actual height above offset
        ytick_labels = [f"{y-radius_offset:.1f}" for y in yticks]
        ytick_labels[0] = "0.0"  # Replace first label with 0.0
        ax_combined.set_yticklabels(ytick_labels)
        
        # Set title for combined plot
        if separate_plots:
            ax_combined.set_title("Phase Distributions Over Time", fontsize=12)
        else:
            if len(valid_timestamps) <= 5:
                times_str = ', '.join([f'{t:.2f}s' for t in valid_timestamps])
            else:
                times_str = ', '.join([f'{t:.2f}s' for t in valid_timestamps[:3]]) + f', ... {valid_timestamps[-1]:.2f}s'
            
            ax_combined.set_title(f"Phase Distributions: {data_type.upper()}\nTimestamps: {times_str}", 
                                fontsize=12)
        
        # Add legend if requested and if there aren't too many timestamps
        if show_legend and valid_timestamps:
            if len(valid_timestamps) <= 10:
                # For fewer timestamps, use a standard legend
                ax_combined.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
                                title="Timestamps")
            else:
                # For many timestamps, use a compact legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color=colormap(i/n_times), lw=4, 
                        label=f"Time {i+1}: {valid_timestamps[idx]:.2f}s")
                    for idx, i in enumerate(valid_indices[:10])  # Show only first 10
                ]
                if len(valid_timestamps) > 10:
                    legend_elements.append(Line2D([0], [0], color='none', lw=0, 
                                                label=f"+ {len(valid_timestamps)-10} more"))
                    
                ax_combined.legend(handles=legend_elements, loc='upper right', 
                                bbox_to_anchor=(1.3, 1.0), title="Timestamps", 
                                fontsize=8, ncol=1)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def compute_wave_velocity(self, data_type, initial_frame, final_frame, array_pitch=100.0, 
                         smoothing_factor=1.0, method='gradient'):
        """
        Compute wave propagation velocity based on instantaneous phase gradients.
        
        Parameters:
        -----------
        data_type : str
            Type of data to analyze (e.g., 'theta', 'gamma', etc.)
        initial_frame, final_frame : int
            Start and end frames for analysis
        array_pitch : float
            Physical distance between adjacent electrodes in micrometers
        smoothing_factor : float
            Smoothing factor for the velocity field (0 for no smoothing)
        method : str
            Method to compute velocity ('gradient' or 'optical_flow')
        
        Returns:
        --------
        dict
            Dictionary containing velocity magnitude and direction information
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter
        
        # Ensure we're analyzing phase data
        if not data_type.endswith('_phase'):
            phase_key = f"{data_type}_phase"
        else:
            phase_key = data_type
        
        # Get phase data matrices and gradients
        filtered_matrices, grad_x, grad_y = self.create_matrix_sequence(
            phase_key, initial_frame, final_frame, compute_gradients=True
        )
        
        # Number of frames
        n_frames = final_frame - initial_frame
        
        # Initialize velocity arrays
        velocity_x = np.zeros_like(filtered_matrices)
        velocity_y = np.zeros_like(filtered_matrices)
        velocity_magnitude = np.zeros_like(filtered_matrices)
        velocity_direction = np.zeros_like(filtered_matrices)
        
        # Compute velocity using specified method
        if method == 'gradient':
            # Compute temporal gradient of the phase
            dt = 1.0 / self.fs  # Time step in seconds
            
            # For interior frames, use central differences
            for t in range(1, n_frames-1):
                # Unwrap phase for consistent temporal gradient calculation
                phase_prev = filtered_matrices[:, :, t-1]
                phase_curr = filtered_matrices[:, :, t]
                phase_next = filtered_matrices[:, :, t+1]
                
                # Ensure phase differences are within -π to π
                diff_next = ((phase_next - phase_curr + np.pi) % (2 * np.pi)) - np.pi
                diff_prev = ((phase_curr - phase_prev + np.pi) % (2 * np.pi)) - np.pi
                
                # Compute temporal gradient (rad/s) using central differences
                dphase_dt = (diff_next + diff_prev) / (2 * dt)
                
                # Spatial gradients are already in normalized units (rad/pixel)
                # Convert to physical units (rad/micrometer)
                physical_grad_x = grad_x[:, :, t] / array_pitch
                physical_grad_y = grad_y[:, :, t] / array_pitch
                
                # Compute spatial gradient magnitude
                k_mag = np.sqrt(physical_grad_x**2 + physical_grad_y**2)
                
                # Create mask for valid points (non-zero gradient)
                mask = k_mag > 1e-6
                
                # Initialize with NaN
                v_mag = np.full_like(k_mag, np.nan)
                v_dir = np.full_like(k_mag, np.nan)
                v_x = np.full_like(k_mag, np.nan)
                v_y = np.full_like(k_mag, np.nan)
                
                # Compute velocity for valid points
                v_mag[mask] = np.abs(dphase_dt[mask] / k_mag[mask])
                
                # Wave direction is perpendicular to phase gradient
                # Rotate gradient by 90 degrees for wave propagation direction
                v_x[mask] = -physical_grad_y[mask] / k_mag[mask] * v_mag[mask]
                v_y[mask] = physical_grad_x[mask] / k_mag[mask] * v_mag[mask]
                
                # Calculate direction in radians (0 is east, π/2 is north)
                v_dir[mask] = np.arctan2(v_y[mask], v_x[mask])
                
                # Store results
                velocity_magnitude[:, :, t] = v_mag
                velocity_direction[:, :, t] = v_dir
                velocity_x[:, :, t] = v_x
                velocity_y[:, :, t] = v_y
            
            # Handle edge frames - use forward/backward differences
            if n_frames > 1:
                # First frame - forward difference
                t = 0
                phase_curr = filtered_matrices[:, :, t]
                phase_next = filtered_matrices[:, :, t+1]
                diff_next = ((phase_next - phase_curr + np.pi) % (2 * np.pi)) - np.pi
                dphase_dt = diff_next / dt
                
                # Calculate velocity components (same as above)
                physical_grad_x = grad_x[:, :, t] / array_pitch
                physical_grad_y = grad_y[:, :, t] / array_pitch
                k_mag = np.sqrt(physical_grad_x**2 + physical_grad_y**2)
                mask = k_mag > 1e-6
                
                v_mag = np.full_like(k_mag, np.nan)
                v_x = np.full_like(k_mag, np.nan)
                v_y = np.full_like(k_mag, np.nan)
                v_dir = np.full_like(k_mag, np.nan)
                
                v_mag[mask] = np.abs(dphase_dt[mask] / k_mag[mask])
                v_x[mask] = -physical_grad_y[mask] / k_mag[mask] * v_mag[mask]
                v_y[mask] = physical_grad_x[mask] / k_mag[mask] * v_mag[mask]
                v_dir[mask] = np.arctan2(v_y[mask], v_x[mask])
                
                velocity_magnitude[:, :, t] = v_mag
                velocity_direction[:, :, t] = v_dir
                velocity_x[:, :, t] = v_x
                velocity_y[:, :, t] = v_y
                
                # Last frame - backward difference
                t = n_frames - 1
                phase_prev = filtered_matrices[:, :, t-1]
                phase_curr = filtered_matrices[:, :, t]
                diff_prev = ((phase_curr - phase_prev + np.pi) % (2 * np.pi)) - np.pi
                dphase_dt = diff_prev / dt
                
                # Calculate velocity components (same as above)
                physical_grad_x = grad_x[:, :, t] / array_pitch
                physical_grad_y = grad_y[:, :, t] / array_pitch
                k_mag = np.sqrt(physical_grad_x**2 + physical_grad_y**2)
                mask = k_mag > 1e-6
                
                v_mag = np.full_like(k_mag, np.nan)
                v_x = np.full_like(k_mag, np.nan)
                v_y = np.full_like(k_mag, np.nan)
                v_dir = np.full_like(k_mag, np.nan)
                
                v_mag[mask] = np.abs(dphase_dt[mask] / k_mag[mask])
                v_x[mask] = -physical_grad_y[mask] / k_mag[mask] * v_mag[mask]
                v_y[mask] = physical_grad_x[mask] / k_mag[mask] * v_mag[mask]
                v_dir[mask] = np.arctan2(v_y[mask], v_x[mask])
                
                velocity_magnitude[:, :, t] = v_mag
                velocity_direction[:, :, t] = v_dir
                velocity_x[:, :, t] = v_x
                velocity_y[:, :, t] = v_y
        
        elif method == 'optical_flow':
            # Alternative method using optical flow could be implemented here
            raise NotImplementedError("Optical flow method is not yet implemented")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply smoothing if requested
        if smoothing_factor > 0:
            for t in range(n_frames):
                if not np.all(np.isnan(velocity_x[:, :, t])):  # Only smooth if there's data
                    # Create mask for valid (non-NaN) values
                    valid_mask = ~np.isnan(velocity_x[:, :, t])
                    
                    # Apply smoothing only to valid values
                    temp_x = velocity_x[:, :, t].copy()
                    temp_y = velocity_y[:, :, t].copy()
                    
                    # Replace NaNs with zeros for smoothing
                    temp_x[~valid_mask] = 0
                    temp_y[~valid_mask] = 0
                    
                    # Apply smoothing
                    smoothed_x = gaussian_filter(temp_x, sigma=smoothing_factor)
                    smoothed_y = gaussian_filter(temp_y, sigma=smoothing_factor)
                    
                    # Restore NaNs where original was NaN
                    smoothed_x[~valid_mask] = np.nan
                    smoothed_y[~valid_mask] = np.nan
                    
                    # Store smoothed values
                    velocity_x[:, :, t] = smoothed_x
                    velocity_y[:, :, t] = smoothed_y
            
            # Recalculate magnitude and direction from smoothed components
            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
            velocity_direction = np.arctan2(velocity_y, velocity_x)
        
        # Compute statistics over space and time (excluding NaNs)
        mean_velocity = np.nanmean(velocity_magnitude)
        std_velocity = np.nanstd(velocity_magnitude)
        
        # Calculate circular mean of directions (using vector averaging)
        cos_values = np.cos(velocity_direction)
        sin_values = np.sin(velocity_direction)
        mean_direction = np.arctan2(np.nanmean(sin_values), np.nanmean(cos_values))
        
        # Create result dictionary
        result = {
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'velocity_magnitude': velocity_magnitude,
            'velocity_direction': velocity_direction,
            'mean_velocity': mean_velocity,
            'std_velocity': std_velocity,
            'mean_direction': mean_direction,
            'parameters': {
                'data_type': data_type,
                'array_pitch': array_pitch,
                'smoothing_factor': smoothing_factor,
                'method': method,
                'initial_frame': initial_frame,
                'final_frame': final_frame,
                'sampling_rate': self.fs
            }
        }
        
        return result
    
    def plot_wave_speed_distributions(self, velocity_result, time_indices=None, n_bins=50,
                                 figsize=(12, 8), cmap='viridis', alpha=0.7, 
                                 log_scale=False, speed_range=None, save_path=None,
                                 show_stats=True, hist_type='step'):
        """
        Plot histograms of wave speed distributions at different time points.
        
        Parameters:
        -----------
        velocity_result : dict
            Output from compute_wave_velocity method
        time_indices : list or None
            List of time indices to plot (None for automated selection)
        n_bins : int
            Number of bins for the histograms
        figsize : tuple
            Figure size
        cmap : str
            Colormap for time progression
        alpha : float
            Opacity of histogram lines/fills
        log_scale : bool
            Whether to use logarithmic scale for y-axis
        speed_range : tuple or None
            Range of speeds to include (min, max) in μm/s, None for auto
        save_path : str or None
            Path to save the figure (None for no saving)
        show_stats : bool
            Whether to show statistical information
        hist_type : str
            Type of histogram ('step', 'stepfilled', 'bar')
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the distributions
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        # Extract velocity magnitude data
        v_mag = velocity_result['velocity_magnitude']
        params = velocity_result['parameters']
        
        # Determine time indices to plot
        n_frames = v_mag.shape[2]
        
        if time_indices is None:
            # Automatically select frames (e.g., evenly spaced)
            n_plots = min(10, n_frames)  # Limit to 10 plots for clarity
            time_indices = np.linspace(0, n_frames-1, n_plots, dtype=int)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine speed range if not provided
        if speed_range is None:
            # Filter out NaNs and zeros
            valid_speeds = v_mag[~np.isnan(v_mag) & (v_mag > 0)]
            if len(valid_speeds) > 0:
                # Use percentiles to avoid extreme outliers
                min_speed = np.percentile(valid_speeds, 1)
                max_speed = np.percentile(valid_speeds, 99)
                # Round to nice numbers
                min_speed = np.floor(min_speed / 10) * 10
                max_speed = np.ceil(max_speed / 10) * 10
                speed_range = (min_speed, max_speed)
            else:
                speed_range = (0, 1000)  # Default range if no valid data
        
        # Set up colormap for time progression
        norm = Normalize(vmin=0, vmax=len(time_indices)-1)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        # Create bins
        bins = np.linspace(speed_range[0], speed_range[1], n_bins)
        
        # Store statistical data for each time point
        stats_data = []
        
        # Plot histogram for each time point
        for i, t_idx in enumerate(time_indices):
            # Extract valid data for this frame
            frame_data = v_mag[:, :, t_idx].flatten()
            valid_data = frame_data[(~np.isnan(frame_data)) & 
                                (frame_data >= speed_range[0]) & 
                                (frame_data <= speed_range[1])]
            
            if len(valid_data) > 0:
                # Compute histogram
                hist, edges = np.histogram(valid_data, bins=bins, density=True)
                bin_centers = (edges[:-1] + edges[1:]) / 2
                
                # Get color for this time point
                color = sm.to_rgba(i)
                
                # Plot histogram
                ax.hist(valid_data, bins=bins, alpha=alpha, color=color, 
                    histtype=hist_type, density=True, 
                    label=f'Frame {params["initial_frame"] + t_idx}')
                
                # Calculate statistics
                mean_speed = np.mean(valid_data)
                median_speed = np.median(valid_data)
                std_speed = np.std(valid_data)
                
                stats_data.append({
                    'frame': params["initial_frame"] + t_idx,
                    'time': (params["initial_frame"] + t_idx) / params["sampling_rate"],
                    'mean': mean_speed,
                    'median': median_speed,
                    'std': std_speed,
                    'min': np.min(valid_data),
                    'max': np.max(valid_data),
                    'count': len(valid_data)
                })
        
        # Add colorbar
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Time Progression')
        
        # Set logarithmic scale if requested
        if log_scale:
            ax.set_yscale('log')
        
        # Add labels and title
        ax.set_xlabel('Wave Speed (μm/s)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        
        title = f"Wave Speed Distributions - {params['data_type']}"
        ax.set_title(title, fontsize=14)
        
        # Add grid
        ax.grid(alpha=0.3, linestyle='--')
        
        # Add stats text if requested
        if show_stats and stats_data:
            # Create stats table text
            stats_text = "Frame | Time (s) | Mean (μm/s) | Median (μm/s) | Std (μm/s)\n"
            stats_text += "------------------------------------------------------\n"
            
            for stat in stats_data:
                stats_text += f"{stat['frame']:5d} | {stat['time']:.3f} | {stat['mean']:.2f} | "
                stats_text += f"{stat['median']:.2f} | {stat['std']:.2f}\n"
            
            # Add text box with statistics
            plt.figtext(0.02, 0.02, stats_text, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                    fontfamily='monospace')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_wave_speed_trends(self, velocity_result, window_size=10, 
                          figsize=(12, 6), save_path=None,
                          percentiles=(25, 75), plot_type='both'):
        """
        Plot how wave speed statistics change over time.
        
        Parameters:
        -----------
        velocity_result : dict
            Output from compute_wave_velocity method
        window_size : int
            Window size for smoothing the trend lines
        figsize : tuple
            Figure size
        save_path : str or None
            Path to save the figure (None for no saving)
        percentiles : tuple
            Percentile bounds to show variability (e.g., 25th and 75th)
        plot_type : str
            Type of plot to generate ('line', 'heatmap', or 'both')
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the trend plots
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        
        # Extract velocity magnitude data
        v_mag = velocity_result['velocity_magnitude']
        params = velocity_result['parameters']
        
        # Number of frames
        n_frames = v_mag.shape[2]
        
        # Create time vector
        frame_indices = np.arange(params['initial_frame'], params['initial_frame'] + n_frames)
        time_seconds = frame_indices / params['sampling_rate']
        
        # Calculate statistics for each frame
        mean_speeds = np.zeros(n_frames)
        median_speeds = np.zeros(n_frames)
        std_speeds = np.zeros(n_frames)
        percentile_low = np.zeros(n_frames)
        percentile_high = np.zeros(n_frames)
        
        for t in range(n_frames):
            # Extract valid data for this frame
            frame_data = v_mag[:, :, t].flatten()
            valid_data = frame_data[~np.isnan(frame_data)]
            
            if len(valid_data) > 0:
                mean_speeds[t] = np.mean(valid_data)
                median_speeds[t] = np.median(valid_data)
                std_speeds[t] = np.std(valid_data)
                percentile_low[t] = np.percentile(valid_data, percentiles[0])
                percentile_high[t] = np.percentile(valid_data, percentiles[1])
            else:
                mean_speeds[t] = np.nan
                median_speeds[t] = np.nan
                std_speeds[t] = np.nan
                percentile_low[t] = np.nan
                percentile_high[t] = np.nan
        
        # Apply smoothing with specified window size
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            mean_speeds_smooth = np.convolve(mean_speeds, kernel, mode='same')
            median_speeds_smooth = np.convolve(median_speeds, kernel, mode='same')
            std_speeds_smooth = np.convolve(std_speeds, kernel, mode='same')
            percentile_low_smooth = np.convolve(percentile_low, kernel, mode='same')
            percentile_high_smooth = np.convolve(percentile_high, kernel, mode='same')
        else:
            mean_speeds_smooth = mean_speeds
            median_speeds_smooth = median_speeds
            std_speeds_smooth = std_speeds
            percentile_low_smooth = percentile_low
            percentile_high_smooth = percentile_high
        
        # Create figure based on plot type
        if plot_type == 'both':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                        gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
        
        # Line plot of statistics
        ax1.plot(time_seconds, mean_speeds_smooth, 'b-', label='Mean Speed', linewidth=2)
        ax1.plot(time_seconds, median_speeds_smooth, 'g-', label='Median Speed', linewidth=2)
        
        # Add percentile range
        ax1.fill_between(time_seconds, percentile_low_smooth, percentile_high_smooth,
                        color='blue', alpha=0.2, 
                        label=f'{percentiles[0]}th-{percentiles[1]}th Percentiles')
        
        # Add labels and title for line plot
        ax1.set_xlabel('Time (seconds)' if plot_type != 'both' else '')
        ax1.set_ylabel('Wave Speed (μm/s)')
        ax1.set_title(f"Wave Speed Trends - {params['data_type']}")
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.legend()
        
        # Add heatmap if requested
        if plot_type in ['heatmap', 'both']:
            if plot_type == 'both':
                ax = ax2
            else:
                ax = ax1
            
            # Create a 2D histogram over time
            time_bins = min(50, n_frames)
            speed_bins = 50
            
            # Prepare data for 2D histogram
            hist_data = []
            for t in range(n_frames):
                frame_data = v_mag[:, :, t].flatten()
                valid_data = frame_data[~np.isnan(frame_data)]
                for val in valid_data:
                    hist_data.append((time_seconds[t], val))
            
            if hist_data:
                hist_data = np.array(hist_data)
                
                # Determine speed range using percentiles to avoid extreme outliers
                all_speeds = hist_data[:, 1]
                speed_min = np.percentile(all_speeds, 1)
                speed_max = np.percentile(all_speeds, 99)
                
                # Create 2D histogram
                hist, xedges, yedges = np.histogram2d(
                    hist_data[:, 0], hist_data[:, 1], 
                    bins=[time_bins, speed_bins],
                    range=[[time_seconds[0], time_seconds[-1]], [speed_min, speed_max]]
                )
                
                # Plot heatmap
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im = ax.imshow(hist.T, origin='lower', aspect='auto', 
                            interpolation='nearest', extent=extent,
                            cmap='hot')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Count')
                
                # Add overlay of mean speed
                ax.plot(time_seconds, mean_speeds_smooth, 'c-', linewidth=2)
                
                # Add labels
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Wave Speed (μm/s)')
                
                # Set y-axis limit to match the histogram
                ax.set_ylim(speed_min, speed_max)
        
        # Set x-axis limits
        if plot_type == 'both':
            ax1.set_xlim(time_seconds[0], time_seconds[-1])
            ax2.set_xlim(time_seconds[0], time_seconds[-1])
        else:
            ax1.set_xlim(time_seconds[0], time_seconds[-1])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def visualize_wave_velocity(self, velocity_result, time_indices=None, figsize=(15, 10), 
                           cmap='viridis', scale=20, save_path=None):
        """
        Visualize wave velocity field for selected time frames.
        
        Parameters:
        -----------
        velocity_result : dict
            Output from compute_wave_velocity method
        time_indices : list or None
            List of time indices to plot (None for automated selection)
        figsize : tuple
            Figure size
        cmap : str
            Colormap for velocity magnitude
        scale : float
            Scaling factor for velocity vectors
        save_path : str or None
            Path to save the figure (None for no saving)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the visualization
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data from result
        v_x = velocity_result['velocity_x']
        v_y = velocity_result['velocity_y']
        v_mag = velocity_result['velocity_magnitude']
        params = velocity_result['parameters']
        
        # Determine time indices to plot
        n_frames = v_mag.shape[2]
        
        if time_indices is None:
            # Automatically select frames (e.g., evenly spaced)
            n_plots = min(6, n_frames)
            time_indices = np.linspace(0, n_frames-1, n_plots, dtype=int)
        
        # Calculate grid dimensions for subplots
        n_plots = len(time_indices)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Create mesh grid for quiver plot
        y_grid, x_grid = np.meshgrid(np.arange(v_mag.shape[0]), np.arange(v_mag.shape[1]), indexing='ij')
        
        # Determine global colormap scale
        vmin = np.nanmin(v_mag)
        vmax = np.nanmax(v_mag)
        
        # Plot each time frame
        for i, t_idx in enumerate(time_indices):
            if i < len(axes_flat):
                ax = axes_flat[i]
                
                # Plot velocity magnitude as background
                im = ax.imshow(v_mag[:, :, t_idx], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
                
                # Plot velocity vectors
                # Downsample to reduce clutter if necessary
                skip = max(1, v_mag.shape[0] // 20)
                quiv = ax.quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip],
                            v_x[::skip, ::skip, t_idx], v_y[::skip, ::skip, t_idx],
                            scale=scale, color='w', width=0.003)
                
                # Add timestamp to title
                frame_number = params['initial_frame'] + t_idx
                time_seconds = frame_number / params['sampling_rate']
                ax.set_title(f'Time: {time_seconds:.3f}s (Frame {frame_number})')
                
                # Set equal aspect ratio
                ax.set_aspect('equal')
        
        # Hide unused subplots
        for i in range(n_plots, len(axes_flat)):
            axes_flat[i].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Wave Velocity (μm/s)')
        
        # Add global title
        title = f"Wave Velocity Field - {params['data_type']} "
        title += f"(Pitch: {params['array_pitch']}μm, {params['method']} method)"
        plt.suptitle(title, fontsize=14)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def create_wave_velocity_animation(self, velocity_result, filename='wave_velocity.mp4', 
                                  fps=10, scale=20, cmap='viridis', figsize=(12, 10),
                                  skip_frames=1, vector_skip=2):
        """
        Create an animation of wave velocity over time.
        
        Parameters:
        -----------
        velocity_result : dict
            Output from compute_wave_velocity method
        filename : str
            Output filename for the animation
        fps : int
            Frames per second for the animation
        scale : float
            Scaling factor for velocity vectors
        cmap : str
            Colormap for velocity magnitude
        figsize : tuple
            Figure size
        skip_frames : int
            Number of frames to skip in the animation (1 = use all frames)
        vector_skip : int
            Factor to downsample velocity vectors for clarity
            
        Returns:
        --------
        None
            The animation is saved to the specified filename
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        import numpy as np
        import os
        
        # Extract data from result
        v_x = velocity_result['velocity_x']
        v_y = velocity_result['velocity_y']
        v_mag = velocity_result['velocity_magnitude']
        params = velocity_result['parameters']
        
        # Determine frames to include
        n_frames = v_mag.shape[2]
        frame_indices = np.arange(0, n_frames, skip_frames)
        
        # Create mesh grid for quiver plot
        y_grid, x_grid = np.meshgrid(np.arange(v_mag.shape[0]), np.arange(v_mag.shape[1]), indexing='ij')
        
        # Downsample grid for quiver
        ys = y_grid[::vector_skip, ::vector_skip]
        xs = x_grid[::vector_skip, ::vector_skip]
        
        # Determine color scale
        vmin = np.nanmin(v_mag)
        vmax = np.nanmax(v_mag)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initial plot
        im = ax.imshow(v_mag[:, :, 0], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        quiv = ax.quiver(xs, ys, v_x[::vector_skip, ::vector_skip, 0], 
                    v_y[::vector_skip, ::vector_skip, 0],
                    scale=scale, color='w', width=0.003)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Wave Velocity (μm/s)')
        
        # Add title
        title = ax.set_title(f'Time: 0.000s (Frame {params["initial_frame"]})')
        
        def update(frame_idx):
            # Get the true frame index
            t_idx = frame_indices[frame_idx]
            
            # Update image and quiver
            im.set_array(v_mag[:, :, t_idx])
            quiv.set_UVC(v_x[::vector_skip, ::vector_skip, t_idx], 
                        v_y[::vector_skip, ::vector_skip, t_idx])
            
            # Update title with time information
            frame_number = params['initial_frame'] + t_idx
            time_seconds = frame_number / params['sampling_rate']
            title.set_text(f'Time: {time_seconds:.3f}s (Frame {frame_number})')
            
            return [im, quiv, title]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=True)
        
        # Save animation
        writer = FFMpegWriter(fps=fps)
        anim.save(os.path.abspath(filename), writer=writer)
        plt.close(fig)
        
        print(f"Animation saved to {filename}")
        
    def plot_wave_speed_histogram(self, data_type='theta', time_point=None, 
                              window_size=0.1, bin_size=20, speed_range=None, 
                              figsize=(10, 6), use_config=None, save_path=None,
                              show_stats=True, colormap='viridis',
                              electrode_subset=None, kernel_density=True):
        """
        Generate a histogram of wave propagation speeds at a specific time point.
        
        Parameters:
        -----------
        data_type : str
            Type of oscillation to analyze ('delta', 'theta', etc.)
        time_point : float
            Time point in seconds to analyze. If None, uses the middle of the recording.
        window_size : float
            Time window around the time point to include (in seconds)
        bin_size : int
            Number of bins for the histogram
        speed_range : tuple
            Range of speeds to include (min, max in μm/s)
        figsize : tuple
            Figure size
        use_config : Config
            Configuration object for phase velocity calculation (uses default if None)
        save_path : str
            Path to save the figure (if None, figure is not saved)
        show_stats : bool
            Whether to display statistics on the figure
        colormap : str
            Colormap for the histogram
        electrode_subset : list
            Subset of electrodes to include in analysis (None for all)
        kernel_density : bool
            Whether to overlay a kernel density estimate
        
        Returns:
        --------
        tuple
            (fig, ax, speeds) - Figure, axis and calculated speeds
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        
        # Check if data_type is available
        if data_type not in self.waves:
            raise ValueError(f"Data type '{data_type}' not found in waves dictionary")
        
        # Set default time point (middle of recording) if not provided
        if time_point is None:
            time_point = self.waves[data_type].shape[1] / (2 * self.fs)
        
        # Convert time point to frame index
        frame_idx = int(time_point * self.fs)
        window_frames = int(window_size * self.fs)
        
        # Calculate window boundaries
        start_frame = max(0, frame_idx - window_frames // 2)
        end_frame = min(self.waves[data_type].shape[1], frame_idx + window_frames // 2)
        
        # Get analytical data for this window
        analytical_data = self._get_analytical_data(data_type, start_frame, end_frame)
        phases = analytical_data['phase']
        
        # Get electrode locations
        if electrode_subset is not None:
            # Filter locations to included subset
            included_mask = np.isin(self.data_df['electrode'].values, electrode_subset)
            locations = self.locations[included_mask]
            phases = phases[included_mask]
        else:
            locations = self.locations
        
        # Import necessary functions from example_phase_velocity
        from example_phase_velocity import Config, compute_wave_speeds
        
        # Use provided config or create default
        cfg = use_config if use_config is not None else Config()
        
        # Calculate times to analyze (center of window)
        center_time = (start_frame + end_frame) / (2 * self.fs)
        times = [center_time]
        
        print(f"Computing wave speeds at t={time_point:.2f}s...")
        
        # Compute wave speeds
        speeds = compute_wave_speeds(phases, locations, self.fs, times, cfg)
        
        # Check if we got any valid speeds
        if len(speeds) == 0:
            print(f"No valid wave speeds detected at time {time_point:.2f}s")
            return None, None, []
        
        print(f"Found {len(speeds)} valid speed measurements")
        
        # Create figure and plot histogram
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set range if not provided
        if speed_range is None:
            # Use reasonable defaults or calculate from data
            min_speed = max(0, np.percentile(speeds, 1))  # 1st percentile or 0
            max_speed = min(np.percentile(speeds, 99), cfg.V_MAX)  # 99th percentile or max
            speed_range = (min_speed, max_speed)
        
        # Create histogram with color from colormap
        cmap = plt.get_cmap(colormap)
        color = cmap(0.6)
        
        counts, edges, patches = ax.hist(speeds, bins=bin_size, range=speed_range, 
                                        alpha=0.7, color=color, 
                                        edgecolor='black', linewidth=0.5,
                                        density=True)
        
        # Add kernel density estimate if requested
        if kernel_density and len(speeds) > 3:
            density = stats.gaussian_kde(speeds)
            x = np.linspace(speed_range[0], speed_range[1], 1000)
            ax.plot(x, density(x), 'k-', linewidth=2, label='Density Estimate')
            ax.legend(fontsize=10)
        
        # Customize appearance
        ax.set_xlabel('Wave Speed (μm/s)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Wave Speed Distribution ({data_type.upper()}) at t={time_point:.2f}s', 
                    fontsize=14)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics if requested
        if show_stats:
            # Calculate basic statistics
            mean_speed = np.mean(speeds)
            median_speed = np.median(speeds)
            std_speed = np.std(speeds)
            min_speed = np.min(speeds)
            max_speed = np.max(speeds)
            
            # Add textbox with statistics
            stats_text = (
                f"n = {len(speeds)}\n"
                f"Mean: {mean_speed:.2f} μm/s\n"
                f"Median: {median_speed:.2f} μm/s\n"
                f"Std Dev: {std_speed:.2f} μm/s\n"
                f"Range: [{min_speed:.2f}, {max_speed:.2f}] μm/s"
            )
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
            
            # Add configuration info
            config_text = (
                f"Band: {cfg.F_LOW}-{cfg.F_HIGH} Hz\n"
                f"Window: {window_size:.3f} s\n"
                f"Min Gradient: {cfg.MIN_GRADIENT:.1e} rad/μm"
            )
            
            ax.text(0.02, 0.02, config_text, transform=ax.transAxes,
                horizontalalignment='left', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.tight_layout()
        
        return fig, ax, speeds
    
    def create_cropped_lfp_animation(
            self,
            data_type,
            initial_frame,
            final_frame,
            initial_time=0.0,             # Initial time in seconds corresponding to frame 0
            x_bounds=(None, None),        # Tuple of (min_x, max_x)
            y_bounds=(None, None),        # Tuple of (min_y, max_y)
            include_gradient=False,
            smoothing_factor=None,
            rescale_gradient=False,
            revert_gradient=False,        # NEW: invert gradient vectors if True
            x_neuron=None,
            y_neuron=None,
            neuron_labels=None,
            fps=10,
            figsize=(21, 15),
            neuron_size=50,
            filename='animation.mp4',
            cmap=None,
            add_colorbar=True,
            add_frame_number=False
    ):
        """
        Create an LFP animation showing only a specified spatial region while performing
        calculations on the full matrix to avoid boundary effects.

        Parameters:
        -----------
        data_type : str
            Type of data to visualize (e.g., 'gamma', 'theta')
        initial_frame, final_frame : int
            Start and end frames for the animation
        initial_time : float
            Initial time in seconds corresponding to frame 0 of the recording
        x_bounds : tuple
            (min_x, max_x) coordinates for the region of interest
        y_bounds : tuple
            (min_y, max_y) coordinates for the region of interest
        include_gradient : bool
            Whether to show phase gradient vectors
        smoothing_factor : float
            Gaussian smoothing sigma (None for no smoothing)
        rescale_gradient : bool
            Whether to rescale gradient vectors
        revert_gradient : bool
            When True, invert gradient vectors (multiply by -1) to show propagation direction
        x_neuron, y_neuron : array-like
            Neuron x and y coordinates
        neuron_labels : array-like
            Labels for neurons (1 for excitatory, -1 for inhibitory)
        fps : int
            Frames per second in output animation
        figsize : tuple
            Figure dimensions
        neuron_size : int
            Size of neuron markers
        filename : str
            Output filename for animation
        cmap : str or Colormap
            Colormap to use (will override default selection)
        add_colorbar : bool
            Whether to add a colorbar
        add_frame_number : bool
            Whether to add frame number to the plot
        """
        import os
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        import matplotlib.gridspec as gridspec
        from scipy.ndimage import gaussian_filter
        from cmcrameri import cm as cmc

        # 1) Get matrices with optional gradient computation
        filtered_matrices, grad_x, grad_y = self.create_matrix_sequence(
            data_type, initial_frame, final_frame, compute_gradients=include_gradient
        )

        # 2) Determine spatial bounds
        x_min = 0 if x_bounds[0] is None else x_bounds[0]
        x_max = filtered_matrices.shape[1] if x_bounds[1] is None else x_bounds[1]
        y_min = 0 if y_bounds[0] is None else y_bounds[0]
        y_max = filtered_matrices.shape[0] if y_bounds[1] is None else y_bounds[1]

        # Validate bounds
        if not (0 <= x_min < x_max <= filtered_matrices.shape[1] and
                0 <= y_min < y_max <= filtered_matrices.shape[0]):
            raise ValueError("Invalid spatial bounds")

        # Apply smoothing if requested (on full matrices)
        if smoothing_factor is not None:
            for t in range(filtered_matrices.shape[2]):
                filtered_matrices[:, :, t] = gaussian_filter(
                    filtered_matrices[:, :, t],
                    sigma=smoothing_factor
                )
                if include_gradient and grad_x is not None and grad_y is not None:
                    grad_x[:, :, t] = gaussian_filter(
                        grad_x[:, :, t],
                        sigma=smoothing_factor
                    )
                    grad_y[:, :, t] = gaussian_filter(
                        grad_y[:, :, t],
                        sigma=smoothing_factor
                    )

        # Choose colormap
        if cmap is None:
            if '_phase' in data_type:
                cmap = romaO_rotated #'twilight'
            else:
                cmap = cmc.berlin

        # Normalization
        if '_phase' in data_type:
            norm = plt.Normalize(-np.pi, np.pi)
        else:
            data_vals = filtered_matrices[y_min:y_max, x_min:x_max, :]
            vmax = max(abs(np.nanmin(data_vals)), abs(np.nanmax(data_vals)))
            norm = plt.Normalize(-vmax, vmax)

        # Set up figure and axes
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])

        # Initial image
        im = ax.imshow(
            filtered_matrices[y_min:y_max, x_min:x_max, 0],
            cmap=cmap,
            norm=norm,
            interpolation='bicubic',
            origin='lower'
        )

        # Add title
        if '_phase' in data_type:
            title = data_type.replace('_phase', ' Phase').title()
        else:
            title = data_type.capitalize() + ' Oscillations'

        # Add title with frame number if requested
        if add_frame_number:
            title_with_frame = f"{title}\nFrame {0}"
            title_obj = ax.set_title(title_with_frame, fontsize=14)
        else:
            ax.set_title(title, fontsize=14)

        # Add colorbar
        if add_colorbar:
            plt.colorbar(im, cax=cax)

        # Add time display
        time_text = ax.text(
            0.98, 0.02,
            "Time: 00:00.000",
            transform=ax.transAxes,
            fontsize=16,
            color='white',
            ha='right', va='bottom',
            bbox=dict(facecolor='black', alpha=0.5, pad=5)
        )

        # Prepare plot elements list
        plot_elements = [im, time_text]

        # Prepare quiver grid
        X, Y = np.meshgrid(
            np.arange(0, x_max - x_min),
            np.arange(0, y_max - y_min)
        )

        # Initialize quiver if requested
        quiv = None
        step = 1
        if include_gradient and grad_x is not None and grad_y is not None:
            # Extract the 0th frame gradient
            if rescale_gradient:
                U, V = self._rescale_gradient(
                    grad_x[y_min:y_max, x_min:x_max, 0],
                    grad_y[y_min:y_max, x_min:x_max, 0]
                )
            else:
                U = grad_x[y_min:y_max, x_min:x_max, 0]
                V = grad_y[y_min:y_max, x_min:x_max, 0]

            # Apply optional inversion
            if revert_gradient:
                U = -U
                V = -V

            # Subsample if grid is large
            if X.shape[0] > 50 or X.shape[1] > 50:
                step = max(1, min(X.shape[0], X.shape[1]) // 50)

            quiv = ax.quiver(
                X[::step, ::step], Y[::step, ::step],
                U[::step, ::step], V[::step, ::step],
                pivot='mid',
                scale=98,
                color='white',
                headwidth=3,
                headlength=3,
                headaxislength=2
            )
            plot_elements.append(quiv)

        # Plot neurons if provided
        if all(v is not None for v in [x_neuron, y_neuron, neuron_labels]):
            x_vis = np.array(x_neuron) - x_min
            y_vis = np.array(y_neuron) - y_min
            mask = (
                (x_vis >= 0) & (x_vis < (x_max - x_min)) &
                (y_vis >= 0) & (y_vis < (y_max - y_min))
            )
            x_vis = x_vis[mask]
            y_vis = y_vis[mask]
            labels_vis = np.array(neuron_labels)[mask]

            exc = labels_vis == 1
            inh = labels_vis == -1

            if np.any(exc):
                plot_elements.append(
                    ax.scatter(
                        x_vis[exc], y_vis[exc],
                        marker='x',
                        c='white',
                        s=neuron_size,
                        label='Excitatory',
                        zorder=3
                    )
                )
            if np.any(inh):
                plot_elements.append(
                    ax.scatter(
                        x_vis[inh], y_vis[inh],
                        marker='o',
                        c='black',
                        s=neuron_size,
                        label='Inhibitory',
                        zorder=3
                    )
                )
            ax.legend(loc='upper right', framealpha=0.7, fontsize=10)

        # Clean up axes
        ax.set_xticks(np.arange(0, x_max - x_min, 10))
        ax.set_yticks(np.arange(0, y_max - y_min, 10))

        def update(frame):
            # Update heatmap
            im.set_array(filtered_matrices[y_min:y_max, x_min:x_max, frame])

            # Update quiver if present
            if include_gradient and quiv is not None and grad_x is not None and grad_y is not None:
                if rescale_gradient:
                    U, V = self._rescale_gradient(
                        grad_x[y_min:y_max, x_min:x_max, frame],
                        grad_y[y_min:y_max, x_min:x_max, frame]
                    )
                else:
                    U = grad_x[y_min:y_max, x_min:x_max, frame]
                    V = grad_y[y_min:y_max, x_min:x_max, frame]

                # Invert if requested
                if revert_gradient:
                    U = -U
                    V = -V

                quiv.set_UVC(
                    U[::step, ::step],
                    V[::step, ::step]
                )

            # Update frame number
            if add_frame_number:
                title_obj.set_text(f"{title}\nFrame {frame}")
                plot_elements.append(title_obj)

            # Update time display
            current_time = initial_time + ((initial_frame + frame) / self.fs)
            m = int(current_time // 60)
            s = int(current_time % 60)
            ms = int((current_time % 1) * 1000)
            time_text.set_text(f"Time: {m:02d}:{s:02d}.{ms:03d}")

            return plot_elements

        # Create and save the animation
        anim = FuncAnimation(
            fig,
            update,
            frames=final_frame - initial_frame,
            interval=1000 / fps,
            blit=True
        )
        writer = FFMpegWriter(fps=fps)
        anim.save(os.path.abspath(filename), writer=writer)
        plt.close(fig)
