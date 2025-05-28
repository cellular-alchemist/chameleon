#!/usr/bin/env python3
"""
Runner script for planar waves and ripples analysis
Processes multiple samples from S3 with organized folder structure
"""
import os
import sys
import boto3
import argparse
import time
import numpy as np
from datetime import datetime
import traceback
from pathlib import Path

# Add src to path
sys.path.append('/workspace/src')

# Import your modules
from loaders import load_curation, load_info_maxwell
from new_lfp_processor_class import LFPDataProcessor
import planar_waves_and_ripples as pwr

class S3DataHandler:
    """Handle S3 operations for data download and upload"""
    
    def __init__(self, endpoint_url='https://s3.braingeneers.gi.ucsc.edu'):
        self.s3_client = boto3.client('s3', endpoint_url=endpoint_url)
        self.bucket = 'braingeneers'
    
    def parse_s3_path(self, s3_path):
        """Parse S3 path to extract bucket and key"""
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
        
        parts = s3_path.split('/', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], ''
    
    def list_folders(self, s3_prefix):
        """List all folders (UUIDs) in the given S3 prefix"""
        _, prefix = self.parse_s3_path(s3_prefix)
        if not prefix.endswith('/'):
            prefix += '/'
        
        folders = set()
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter='/'):
                if 'CommonPrefixes' in page:
                    for obj in page['CommonPrefixes']:
                        folder_path = obj['Prefix']
                        folder_name = folder_path.rstrip('/').split('/')[-1]
                        folders.add(folder_name)
        except Exception as e:
            print(f"Error listing folders in {s3_prefix}: {e}")
        
        return sorted(list(folders))
    
    def find_sample_files(self, s3_prefix, uuid):
        """Find the three required files for a sample"""
        _, prefix = self.parse_s3_path(s3_prefix)
        if not prefix.endswith('/'):
            prefix += '/'
        
        sample_prefix = f"{prefix}{uuid}/"
        
        files = {
            'raw': None,
            'spike': None,
            'lfp': None
        }
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=sample_prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    filename = key.split('/')[-1]
                    
                    if filename.endswith('.h5'):
                        files['raw'] = key
                    elif filename.endswith('.zip'):
                        files['spike'] = key
                    elif filename.endswith('.npz'):
                        files['lfp'] = key
        
        except Exception as e:
            print(f"Error finding files for {uuid}: {e}")
        
        return files
    
    def download_file(self, s3_key, local_path):
        """Download a file from S3"""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            print(f"  Downloading {os.path.basename(s3_key)}...")
            self.s3_client.download_file(self.bucket, s3_key, local_path)
            return True
        except Exception as e:
            print(f"  Error downloading {s3_key}: {e}")
            return False
    
    def upload_file(self, local_path, s3_key):
        """Upload a file to S3"""
        try:
            self.s3_client.upload_file(local_path, self.bucket, s3_key)
            print(f"  Uploaded {os.path.basename(local_path)} to S3")
            return True
        except Exception as e:
            print(f"  Error uploading {local_path}: {e}")
            return False

def run_analysis_for_sample(uuid, raw_path, spike_path, lfp_path, output_dir):
    """
    Run the planar waves and ripples analysis for a single sample
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running analysis for sample: {uuid}")
    print(f"{'='*60}")
    
    try:
        # Create output directory for this sample
        sample_output_dir = os.path.join(output_dir, uuid)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # Change to output directory so plots are saved there
        original_dir = os.getcwd()
        os.chdir(sample_output_dir)
        
        # Load data
        print("Loading curation data...")
        train, neuron_data, config, fs = load_curation(spike_path)
        train = [np.array(t)*1000 for t in train]
        
        print("Loading Maxwell info...")
        version, time_stamp, config_df, raster_df = load_info_maxwell(raw_path)
        
        print("Loading LFP data...")
        waves = np.load(lfp_path)
        
        # Determine the actual length of the recording dynamically
        print("Determining recording length...")
        if 'lfp' in waves:
            n_samples = waves['lfp'].shape[1]
            sampling_rate = waves.get('fs', 20000.0)  # Get sampling rate from file or use default
            recording_duration = n_samples / sampling_rate
            print(f"Found LFP data with {n_samples} samples at {sampling_rate} Hz")
            print(f"Total recording duration: {recording_duration:.2f} seconds")
        else:
            # Fallback: use any available data to determine length
            available_keys = [k for k in waves.keys() if hasattr(waves[k], 'shape') and len(waves[k].shape) >= 2]
            if available_keys:
                first_key = available_keys[0]
                n_samples = waves[first_key].shape[1]
                sampling_rate = waves.get('fs', 20000.0)
                recording_duration = n_samples / sampling_rate
                print(f"Using {first_key} data to determine length: {n_samples} samples")
                print(f"Total recording duration: {recording_duration:.2f} seconds")
            else:
                print("Error: Could not determine recording length from LFP data")
                raise ValueError("No suitable data found to determine recording length")
        
        # Set analysis parameters dynamically based on actual recording length
        window_start = 14000 #0
        window_length = 14100 #n_samples
        print(f"Setting dynamic analysis window:")
        print(f"  - Window start: {window_start} samples (0.0 seconds)")
        print(f"  - Window length: {window_length} samples ({recording_duration:.2f} seconds)")
        print(f"  - Processing entire recording duration")
        
        # Create processor
        print("Initializing LFP processor...")
        # You'll need to load or create x_mod and y_mod
        # For now, assuming they're in the waves file or need to be computed
        if 'location' in waves:
            locations = waves['location']
            x_mod = locations[:, 0]
            y_mod = locations[:, 1]
        else:
            # Fallback: use locations from config_df
            x_mod = config_df['pos_x'].values
            y_mod = config_df['pos_y'].values
        
        processor = LFPDataProcessor(waves, x_mod, y_mod, config_df)
        processor.add_frequency_band(1, 30, band_name="sharpWave", use_gpu=True, store_analytical=True)
        
        print("Computing PGD for Sharp Wave band...")
        sharp_wave_pgd_data = pwr.compute_pgd_for_window(
            processor,
            data_type='sharpWave',
            window_start=window_start,
            window_length=window_length,
            smoothing_sigma=20,
            min_gradient=1e-5,
            use_gpu=True,
            batch_size=100,
            verbose=False
        )
        
        print("Computing PGD for Narrow Ripple band...")
        narrow_ripple_pgd_data = pwr.compute_pgd_for_window(
            processor,
            data_type='narrowRipples',
            window_start=window_start,
            window_length=window_length,
            smoothing_sigma=15,
            min_gradient=1e-5,
            use_gpu=True,
            batch_size=100,
            verbose=False
        )
        
        # Store precomputed PGD data
        pgd_data_dict = {
            'sharpWave': sharp_wave_pgd_data,
            'narrowRipples': narrow_ripple_pgd_data
        }
        
        print("Detecting planar waves...")
        sharp_wave_pgd = pwr.detect_pgd_peaks_from_precomputed(
            sharp_wave_pgd_data,
            threshold=1.4,
            min_duration=0.1,
            plot_results=True,
            save_path=f'{uuid}_sharp_wave_pgd_peaks'
        )
        
        narrow_ripple_pgd = pwr.detect_pgd_peaks_from_precomputed(
            narrow_ripple_pgd_data,
            threshold=1.5,
            min_duration=0.05,
            plot_results=True,
            save_path=f'{uuid}_narrow_ripple_pgd_peaks'
        )
        
        print("Detecting ripple events...")
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
        
        print("Running comprehensive analysis...")
        results = pwr.analyze_planar_waves_and_ripples_optimized(
            processor,
            sharp_wave_pgd,
            ripple_events,
            pgd_data_dict,
            bands=['sharpWave', 'narrowRipples'],
            window_size=0.5,
            save_dir=f'{uuid}_planar_wave_analysis',
            smoothing_sigma=15,
            verbose=False,
            swr_marker='waveform',
            swr_waveform_window=0.1,
            swr_waveform_height_scale=0.013,
            horizontal_scale_factor=0.2,
        )
        
        print("Visualizing SWR components...")
        swr_result = pwr.visualize_all_swr_components(
            processor, 
            ripple_events,
            time_window=0.3,
            waveform_smoothing_sigma=0.01,
            include_waveform=True,
            save_path=f"{uuid}_swr_analysis",
        )
        
        print("Creating wave analysis plots...")
        figs, axes, wave_results = pwr.plot_pgd_wave_analysis_optimized(
            processor,
            sharp_wave_pgd,
            sharp_wave_pgd_data,
            data_type='sharpWave',
            colormap='cmc.lapaz',
            save_path_base=f'{uuid}_theta_pgd_analysis',
            use_joypy=False,
            fig_width=8,
            fig_height=8
        )
        
        # Save summary statistics
        summary_path = os.path.join(sample_output_dir, f'{uuid}_analysis_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Analysis Summary for {uuid}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add recording information
            f.write(f"Recording Information:\n")
            f.write(f"- Total samples: {n_samples:,}\n")
            f.write(f"- Sampling rate: {sampling_rate} Hz\n")
            f.write(f"- Recording duration: {recording_duration:.2f} seconds ({recording_duration/60:.2f} minutes)\n")
            f.write(f"- Analysis window: {window_start} to {window_length} samples (entire recording)\n\n")
            
            if 'report' in results:
                f.write(results['report'])
            
            if 'statistics' in results:
                f.write("\n\nKey Statistics:\n")
                f.write(f"- Planar waves with ripples: {results['statistics']['pgd_peaks_with_ripples_pct']:.1f}%\n")
                f.write(f"- Ripples during planar waves: {results['statistics']['ripples_during_planar_waves_pct']:.1f}%\n")
        
        # Change back to original directory
        os.chdir(original_dir)
        
        print(f"✅ Successfully completed analysis for {uuid}")
        print(f"   Processed {recording_duration:.2f} seconds of recording data")
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing {uuid}: {str(e)}")
        traceback.print_exc()
        os.chdir(original_dir)  # Make sure to change back even on error
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Run planar waves analysis on multiple samples from S3'
    )
    parser.add_argument(
        '--s3-input-path',
        required=True,
        help='S3 path containing sample folders (e.g., s3://braingeneers/ephys/)'
    )
    parser.add_argument(
        '--s3-output-path',
        required=True,
        help='S3 path for output (e.g., s3://braingeneers/personal/user/planar-waves-results/)'
    )
    parser.add_argument(
        '--specific-samples',
        nargs='+',
        help='Process only specific sample UUIDs (optional)'
    )
    parser.add_argument(
        '--local-temp-dir',
        default='/workspace/data',
        help='Local directory for temporary data storage'
    )
    parser.add_argument(
        '--local-output-dir',
        default='/workspace/output',
        help='Local directory for output before uploading to S3'
    )
    
    args = parser.parse_args()
    
    # Initialize S3 handler
    s3_handler = S3DataHandler()
    
    # Create local directories
    os.makedirs(args.local_temp_dir, exist_ok=True)
    os.makedirs(args.local_output_dir, exist_ok=True)
    
    # Get list of samples to process
    if args.specific_samples:
        samples = args.specific_samples
        print(f"Processing specific samples: {samples}")
    else:
        print(f"Listing samples in {args.s3_input_path}...")
        samples = s3_handler.list_folders(args.s3_input_path)
        print(f"Found {len(samples)} samples to process")
    
    if not samples:
        print("No samples found to process!")
        return
    
    # Process each sample
    successful = 0
    failed = 0
    
    for i, uuid in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Processing {uuid}")
        
        # Find files for this sample
        files = s3_handler.find_sample_files(args.s3_input_path, uuid)
        
        # Check if all required files exist
        if not all(files.values()):
            print(f"⚠️  Skipping {uuid}: Missing required files")
            print(f"   Found: {[k for k, v in files.items() if v]}")
            print(f"   Missing: {[k for k, v in files.items() if not v]}")
            failed += 1
            continue
        
        # Download files
        local_paths = {}
        sample_temp_dir = os.path.join(args.local_temp_dir, uuid)
        os.makedirs(sample_temp_dir, exist_ok=True)
        
        download_success = True
        for file_type, s3_key in files.items():
            local_path = os.path.join(sample_temp_dir, os.path.basename(s3_key))
            if s3_handler.download_file(s3_key, local_path):
                local_paths[file_type] = local_path
            else:
                download_success = False
                break
        
        if not download_success:
            print(f"⚠️  Skipping {uuid}: Download failed")
            failed += 1
            continue
        
        # Run analysis
        analysis_success = run_analysis_for_sample(
            uuid,
            local_paths['raw'],
            local_paths['spike'],
            local_paths['lfp'],
            args.local_output_dir
        )
        
        if analysis_success:
            successful += 1
            
            # Upload results to S3
            sample_output_dir = os.path.join(args.local_output_dir, uuid)
            _, output_prefix = s3_handler.parse_s3_path(args.s3_output_path)
            if not output_prefix.endswith('/'):
                output_prefix += '/'
            
            print(f"Uploading results for {uuid}...")
            for root, dirs, files in os.walk(sample_output_dir):
                for file in files:
                    local_file = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file, args.local_output_dir)
                    s3_key = f"{output_prefix}{relative_path}"
                    s3_handler.upload_file(local_file, s3_key)
        else:
            failed += 1
        
        # Clean up local temp files to save space
        print(f"Cleaning up temporary files for {uuid}...")
        import shutil
        shutil.rmtree(sample_temp_dir, ignore_errors=True)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results uploaded to: {args.s3_output_path}")

if __name__ == "__main__":
    main()