#!/usr/bin/env python3
"""
Runner script for unified traveling waves analysis pipeline
Processes multiple samples from S3 with organized folder structure
Includes planar waves, ripples, and spectral coherence analysis
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
import json

# Configure matplotlib for headless environment BEFORE any imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('/workspace/src')

# Import your modules
from loaders import load_curation, load_info_maxwell
from new_lfp_processor_class import LFPDataProcessor
from spectral_coherence_and_directionality import (
    WaveAnalysisConfig, 
    run_unified_analysis
)

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

def run_analysis_for_sample(uuid, raw_path, spike_path, lfp_path, output_dir, config):
    """
    Run the unified traveling waves analysis for a single sample
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running unified analysis for sample: {uuid}")
    print(f"{'='*60}")
    
    try:
        # Create output directory for this sample
        sample_output_dir = os.path.join(output_dir, uuid)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        print(f"Output directory: {sample_output_dir}")
        
        # Load data
        print("Loading curation data...")
        train, neuron_data, config_dict, fs = load_curation(spike_path)
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
        window_start = 0
        window_length = n_samples
        print(f"Setting dynamic analysis window:")
        print(f"  - Window start: {window_start} samples (0.0 seconds)")
        print(f"  - Window length: {window_length} samples ({recording_duration:.2f} seconds)")
        print(f"  - Processing entire recording duration")
        
        # Create processor
        print("Initializing LFP processor...")
        # Extract electrode locations
        if 'location' in waves:
            locations = waves['location']
            x_mod = locations[:, 0]
            y_mod = locations[:, 1]
        else:
            # Fallback: use locations from config_df
            x_mod = config_df['pos_x'].values
            y_mod = config_df['pos_y'].values
        
        processor = LFPDataProcessor(waves, x_mod, y_mod, config_df)
        
        # Add required frequency bands for unified analysis
        print("Adding frequency bands...")
        processor.add_frequency_band(1, 30, band_name="sharpWave", use_gpu=config.use_gpu, store_analytical=True)
        processor.add_frequency_band(140, 220, band_name="narrowRipples", use_gpu=config.use_gpu, store_analytical=True)
        processor.add_frequency_band(80, 250, band_name="broadRipples", use_gpu=config.use_gpu, store_analytical=False)
        
        # Run unified analysis
        print("\nRunning unified traveling waves analysis...")
        print(f"Configuration:")
        print(f"  - PGD threshold: {config.pgd_threshold}")
        print(f"  - Energy threshold: {config.energy_threshold}")
        print(f"  - Ripple thresholds: {config.ripple_low_threshold}/{config.ripple_high_threshold}")
        print(f"  - GPU enabled: {config.use_gpu}")
        print(f"  - Cache enabled: {config.cache_computations}")
        
        # Define optogenetic intervals if needed (empty for now, can be loaded from metadata)
        optogenetic_intervals = []
        
        # Run the unified analysis
        results = run_unified_analysis(
            lfp_processor=processor,
            window_start=window_start,
            window_length=window_length,
            config=config,
            optogenetic_intervals=optogenetic_intervals,
            output_dir=sample_output_dir
        )
        
        # Save metadata
        metadata = {
            'uuid': uuid,
            'analysis_timestamp': datetime.now().isoformat(),
            'recording_duration_seconds': recording_duration,
            'n_samples': int(n_samples),
            'sampling_rate': float(sampling_rate),
            'window_start': window_start,
            'window_length': window_length,
            'config': {
                'pgd_threshold': config.pgd_threshold,
                'energy_threshold': config.energy_threshold,
                'ripple_low_threshold': config.ripple_low_threshold,
                'ripple_high_threshold': config.ripple_high_threshold,
                'pgd_smoothing_sigma': config.pgd_smoothing_sigma,
                'min_event_duration': config.min_event_duration,
                'use_gpu': config.use_gpu
            }
        }
        
        metadata_path = os.path.join(sample_output_dir, 'analysis_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # List all files generated
        print("\nGenerated files:")
        generated_files = []
        for root, dirs, files in os.walk(sample_output_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), sample_output_dir)
                print(f"  - {rel_path}")
                generated_files.append(rel_path)
        
        # Save file list
        with open(os.path.join(sample_output_dir, 'generated_files.txt'), 'w') as f:
            f.write('\n'.join(generated_files))
        
        print(f"✅ Successfully completed unified analysis for {uuid}")
        print(f"   Processed {recording_duration:.2f} seconds of recording data")
        print(f"   Generated {len(generated_files)} output files")
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing {uuid}: {str(e)}")
        traceback.print_exc()
        
        # Save error log
        error_log_path = os.path.join(sample_output_dir, 'error_log.txt')
        with open(error_log_path, 'w') as f:
            f.write(f"Error analyzing {uuid}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
        
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Run unified traveling waves analysis on multiple samples from S3'
    )
    parser.add_argument(
        '--s3-input-path',
        required=True,
        help='S3 path containing sample folders (e.g., s3://braingeneers/ephys/)'
    )
    parser.add_argument(
        '--s3-output-path',
        required=True,
        help='S3 path for output (e.g., s3://braingeneers/personal/user/unified-waves-results/)'
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
    
    # Add configuration parameters
    parser.add_argument(
        '--pgd-threshold',
        type=float,
        default=1.4,
        help='Threshold for PGD peak detection (default: 1.4)'
    )
    parser.add_argument(
        '--energy-threshold',
        type=float,
        default=1.3,
        help='Threshold for energy peak detection (default: 1.3)'
    )
    parser.add_argument(
        '--ripple-low-threshold',
        type=float,
        default=3.5,
        help='Low threshold for ripple detection (default: 3.5)'
    )
    parser.add_argument(
        '--ripple-high-threshold',
        type=float,
        default=5.0,
        help='High threshold for ripple detection (default: 5.0)'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable computation caching'
    )
    
    args = parser.parse_args()
    
    # Create analysis configuration
    config = WaveAnalysisConfig(
        pgd_threshold=args.pgd_threshold,
        energy_threshold=args.energy_threshold,
        ripple_low_threshold=args.ripple_low_threshold,
        ripple_high_threshold=args.ripple_high_threshold,
        use_gpu=not args.no_gpu,
        cache_computations=not args.no_cache,
        # Keep other defaults from WaveAnalysisConfig
    )
    
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
    
    # Print configuration summary
    print(f"\nAnalysis Configuration:")
    print(f"  PGD threshold: {config.pgd_threshold}")
    print(f"  Energy threshold: {config.energy_threshold}")
    print(f"  Ripple thresholds: {config.ripple_low_threshold}/{config.ripple_high_threshold}")
    print(f"  GPU enabled: {config.use_gpu}")
    print(f"  Caching enabled: {config.cache_computations}")
    
    # Process each sample
    successful = 0
    failed = 0
    processing_times = []
    
    for i, uuid in enumerate(samples, 1):
        start_time = time.time()
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
            args.local_output_dir,
            config
        )
        
        if analysis_success:
            successful += 1
            
            # Upload results to S3
            sample_output_dir = os.path.join(args.local_output_dir, uuid)
            _, output_prefix = s3_handler.parse_s3_path(args.s3_output_path)
            if not output_prefix.endswith('/'):
                output_prefix += '/'
            
            print(f"Uploading results for {uuid}...")
            upload_count = 0
            upload_start = time.time()
            
            for root, dirs, files in os.walk(sample_output_dir):
                for file in files:
                    local_file = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file, args.local_output_dir)
                    s3_key = f"{output_prefix}{relative_path}"
                    if s3_handler.upload_file(local_file, s3_key):
                        upload_count += 1
            
            upload_time = time.time() - upload_start
            print(f"Uploaded {upload_count} files for {uuid} in {upload_time:.2f} seconds")
        else:
            failed += 1
        
        # Clean up local temp files to save space
        print(f"Cleaning up temporary files for {uuid}...")
        import shutil
        shutil.rmtree(sample_temp_dir, ignore_errors=True)
        
        # Track processing time
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        print(f"Sample processing time: {processing_time:.2f} seconds")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"UNIFIED ANALYSIS PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results uploaded to: {args.s3_output_path}")
    
    if processing_times:
        avg_time = np.mean(processing_times)
        total_time = np.sum(processing_times)
        print(f"\nProcessing Statistics:")
        print(f"  Average time per sample: {avg_time:.2f} seconds")
        print(f"  Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Save summary report
    summary_path = os.path.join(args.local_output_dir, 'pipeline_summary.json')
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(samples),
        'successful': successful,
        'failed': failed,
        'input_path': args.s3_input_path,
        'output_path': args.s3_output_path,
        'configuration': {
            'pgd_threshold': config.pgd_threshold,
            'energy_threshold': config.energy_threshold,
            'ripple_low_threshold': config.ripple_low_threshold,
            'ripple_high_threshold': config.ripple_high_threshold,
            'use_gpu': config.use_gpu,
            'cache_computations': config.cache_computations
        },
        'processing_times': {
            'average_seconds': float(np.mean(processing_times)) if processing_times else 0,
            'total_seconds': float(np.sum(processing_times)) if processing_times else 0
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Upload summary
    _, output_prefix = s3_handler.parse_s3_path(args.s3_output_path)
    if not output_prefix.endswith('/'):
        output_prefix += '/'
    summary_s3_key = f"{output_prefix}pipeline_summary.json"
    s3_handler.upload_file(summary_path, summary_s3_key)

if __name__ == "__main__":
    main()