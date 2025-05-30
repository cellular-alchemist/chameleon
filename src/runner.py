#!/usr/bin/env python3
"""
NRP Cluster Runner for Unified Traveling Waves Analysis Pipeline
Processes experiments with sample/condition hierarchy using S3 storage
"""

import os
import sys
import json
import yaml
import boto3
import logging
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import argparse
import tempfile
import shutil
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Configure matplotlib for headless environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add module paths
sys.path.append('/workspace/src')

# Import required modules
from braingeneers import analysis
from loaders import load_curation, load_info_maxwell
from new_lfp_processor_class import LFPDataProcessor
from unified_coherence_analysis import run_unified_analysis, WaveAnalysisConfig

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FileSet:
    """Represents the three required files for analysis"""
    h5_path: str
    zip_path: str
    npz_path: str
    opto_events_path: Optional[str] = None
    
    def validate(self) -> bool:
        """Check if all required paths are provided"""
        return all([self.h5_path, self.zip_path, self.npz_path])

@dataclass
class Condition:
    """Represents a single experimental condition"""
    name: str  # 'baseline', 'opto', or 'acid'
    files: FileSet
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Sample:
    """Represents a sample with multiple conditions"""
    name: str
    conditions: Dict[str, Condition]  # key: condition name
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Experiment:
    """Represents the entire experiment"""
    name: str
    samples: List[Sample]
    config: Dict[str, Any]
    s3_output_base: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# S3 HANDLER
# ============================================================================

class S3Handler:
    """Handles all S3 operations with retry logic"""
    
    def __init__(self, endpoint_url: Optional[str] = None, 
                 region_name: str = 'us-west-2',
                 max_retries: int = 3):
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self.max_retries = max_retries
        
        # Initialize S3 client with retry configuration and s3v4 signature
        from botocore.config import Config
        config = Config(
            signature_version='s3v4',  # CRITICAL: Required for custom S3 endpoints
            region_name=region_name,
            retries={'max_attempts': max_retries, 'mode': 'adaptive'},
            connect_timeout=60,
            read_timeout=300,
            max_pool_connections=20
        )
        
        if endpoint_url:
            self.s3_client = boto3.client('s3', endpoint_url=endpoint_url, config=config)
        else:
            self.s3_client = boto3.client('s3', config=config)
        
        self.logger = logging.getLogger('S3Handler')
    
    def parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """Parse S3 path into bucket and key"""
        if not s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {s3_path}")
        
        path_parts = s3_path[5:].split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        
        return bucket, key
    
    def download_file(self, s3_path: str, local_path: str) -> bool:
        """Download file from S3 with retry logic"""
        bucket, key = self.parse_s3_path(s3_path)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            self.logger.debug(f"Downloading {s3_path} to {local_path}")
            self.s3_client.download_file(bucket, key, local_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to download {s3_path}: {e}")
            return False
    
    def upload_file(self, local_path: str, s3_path: str) -> bool:
        """Upload file to S3 with retry logic"""
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            self.logger.debug(f"Uploading {local_path} to {s3_path}")
            self.s3_client.upload_file(local_path, bucket, key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to upload to {s3_path}: {e}")
            return False
    
    def upload_directory(self, local_dir: str, s3_prefix: str) -> int:
        """Upload entire directory to S3"""
        uploaded_count = 0
        local_dir = Path(local_dir)
        
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                s3_path = f"{s3_prefix}/{relative_path}".replace('\\', '/')
                
                if self.upload_file(str(file_path), s3_path):
                    uploaded_count += 1
        
        return uploaded_count
    
    def check_exists(self, s3_path: str) -> bool:
        """Check if S3 object exists"""
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False

# ============================================================================
# UNIFIED PIPELINE PROCESSOR
# ============================================================================

class UnifiedPipelineProcessor:
    """Processes recordings using the unified analysis pipeline"""
    
    def __init__(self, experiment: Experiment, s3_handler: S3Handler, 
                 work_dir: str, logger: logging.Logger):
        self.experiment = experiment
        self.s3_handler = s3_handler
        self.work_dir = Path(work_dir)
        self.logger = logger
        
        # Create wave analysis config from experiment config
        self.wave_config = self._create_wave_config()
        
        # Setup GPU if available
        self._setup_gpu()
    
    def _create_wave_config(self) -> WaveAnalysisConfig:
        """Create WaveAnalysisConfig from experiment configuration"""
        analysis_cfg = self.experiment.config.get('analysis', {})
        
        return WaveAnalysisConfig(
            # Gradient computation
            max_neighbor_dist=analysis_cfg.get('max_neighbor_dist', 50.0),
            spatial_sigma=analysis_cfg.get('spatial_sigma', 45.0),
            ridge_lambda=analysis_cfg.get('ridge_lambda', 1e-5),
            coverage_angle_gap=analysis_cfg.get('coverage_angle_gap', 120.0),
            min_gradient=analysis_cfg.get('min_gradient', 1e-5),
            
            # PGD computation
            pgd_downsample_factor=analysis_cfg.get('pgd_downsample_factor', 1),
            pgd_smoothing_sigma=analysis_cfg.get('pgd_smoothing_sigma', 15),
            pgd_batch_size=analysis_cfg.get('pgd_batch_size', 100),
            
            # Event detection
            energy_threshold=analysis_cfg.get('energy_threshold', 1.3),
            pgd_threshold=analysis_cfg.get('pgd_threshold', 1.4),
            min_event_duration=analysis_cfg.get('min_event_duration', 0.1),
            min_event_interval=analysis_cfg.get('min_event_interval', 0.2),
            
            # Ripple detection
            ripple_low_threshold=analysis_cfg.get('ripple_low_threshold', 3.5),
            ripple_high_threshold=analysis_cfg.get('ripple_high_threshold', 5.0),
            ripple_min_duration=analysis_cfg.get('ripple_min_duration', 20),
            ripple_max_duration=analysis_cfg.get('ripple_max_duration', 200),
            
            # Spectral analysis
            wavelet_freqs=tuple(analysis_cfg.get('wavelet_freqs', [1, 150])),
            n_wavelets=analysis_cfg.get('n_wavelets', 90),
            wavelet_width=analysis_cfg.get('wavelet_width', 10.0),
            coherence_method=analysis_cfg.get('coherence_method', 'kl_divergence'),
            
            # Performance
            use_gpu=analysis_cfg.get('use_gpu', True),
            cache_computations=analysis_cfg.get('cache_computations', True),
            n_processes=analysis_cfg.get('n_processes', None)
        )
    
    def _setup_gpu(self):
        """Configure GPU if available"""
        if self.wave_config.use_gpu:
            try:
                import cupy as cp
                device_id = self.experiment.config.get('cluster', {}).get('gpu_device', 0)
                cp.cuda.Device(device_id).use()
                self.logger.info(f"Using GPU device {device_id}")
                
                # Log GPU memory
                mempool = cp.get_default_memory_pool()
                self.logger.info(f"GPU memory pool: {mempool.used_bytes() / 1024**3:.2f} GB used")
            except Exception as e:
                self.logger.warning(f"GPU setup failed: {e}. Using CPU.")
                self.wave_config.use_gpu = False
    
    def process_condition(self, sample: Sample, condition: Condition) -> Dict[str, Any]:
        """Process a single condition for a sample"""
        start_time = datetime.now()
        condition_id = f"{sample.name}/{condition.name}"
        
        self.logger.info(f"Processing {condition_id}")
        
        result = {
            'sample': sample.name,
            'condition': condition.name,
            'status': 'started',
            'start_time': start_time.isoformat(),
            'errors': []
        }
        
        # Create temporary directory for this condition
        temp_dir = self.work_dir / sample.name / condition.name
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Download required files
            self.logger.info(f"  Downloading files for {condition_id}")
            local_files = self._download_condition_files(condition, temp_dir)
            
            # Step 2: Load optogenetic intervals if provided
            opto_intervals = self._load_opto_intervals(condition, temp_dir)
            
            # Step 3: Load data and create LFPDataProcessor
            self.logger.info(f"  Loading data for {condition_id}")
            lfp_processor = self._load_data(local_files)
            
            # Step 4: Add frequency bands
            self.logger.info(f"  Adding frequency bands for {condition_id}")
            self._add_frequency_bands(lfp_processor)
            
            # Step 5: Determine analysis window
            window_params = self._determine_window_params(lfp_processor)
            
            # Step 6: Create output directory
            output_dir = temp_dir / 'output'
            output_dir.mkdir(exist_ok=True)
            
            # Step 7: Run unified analysis
            self.logger.info(f"  Running unified analysis for {condition_id}")
            analysis_results = run_unified_analysis(
                lfp_processor,
                window_start=window_params['start'],
                window_length=window_params['length'],
                config=self.wave_config,
                optogenetic_intervals=opto_intervals,
                output_dir=output_dir
            )
            
            # Step 8: Save metadata
            self._save_condition_metadata(
                output_dir, sample, condition, window_params, analysis_results
            )
            
            # Step 9: Upload results to S3
            self.logger.info(f"  Uploading results for {condition_id}")
            s3_output_path = f"{self.experiment.s3_output_base}/{sample.name}/{condition.name}"
            n_uploaded = self.s3_handler.upload_directory(output_dir, s3_output_path)
            
            # Update result
            end_time = datetime.now()
            result.update({
                'status': 'completed',
                'end_time': end_time.isoformat(),
                'duration': (end_time - start_time).total_seconds(),
                's3_output_path': s3_output_path,
                'n_files_uploaded': n_uploaded,
                'window_params': window_params
            })
            
            self.logger.info(f"  Completed {condition_id} in {result['duration']:.2f} seconds")
            
        except Exception as e:
            # Handle errors
            error_msg = f"Error processing {condition_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            result.update({
                'status': 'failed',
                'end_time': datetime.now().isoformat(),
                'errors': [error_msg, traceback.format_exc()]
            })
            
            # Save error log to S3
            error_path = temp_dir / 'error_log.txt'
            with open(error_path, 'w') as f:
                f.write(f"Error processing {condition_id}\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
            
            error_s3_path = f"{self.experiment.s3_output_base}/{sample.name}/{condition.name}/error_log.txt"
            self.s3_handler.upload_file(str(error_path), error_s3_path)
        
        finally:
            # Clean up temporary files
            if self.experiment.config.get('cleanup_temp_files', True):
                self.logger.debug(f"  Cleaning up temporary files for {condition_id}")
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    def _download_condition_files(self, condition: Condition, temp_dir: Path) -> Dict[str, str]:
        """Download all required files for a condition"""
        local_files = {}
        
        # Download main data files
        file_mapping = {
            'h5': ('raw.h5', condition.files.h5_path),
            'zip': ('spike.zip', condition.files.zip_path),
            'npz': ('lfp.npz', condition.files.npz_path)
        }
        
        for file_type, (local_name, s3_path) in file_mapping.items():
            local_path = temp_dir / local_name
            if not self.s3_handler.download_file(s3_path, str(local_path)):
                raise RuntimeError(f"Failed to download {file_type} file from {s3_path}")
            local_files[file_type] = str(local_path)
        
        # Download optogenetic events if provided
        if condition.files.opto_events_path:
            local_path = temp_dir / 'opto_events.npy'
            if self.s3_handler.download_file(condition.files.opto_events_path, str(local_path)):
                local_files['opto_events'] = str(local_path)
        
        return local_files
    
    def _load_opto_intervals(self, condition: Condition, temp_dir: Path) -> Optional[List[Tuple[float, float]]]:
        """Load optogenetic intervals from file"""
        opto_file = temp_dir / 'opto_events.npy'
        
        if opto_file.exists():
            try:
                opto_data = np.load(opto_file)
                # Assume opto_data is array of [start, end] pairs
                if opto_data.ndim == 2 and opto_data.shape[1] == 2:
                    return [(start, end) for start, end in opto_data]
                else:
                    self.logger.warning(f"Unexpected opto_events format: {opto_data.shape}")
            except Exception as e:
                self.logger.warning(f"Failed to load opto_events: {e}")
        
        return None
    
    def _load_data(self, local_files: Dict[str, str]) -> LFPDataProcessor:
        """Load all data and create LFPDataProcessor"""
        # Load spike data
        train, neuron_data, config, fs = load_curation(local_files['zip'])
        train = [np.array(t)*1000 for t in train]
        
        # Load raw data info
        version, time_stamp, config_df, raster_df = load_info_maxwell(local_files['h5'])
        
        # Load LFP data
        waves = np.load(local_files['npz'])
        
        # Get electrode positions
        if 'location' in waves:
            locations = waves['location']
            x_mod = locations[:, 0]
            y_mod = locations[:, 1]
        else:
            # Use positions from config_df
            x_mod = config_df['pos_x'].values
            y_mod = config_df['pos_y'].values
        
        # Create processor
        processor = LFPDataProcessor(waves, x_mod, y_mod, config_df)
        
        return processor
    
    def _add_frequency_bands(self, processor: LFPDataProcessor):
        """Add configured frequency bands to processor"""
        freq_bands = self.experiment.config.get('processing', {}).get('frequency_bands', [])
        
        # Default bands if not specified
        if not freq_bands:
            freq_bands = [
                {'name': 'sharpWave', 'low': 1, 'high': 30, 'store_analytical': True},
                {'name': 'narrowRipples', 'low': 150, 'high': 250, 'store_analytical': False},
                {'name': 'broadRipples', 'low': 80, 'high': 250, 'store_analytical': False}
            ]
        
        for band in freq_bands:
            processor.add_frequency_band(
                band['low'],
                band['high'],
                band_name=band['name'],
                use_gpu=self.wave_config.use_gpu,
                store_analytical=band.get('store_analytical', False)
            )
    
    def _determine_window_params(self, processor: LFPDataProcessor) -> Dict[str, int]:
        """Determine analysis window parameters"""
        # Check if window params are specified in config
        window_config = self.experiment.config.get('processing', {}).get('window', {})
        
        if 'start' in window_config and 'length' in window_config:
            return {
                'start': window_config['start'],
                'length': window_config['length']
            }
        
        # Otherwise, analyze entire recording
        if 'lfp' in processor.waves:
            n_samples = processor.waves['lfp'].shape[1]
        else:
            # Use any available wave data
            for key, data in processor.waves.items():
                if hasattr(data, 'shape') and len(data.shape) >= 2:
                    n_samples = data.shape[1]
                    break
            else:
                raise ValueError("Could not determine recording length")
        
        return {
            'start': 0,
            'length': n_samples
        }
    
    def _save_condition_metadata(self, output_dir: Path, sample: Sample, 
                               condition: Condition, window_params: Dict,
                               analysis_results: Dict):
        """Save metadata for the processed condition"""
        metadata = {
            'experiment': {
                'name': self.experiment.name,
                'metadata': self.experiment.metadata
            },
            'sample': {
                'name': sample.name,
                'metadata': sample.metadata
            },
            'condition': {
                'name': condition.name,
                'metadata': condition.metadata,
                'files': {
                    'h5': condition.files.h5_path,
                    'zip': condition.files.zip_path,
                    'npz': condition.files.npz_path,
                    'opto_events': condition.files.opto_events_path
                }
            },
            'processing': {
                'timestamp': datetime.now().isoformat(),
                'window_params': window_params,
                'config': self.experiment.config
            }
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

# ============================================================================
# EXPERIMENT PROCESSOR
# ============================================================================

class ExperimentProcessor:
    """Manages processing of entire experiments"""
    
    def __init__(self, work_dir: str = '/tmp/pipeline_work',
                 endpoint_url: Optional[str] = None,
                 max_parallel: int = 1):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.s3_handler = S3Handler(endpoint_url=endpoint_url)
        self.setup_logging()
        
        self.max_parallel = max_parallel
        
        # Track progress
        self.progress = {
            'total_conditions': 0,
            'completed': 0,
            'failed': 0,
            'current': None
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.work_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('ExperimentProcessor')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = log_dir / f'experiment_{datetime.now():%Y%m%d_%H%M%S}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Also configure S3Handler logger
        s3_logger = logging.getLogger('S3Handler')
        s3_logger.setLevel(logging.INFO)
        s3_logger.addHandler(fh)
        s3_logger.addHandler(ch)
    
    def load_experiment(self, config_path: str) -> Experiment:
        """Load experiment configuration from file"""
        self.logger.info(f"Loading experiment configuration from {config_path}")
        
        # Check if config is local file or S3 path
        if config_path.startswith('s3://'):
            # Download from S3
            local_config = self.work_dir / 'experiment_config.yaml'
            if not self.s3_handler.download_file(config_path, str(local_config)):
                raise RuntimeError(f"Failed to download config from {config_path}")
            config_path = str(local_config)
        
        # Load configuration
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        # Parse experiment structure
        experiment = self._parse_experiment_config(config_data)
        
        # Count total conditions
        self.progress['total_conditions'] = sum(
            len(sample.conditions) for sample in experiment.samples
        )
        
        self.logger.info(
            f"Loaded experiment '{experiment.name}' with "
            f"{len(experiment.samples)} samples and "
            f"{self.progress['total_conditions']} total conditions"
        )
        
        return experiment
    
    def _parse_experiment_config(self, config_data: Dict) -> Experiment:
        """Parse configuration into Experiment structure"""
        samples = []
        
        for sample_data in config_data['samples']:
            conditions = {}
            
            for condition_name, condition_data in sample_data['conditions'].items():
                files = FileSet(
                    h5_path=condition_data['files']['h5'],
                    zip_path=condition_data['files']['zip'],
                    npz_path=condition_data['files']['npz'],
                    opto_events_path=condition_data['files'].get('opto_events')
                )
                
                condition = Condition(
                    name=condition_name,
                    files=files,
                    metadata=condition_data.get('metadata', {})
                )
                
                conditions[condition_name] = condition
            
            sample = Sample(
                name=sample_data['name'],
                conditions=conditions,
                metadata=sample_data.get('metadata', {})
            )
            
            samples.append(sample)
        
        experiment = Experiment(
            name=config_data['experiment']['name'],
            samples=samples,
            config=config_data.get('config', {}),
            s3_output_base=config_data['experiment']['s3_output_base'],
            metadata=config_data['experiment'].get('metadata', {})
        )
        
        return experiment
    
    def process_experiment(self, experiment: Experiment, 
                         resume_from: Optional[Tuple[str, str]] = None) -> Dict:
        """
        Process entire experiment
        
        Args:
            experiment: Experiment configuration
            resume_from: Optional (sample_name, condition_name) to resume from
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Starting processing of experiment '{experiment.name}'")
        start_time = datetime.now()
        
        # Initialize processor
        processor = UnifiedPipelineProcessor(
            experiment, self.s3_handler, self.work_dir, self.logger
        )
        
        # Create task list
        tasks = []
        skip_mode = resume_from is not None
        
        for sample in experiment.samples:
            for condition_name, condition in sample.conditions.items():
                # Handle resume logic
                if skip_mode:
                    if (sample.name, condition_name) == resume_from:
                        skip_mode = False
                    else:
                        self.logger.info(f"Skipping {sample.name}/{condition_name} (before resume point)")
                        continue
                
                # Check if already processed
                result_marker = f"{experiment.s3_output_base}/{sample.name}/{condition_name}/metadata.json"
                if self.s3_handler.check_exists(result_marker):
                    self.logger.info(f"Skipping {sample.name}/{condition_name} (already processed)")
                    continue
                
                tasks.append((sample, condition))
        
        # Process tasks
        results = []
        
        if self.max_parallel == 1:
            # Sequential processing
            for sample, condition in tasks:
                self.progress['current'] = f"{sample.name}/{condition.name}"
                result = processor.process_condition(sample, condition)
                results.append(result)
                
                # Update progress
                if result['status'] == 'completed':
                    self.progress['completed'] += 1
                else:
                    self.progress['failed'] += 1
                
                self._log_progress()
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                future_to_task = {
                    executor.submit(processor.process_condition, sample, condition): (sample, condition)
                    for sample, condition in tasks
                }
                
                for future in as_completed(future_to_task):
                    sample, condition = future_to_task[future]
                    self.progress['current'] = f"{sample.name}/{condition.name}"
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['status'] == 'completed':
                            self.progress['completed'] += 1
                        else:
                            self.progress['failed'] += 1
                    except Exception as e:
                        self.logger.error(f"Unexpected error processing {sample.name}/{condition.name}: {e}")
                        results.append({
                            'sample': sample.name,
                            'condition': condition.name,
                            'status': 'failed',
                            'errors': [str(e)]
                        })
                        self.progress['failed'] += 1
                    
                    self._log_progress()
        
        # Generate summary
        end_time = datetime.now()
        summary = {
            'experiment_name': experiment.name,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': (end_time - start_time).total_seconds(),
            'total_conditions': self.progress['total_conditions'],
            'completed': self.progress['completed'],
            'failed': self.progress['failed'],
            'results': results
        }
        
        # Save summary to S3
        self._save_experiment_summary(experiment, summary)
        
        # Print final summary
        self._print_summary(summary)
        
        return summary
    
    def _log_progress(self):
        """Log current progress"""
        total = self.progress['total_conditions']
        completed = self.progress['completed']
        failed = self.progress['failed']
        processed = completed + failed
        
        self.logger.info(
            f"Progress: {processed}/{total} "
            f"(Completed: {completed}, Failed: {failed}) "
            f"Current: {self.progress['current']}"
        )
    
    def _save_experiment_summary(self, experiment: Experiment, summary: Dict):
        """Save experiment summary to S3"""
        summary_path = self.work_dir / 'experiment_summary.json'
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        s3_summary_path = f"{experiment.s3_output_base}/experiment_summary.json"
        self.s3_handler.upload_file(str(summary_path), s3_summary_path)
        
        self.logger.info(f"Experiment summary saved to {s3_summary_path}")
    
    def _print_summary(self, summary: Dict):
        """Print processing summary"""
        print("\n" + "="*70)
        print("EXPERIMENT PROCESSING COMPLETE")
        print("="*70)
        print(f"Experiment: {summary['experiment_name']}")
        print(f"Duration: {summary['duration']:.2f} seconds ({summary['duration']/60:.1f} minutes)")
        print(f"Total conditions: {summary['total_conditions']}")
        print(f"Completed: {summary['completed']}")
        print(f"Failed: {summary['failed']}")
        
        if summary['failed'] > 0:
            print("\nFailed conditions:")
            for result in summary['results']:
                if result['status'] == 'failed':
                    print(f"  - {result['sample']}/{result['condition']}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the runner script"""
    parser = argparse.ArgumentParser(
        description='Process traveling waves experiment with unified pipeline'
    )
    
    parser.add_argument(
        'config',
        help='Path to experiment configuration file (local or S3)'
    )
    
    parser.add_argument(
        '--work-dir',
        default='/tmp/pipeline_work',
        help='Local working directory for temporary files'
    )
    
    parser.add_argument(
        '--endpoint-url',
        default='https://s3.braingeneers.gi.ucsc.edu',
        help='S3 endpoint URL (for custom S3 implementations)'
    )
    
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=1,
        help='Maximum parallel conditions to process'
    )
    
    parser.add_argument(
        '--resume-from',
        nargs=2,
        metavar=('SAMPLE', 'CONDITION'),
        help='Resume from specific sample and condition'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Load configuration and show what would be processed without running'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ExperimentProcessor(
        work_dir=args.work_dir,
        endpoint_url=args.endpoint_url,
        max_parallel=args.max_parallel
    )
    
    try:
        # Load experiment
        experiment = processor.load_experiment(args.config)
        
        if args.dry_run:
            # Print what would be processed
            print(f"\nExperiment: {experiment.name}")
            print(f"Output base: {experiment.s3_output_base}")
            print(f"\nSamples and conditions to process:")
            
            for sample in experiment.samples:
                print(f"\n  {sample.name}:")
                for condition_name, condition in sample.conditions.items():
                    print(f"    - {condition_name}")
                    print(f"        h5:  {condition.files.h5_path}")
                    print(f"        zip: {condition.files.zip_path}")
                    print(f"        npz: {condition.files.npz_path}")
                    if condition.files.opto_events_path:
                        print(f"        opto: {condition.files.opto_events_path}")
            
            print(f"\nTotal conditions: {processor.progress['total_conditions']}")
            return
        
        # Process experiment
        resume_from = tuple(args.resume_from) if args.resume_from else None
        summary = processor.process_experiment(experiment, resume_from=resume_from)
        
        # Exit with appropriate code
        sys.exit(0 if summary['failed'] == 0 else 1)
        
    except Exception as e:
        processor.logger.error(f"Fatal error: {e}")
        processor.logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()