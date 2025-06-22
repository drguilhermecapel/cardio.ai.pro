"""
PTB-XL Model Integration Script
Integrates trained PTB-XL models into CardioAI Pro production system
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml

from app.core.config import settings
from app.ml.hybrid_architecture import ModelConfig, create_hybrid_model
from app.services.advanced_ml_service import AdvancedMLService, MLServiceConfig


class PTBXLModelIntegrator:
    """Handles integration of PTB-XL trained models into production"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.production_dir = self.models_dir / "production"
        self.production_dir.mkdir(exist_ok=True)
        
        self.staging_dir = self.models_dir / "staging"
        self.staging_dir.mkdir(exist_ok=True)
    
    def prepare_model_for_production(
        self,
        checkpoint_path: str,
        model_name: str = "ptbxl_hybrid",
        validate: bool = True
    ) -> Dict:
        """Prepare a trained model for production deployment"""
        print(f"Preparing model from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model configuration
        if 'config' in checkpoint:
            training_config = checkpoint['config']
            model_config = training_config.get('model_config', ModelConfig())
        else:
            model_config = ModelConfig()
        
        # Create model and load weights
        model = create_hybrid_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Validate model if requested
        if validate:
            validation_results = self._validate_model(model, model_config)
            print(f"Validation results: {validation_results}")
        
        # Prepare production package
        production_package = {
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'training_metrics': checkpoint.get('metrics', {}),
            'metadata': {
                'model_name': model_name,
                'trained_on': 'PTB-XL',
                'num_classes': model_config.num_classes,
                'created_at': datetime.now().isoformat(),
                'source_checkpoint': str(checkpoint_path),
                'class_names': self._get_ptbxl_classes(model_config.num_classes)
            },
            'preprocessing': {
                'sampling_rate': 500,
                'sequence_length': 5000,
                'num_leads': 12,
                'normalization': 'standardize'
            }
        }
        
        # Save production model
        output_path = self.staging_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(production_package, output_path)
        
        print(f"Production model saved to: {output_path}")
        
        # Generate configuration file
        config_path = self._generate_config_file(output_path, model_name, model_config)
        
        return {
            'model_path': str(output_path),
            'config_path': str(config_path),
            'validation_results': validation_results if validate else None
        }
    
    def _validate_model(self, model: torch.nn.Module, config: ModelConfig) -> Dict:
        """Validate model functionality"""
        model.eval()
        results = {}
        
        try:
            # Test inference
            dummy_input = torch.randn(1, config.input_channels, config.sequence_length)
            with torch.no_grad():
                output = model(dummy_input)
            
            results['inference_test'] = 'PASSED'
            results['output_shape'] = output['logits'].shape
            
            # Check output validity
            predictions = torch.softmax(output['logits'], dim=-1)
            results['predictions_sum'] = float(predictions.sum().item())
            results['predictions_valid'] = abs(results['predictions_sum'] - 1.0) < 1e-5
            
        except Exception as e:
            results['inference_test'] = 'FAILED'
            results['error'] = str(e)
        
        return results
    
    def _get_ptbxl_classes(self, num_classes: int) -> list:
        """Get PTB-XL class names based on number of classes"""
        if num_classes == 5:
            # Superclasses
            return ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        elif num_classes == 71:
            # All SCP codes - simplified list
            return [f'SCP_{i:03d}' for i in range(71)]
        else:
            return [f'Class_{i}' for i in range(num_classes)]
    
    def _generate_config_file(
        self,
        model_path: Path,
        model_name: str,
        model_config: ModelConfig
    ) -> Path:
        """Generate configuration file for the model"""
        config = {
            'model': {
                'name': model_name,
                'path': str(model_path),
                'type': 'hybrid_full',
                'version': '1.0.0'
            },
            'inference': {
                'batch_size': 32,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'mixed_precision': True,
                'num_workers': 4
            },
            'preprocessing': {
                'sampling_rate': 500,
                'target_length': 5000,
                'normalization': 'standardize',
                'filter_params': {
                    'lowcut': 0.5,
                    'highcut': 40.0,
                    'order': 4
                }
            },
            'postprocessing': {
                'confidence_threshold': 0.7,
                'top_k_predictions': 5,
                'apply_clinical_rules': True
            },
            'model_config': {
                'num_classes': model_config.num_classes,
                'input_channels': model_config.input_channels,
                'sequence_length': model_config.sequence_length
            }
        }
        
        config_path = model_path.with_suffix('.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def deploy_to_production(
        self,
        staging_model_path: str,
        force: bool = False
    ) -> Dict:
        """Deploy a staging model to production"""
        staging_path = Path(staging_model_path)
        
        if not staging_path.exists():
            raise FileNotFoundError(f"Staging model not found: {staging_path}")
        
        # Load and validate staging model
        staging_model = torch.load(staging_path, map_location='cpu')
        model_name = staging_model['metadata']['model_name']
        
        # Check if model already exists in production
        prod_path = self.production_dir / f"{model_name}.pth"
        
        if prod_path.exists() and not force:
            raise ValueError(
                f"Model {model_name} already exists in production. "
                "Use --force to overwrite."
            )
        
        # Backup existing model if it exists
        if prod_path.exists():
            backup_path = self.production_dir / f"{model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            shutil.copy(prod_path, backup_path)
            print(f"Backed up existing model to: {backup_path}")
        
        # Copy to production
        shutil.copy(staging_path, prod_path)
        
        # Copy config file if exists
        staging_config = staging_path.with_suffix('.yaml')
        if staging_config.exists():
            prod_config = prod_path.with_suffix('.yaml')
            shutil.copy(staging_config, prod_config)
        
        # Update production manifest
        self._update_production_manifest(model_name, prod_path)
        
        print(f"Model deployed to production: {prod_path}")
        
        return {
            'production_path': str(prod_path),
            'model_name': model_name,
            'deployed_at': datetime.now().isoformat()
        }
    
    def _update_production_manifest(self, model_name: str, model_path: Path):
        """Update the production models manifest"""
        manifest_path = self.production_dir / "manifest.json"
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {'models': {}}
        
        manifest['models'][model_name] = {
            'path': str(model_path),
            'deployed_at': datetime.now().isoformat(),
            'active': True
        }
        
        # Set other models as inactive
        for name in manifest['models']:
            if name != model_name:
                manifest['models'][name]['active'] = False
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def create_production_service(
        self,
        model_name: str = "ptbxl_hybrid"
    ) -> AdvancedMLService:
        """Create a production-ready ML service"""
        # Get model path from manifest
        manifest_path = self.production_dir / "manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError("No production models found")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        if model_name not in manifest['models']:
            raise ValueError(f"Model {model_name} not found in production")
        
        model_info = manifest['models'][model_name]
        model_path = model_info['path']
        
        # Create service configuration
        config = MLServiceConfig(
            model_type="hybrid_full",
            model_path=model_path,
            use_pretrained=True,
            inference_mode="accurate",
            enable_interpretability=True,
            confidence_threshold=0.7,
            quality_threshold=0.7
        )
        
        # Create service
        service = AdvancedMLService(config)
        
        print(f"Created production service with model: {model_name}")
        
        return service
    
    def benchmark_production_model(
        self,
        model_name: str = "ptbxl_hybrid",
        num_samples: int = 100
    ) -> Dict:
        """Benchmark production model performance"""
        import time
        
        service = self.create_production_service(model_name)
        
        # Generate test data
        test_data = np.random.randn(num_samples, 12, 5000).astype(np.float32)
        
        latencies = []
        
        print(f"Benchmarking {model_name} with {num_samples} samples...")
        
        for i in range(num_samples):
            start = time.time()
            
            # Run inference
            prediction = service.analyze_ecg(
                test_data[i],
                sampling_rate=500,
                return_interpretability=False
            )
            
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_samples} samples")
        
        # Calculate statistics
        results = {
            'model_name': model_name,
            'num_samples': num_samples,
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_samples_per_sec': 1000 / np.mean(latencies)
        }
        
        print("\nBenchmark Results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Integrate PTB-XL trained models into CardioAI Pro"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Prepare command
    prepare_parser = subparsers.add_parser(
        'prepare',
        help='Prepare model for production'
    )
    prepare_parser.add_argument(
        'checkpoint',
        help='Path to model checkpoint'
    )
    prepare_parser.add_argument(
        '--name',
        default='ptbxl_hybrid',
        help='Model name'
    )
    prepare_parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation'
    )
    
    # Deploy command
    deploy_parser = subparsers.add_parser(
        'deploy',
        help='Deploy model to production'
    )
    deploy_parser.add_argument(
        'model',
        help='Path to staging model'
    )
    deploy_parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing model'
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Benchmark production model'
    )
    benchmark_parser.add_argument(
        '--name',
        default='ptbxl_hybrid',
        help='Model name'
    )
    benchmark_parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of samples'
    )
    
    # Test command
    test_parser = subparsers.add_parser(
        'test',
        help='Test production service'
    )
    test_parser.add_argument(
        '--name',
        default='ptbxl_hybrid',
        help='Model name'
    )
    
    args = parser.parse_args()
    
    # Execute command
    integrator = PTBXLModelIntegrator()
    
    if args.command == 'prepare':
        result = integrator.prepare_model_for_production(
            args.checkpoint,
            args.name,
            validate=not args.no_validate
        )
        print(f"\nModel prepared successfully!")
        print(f"Staging path: {result['model_path']}")
        
    elif args.command == 'deploy':
        result = integrator.deploy_to_production(
            args.model,
            force=args.force
        )
        print(f"\nModel deployed successfully!")
        print(f"Production path: {result['production_path']}")
        
    elif args.command == 'benchmark':
        results = integrator.benchmark_production_model(
            args.name,
            args.samples
        )
        
    elif args.command == 'test':
        # Test production service
        service = integrator.create_production_service(args.name)
        
        # Generate test ECG
        test_ecg = np.random.randn(12, 5000).astype(np.float32)
        
        print("Testing production service...")
        prediction = service.analyze_ecg(
            test_ecg,
            sampling_rate=500,
            return_interpretability=True
        )
        
        print(f"\nPrediction:")
        print(f"Condition: {prediction.condition_name}")
        print(f"Probability: {prediction.probability:.4f}")
        print(f"Confidence: {prediction.confidence}")
        print(f"Processing time: {prediction.processing_time_ms:.2f} ms")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
