"""
PTB-XL Experiment Runner
Automated training pipeline with multiple experimental configurations
Implements best practices for PTB-XL dataset training
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tabulate import tabulate

# Experimental configurations for PTB-XL
EXPERIMENT_CONFIGS = {
    "baseline": {
        "name": "Baseline CNN-BiGRU",
        "description": "Standard hybrid architecture without bells and whistles",
        "config": {
            "model_type": "hybrid_full",
            "batch_size": 32,
            "epochs": 50,
            "lr": 3e-4,
            "use_augmentation": False,
            "use_curriculum": False,
            "use_multi_task": False
        }
    },
    
    "full_advanced": {
        "name": "Full Advanced Pipeline",
        "description": "All optimizations: curriculum learning, multi-task, advanced augmentation",
        "config": {
            "model_type": "hybrid_full",
            "batch_size": 32,
            "epochs": 100,
            "lr": 3e-4,
            "use_augmentation": True,
            "use_curriculum": True,
            "use_multi_task": True,
            "mixup_alpha": 0.2,
            "cutmix_alpha": 1.0
        }
    },
    
    "superclass_hierarchical": {
        "name": "Hierarchical Superclass Training",
        "description": "Train on 5 superclasses first, then fine-tune on all classes",
        "config": {
            "model_type": "hybrid_full",
            "batch_size": 64,
            "epochs": 50,
            "lr": 1e-3,
            "use_superclass": True,
            "hierarchical_training": True,
            "freeze_backbone_epochs": 10
        }
    },
    
    "multi_resolution": {
        "name": "Multi-Resolution Training",
        "description": "Train with both 100Hz and 500Hz versions",
        "config": {
            "model_type": "hybrid_full",
            "batch_size": 32,
            "epochs": 75,
            "lr": 3e-4,
            "use_both_frequencies": True,
            "frequency_augmentation": True
        }
    },
    
    "ensemble_diverse": {
        "name": "Diverse Ensemble",
        "description": "Train multiple models with different initializations and architectures",
        "config": {
            "model_type": "ensemble",
            "num_models": 3,
            "batch_size": 32,
            "epochs": 50,
            "lr": 3e-4,
            "diversity_loss_weight": 0.1
        }
    },
    
    "knowledge_distillation": {
        "name": "Knowledge Distillation",
        "description": "Train student model from teacher ensemble",
        "config": {
            "model_type": "hybrid_mobile",
            "teacher_checkpoint": "best_ensemble.pth",
            "batch_size": 64,
            "epochs": 50,
            "lr": 1e-3,
            "distillation_temperature": 4.0,
            "distillation_alpha": 0.7
        }
    },
    
    "active_learning": {
        "name": "Active Learning",
        "description": "Iteratively train on most uncertain samples",
        "config": {
            "model_type": "hybrid_full",
            "initial_samples": 1000,
            "samples_per_iteration": 500,
            "num_iterations": 20,
            "uncertainty_method": "entropy",
            "batch_size": 32,
            "epochs_per_iteration": 10
        }
    }
}


class PTBXLExperimentRunner:
    """Manages and runs PTB-XL experiments"""
    
    def __init__(self, base_dir: str = "experiments/ptbxl"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.base_dir / "experiment_results.csv"
        self.best_models_dir = self.base_dir / "best_models"
        self.best_models_dir.mkdir(exist_ok=True)
    
    def setup_ptbxl_data(self, force_download: bool = False) -> str:
        """Setup PTB-XL dataset"""
        data_path = self.base_dir / "data" / "ptbxl"
        
        if not data_path.exists() or force_download:
            print("Downloading PTB-XL dataset...")
            from app.datasets.ecg_public_datasets import ECGDatasetDownloader
            downloader = ECGDatasetDownloader()
            data_path = downloader.download_ptb_xl(str(data_path.parent))
        
        # Verify dataset integrity
        required_files = [
            "ptbxl_database.csv",
            "scp_statements.csv",
            "records100/00000/00001_hr.hea"  # Sample file
        ]
        
        for file in required_files:
            if not (data_path / file).exists():
                raise FileNotFoundError(f"Required file {file} not found in PTB-XL dataset")
        
        print(f"PTB-XL dataset ready at: {data_path}")
        return str(data_path)
    
    def run_experiment(
        self,
        experiment_name: str,
        config: Dict,
        data_path: str,
        gpu_id: int = 0
    ) -> Dict:
        """Run a single experiment"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"Config: {json.dumps(config, indent=2)}")
        print(f"{'='*60}\n")
        
        # Create experiment directory
        exp_dir = self.base_dir / experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Build command
        cmd = [
            sys.executable,
            "backend/train_ptbxl.py",
            "--data-path", data_path,
            "--batch-size", str(config.get("batch_size", 32)),
            "--epochs", str(config.get("epochs", 50)),
            "--lr", str(config.get("lr", 3e-4)),
            "--device", f"cuda:{gpu_id}",
        ]
        
        if config.get("mixed_precision", True):
            cmd.append("--mixed-precision")
        
        if config.get("use_wandb", False):
            cmd.append("--wandb")
        
        # Set environment variables for advanced features
        env = os.environ.copy()
        env["EXPERIMENT_DIR"] = str(exp_dir)
        env["EXPERIMENT_CONFIG"] = json.dumps(config)
        
        # Run training
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            
            success = True
            error_msg = None
            
        except subprocess.CalledProcessError as e:
            success = False
            error_msg = e.stderr
            print(f"Experiment failed: {error_msg}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Parse results
        if success:
            results = self._parse_training_results(exp_dir)
        else:
            results = {"error": error_msg}
        
        # Save experiment summary
        summary = {
            "experiment_name": experiment_name,
            "config": config,
            "success": success,
            "duration_seconds": duration,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_path = exp_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _parse_training_results(self, exp_dir: Path) -> Dict:
        """Parse training results from experiment directory"""
        results = {}
        
        # Look for test results file
        test_results_file = exp_dir / "checkpoints" / "test_results.json"
        if test_results_file.exists():
            with open(test_results_file, 'r') as f:
                test_results = json.load(f)
                results.update(test_results)
        
        # Look for best model metrics
        best_model_file = exp_dir / "checkpoints" / "best_model.pth"
        if best_model_file.exists():
            import torch
            checkpoint = torch.load(best_model_file, map_location='cpu')
            if 'metrics' in checkpoint:
                results['best_val_metrics'] = checkpoint['metrics']
        
        return results
    
    def run_all_experiments(
        self,
        experiments: Optional[List[str]] = None,
        gpu_ids: Optional[List[int]] = None
    ):
        """Run multiple experiments"""
        data_path = self.setup_ptbxl_data()
        
        if experiments is None:
            experiments = list(EXPERIMENT_CONFIGS.keys())
        
        if gpu_ids is None:
            gpu_ids = [0]  # Default to single GPU
        
        all_results = []
        
        for i, exp_name in enumerate(experiments):
            if exp_name not in EXPERIMENT_CONFIGS:
                print(f"Unknown experiment: {exp_name}")
                continue
            
            exp_config = EXPERIMENT_CONFIGS[exp_name]
            gpu_id = gpu_ids[i % len(gpu_ids)]  # Round-robin GPU assignment
            
            result = self.run_experiment(
                exp_name,
                exp_config["config"],
                data_path,
                gpu_id
            )
            
            all_results.append(result)
            
            # Save incremental results
            self._save_results_table(all_results)
        
        # Generate final report
        self._generate_final_report(all_results)
    
    def _save_results_table(self, results: List[Dict]):
        """Save results in tabular format"""
        rows = []
        
        for result in results:
            row = {
                "Experiment": result["experiment_name"],
                "Success": "✓" if result["success"] else "✗",
                "Duration (min)": f"{result['duration_seconds']/60:.1f}",
                "Test AUC": result.get("results", {}).get("test_auc_macro", "-"),
                "Test F1": result.get("results", {}).get("test_f1_macro", "-"),
                "Val AUC": result.get("results", {}).get("best_val_metrics", {}).get("val_auc_macro", "-"),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.results_file, index=False)
        
        # Print table
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS")
        print("="*80)
        print(tabulate(df, headers='keys', tablefmt='grid', floatfmt=".4f"))
    
    def _generate_final_report(self, results: List[Dict]):
        """Generate comprehensive final report"""
        report_path = self.base_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# PTB-XL Training Experiments Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            successful = sum(1 for r in results if r["success"])
            f.write(f"## Summary\n")
            f.write(f"- Total experiments: {len(results)}\n")
            f.write(f"- Successful: {successful}\n")
            f.write(f"- Failed: {len(results) - successful}\n\n")
            
            # Best performing model
            best_result = max(
                [r for r in results if r["success"]],
                key=lambda x: x.get("results", {}).get("test_auc_macro", 0),
                default=None
            )
            
            if best_result:
                f.write(f"## Best Performing Model\n")
                f.write(f"- Experiment: {best_result['experiment_name']}\n")
                f.write(f"- Test AUC: {best_result['results'].get('test_auc_macro', 0):.4f}\n")
                f.write(f"- Test F1: {best_result['results'].get('test_f1_macro', 0):.4f}\n\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            for result in results:
                f.write(f"### {result['experiment_name']}\n")
                f.write(f"- Status: {'Success' if result['success'] else 'Failed'}\n")
                f.write(f"- Duration: {result['duration_seconds']/60:.1f} minutes\n")
                
                if result["success"] and "results" in result:
                    f.write("- Metrics:\n")
                    for key, value in result["results"].items():
                        if isinstance(value, float):
                            f.write(f"  - {key}: {value:.4f}\n")
                
                f.write("\n")
        
        print(f"\nFinal report saved to: {report_path}")
    
    def compare_models(self, experiment_names: List[str]):
        """Compare multiple trained models"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics_to_plot = [
            ("test_auc_macro", "Test AUC (Macro)"),
            ("test_f1_macro", "Test F1 (Macro)"),
            ("test_mean_label_accuracy", "Mean Label Accuracy"),
            ("val_loss", "Validation Loss")
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            values = []
            labels = []
            
            for exp_name in experiment_names:
                # Load experiment results
                exp_dirs = list((self.base_dir / exp_name).glob("*/summary.json"))
                if exp_dirs:
                    with open(exp_dirs[-1], 'r') as f:  # Latest run
                        summary = json.load(f)
                        if summary["success"]:
                            value = summary.get("results", {}).get(metric, 0)
                            values.append(value)
                            labels.append(exp_name)
            
            ax.bar(labels, values)
            ax.set_title(title)
            ax.set_xlabel("Experiment")
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.base_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run PTB-XL experiments")
    parser.add_argument("--experiments", nargs="+", help="Experiments to run")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--compare", nargs="+", help="Compare specific experiments")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0], help="GPU IDs to use")
    parser.add_argument("--base-dir", type=str, default="experiments/ptbxl", help="Base directory")
    
    args = parser.parse_args()
    
    runner = PTBXLExperimentRunner(args.base_dir)
    
    if args.compare:
        runner.compare_models(args.compare)
    else:
        experiments = None if args.all else args.experiments
        runner.run_all_experiments(experiments, args.gpus)


if __name__ == "__main__":
    main()
