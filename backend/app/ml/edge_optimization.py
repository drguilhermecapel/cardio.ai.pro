"""
Edge Computing Optimization Module for ECG Analysis
Implements model compression, quantization, and hardware acceleration
Enables deployment on mobile devices and embedded systems
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    get_default_qconfig,
    prepare,
    quantize_dynamic,
)

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX not available. Install with: pip install onnx onnxruntime")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT not available for GPU acceleration")

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logging.warning("CoreML not available for iOS deployment")

from app.ml.hybrid_architecture import HybridECGModel, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class EdgeOptimizationConfig:
    """Configuration for edge optimization"""
    
    # Compression techniques
    enable_pruning: bool = True
    pruning_sparsity: float = 0.5
    structured_pruning: bool = True
    
    enable_quantization: bool = True
    quantization_backend: str = "qnnpack"  # "qnnpack" for mobile, "fbgemm" for server
    quantization_mode: str = "dynamic"  # "dynamic", "static", "qat"
    
    enable_knowledge_distillation: bool = True
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    
    # Model architecture optimization
    use_depthwise_separable: bool = True
    use_mobile_inverted_bottleneck: bool = True
    width_multiplier: float = 1.0
    resolution_multiplier: float = 1.0
    
    # Hardware acceleration
    use_gpu: bool = True
    use_tensorrt: bool = TENSORRT_AVAILABLE
    tensorrt_precision: str = "fp16"  # "fp32", "fp16", "int8"
    
    use_nnapi: bool = True  # Android Neural Networks API
    use_coreml: bool = COREML_AVAILABLE  # iOS Core ML
    
    # Export formats
    export_onnx: bool = True
    export_tflite: bool = True
    export_coreml: bool = COREML_AVAILABLE
    export_torchscript: bool = True
    
    # Performance targets
    target_latency_ms: float = 100.0
    target_model_size_mb: float = 10.0
    target_memory_usage_mb: float = 50.0
    
    # Optimization constraints
    min_accuracy_threshold: float = 0.95
    max_accuracy_drop: float = 0.02


class ModelPruner:
    """Advanced model pruning with structured and unstructured techniques"""
    
    def __init__(self, config: EdgeOptimizationConfig):
        self.config = config
    
    def prune_model(
        self,
        model: nn.Module,
        validation_fn: Optional[callable] = None
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """Apply pruning to model"""
        original_params = self._count_parameters(model)
        
        if self.config.structured_pruning:
            model = self._structured_pruning(model)
        else:
            model = self._unstructured_pruning(model)
        
        # Fine-tune after pruning if validation function provided
        if validation_fn is not None:
            model = self._finetune_pruned_model(model, validation_fn)
        
        # Remove pruning reparameterization
        model = self._remove_pruning(model)
        
        pruned_params = self._count_parameters(model)
        
        metrics = {
            'original_parameters': original_params,
            'pruned_parameters': pruned_params,
            'compression_ratio': original_params / pruned_params,
            'sparsity': 1 - (pruned_params / original_params)
        }
        
        return model, metrics
    
    def _structured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning (channel/filter pruning)"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                # Prune output channels based on L2 norm
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=self.config.pruning_sparsity,
                    n=2,
                    dim=0
                )
            elif isinstance(module, nn.Linear):
                # Prune neurons
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=self.config.pruning_sparsity,
                    n=2,
                    dim=0
                )
        
        return model
    
    def _unstructured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply unstructured pruning (weight pruning)"""
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config.pruning_sparsity
        )
        
        return model
    
    def _remove_pruning(self, model: nn.Module) -> nn.Module:
        """Remove pruning reparameterization"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                if hasattr(module, 'weight_orig'):
                    prune.remove(module, 'weight')
        
        return model
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count non-zero parameters"""
        total = 0
        for p in model.parameters():
            total += (p != 0).sum().item()
        return total
    
    def _finetune_pruned_model(
        self,
        model: nn.Module,
        validation_fn: callable,
        epochs: int = 10
    ) -> nn.Module:
        """Fine-tune pruned model"""
        # Simplified fine-tuning - in practice, use full training pipeline
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            loss = validation_fn(model)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        return model


class ModelQuantizer:
    """Model quantization for reduced precision inference"""
    
    def __init__(self, config: EdgeOptimizationConfig):
        self.config = config
        
        # Set backend
        torch.backends.quantized.engine = config.quantization_backend
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[DataLoader] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply quantization to model"""
        original_size = self._get_model_size(model)
        
        if self.config.quantization_mode == "dynamic":
            quantized_model = self._dynamic_quantization(model)
        elif self.config.quantization_mode == "static":
            if calibration_data is None:
                raise ValueError("Static quantization requires calibration data")
            quantized_model = self._static_quantization(model, calibration_data)
        elif self.config.quantization_mode == "qat":
            quantized_model = self._quantization_aware_training(model)
        else:
            raise ValueError(f"Unknown quantization mode: {self.config.quantization_mode}")
        
        quantized_size = self._get_model_size(quantized_model)
        
        metrics = {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size,
            'quantization_mode': self.config.quantization_mode
        }
        
        return quantized_model, metrics
    
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization"""
        quantized_model = quantize_dynamic(
            model,
            qconfig_spec={
                nn.Linear: torch.quantization.default_dynamic_qconfig,
                nn.LSTM: torch.quantization.default_dynamic_qconfig,
                nn.GRU: torch.quantization.default_dynamic_qconfig
            },
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def _static_quantization(
        self,
        model: nn.Module,
        calibration_data: DataLoader
    ) -> nn.Module:
        """Apply static quantization with calibration"""
        # Prepare model
        model.eval()
        
        # Fuse modules
        model = self._fuse_modules(model)
        
        # Prepare for quantization
        model.qconfig = get_default_qconfig(self.config.quantization_backend)
        prepare(model, inplace=True)
        
        # Calibrate with representative data
        with torch.no_grad():
            for data, _ in calibration_data:
                model(data)
        
        # Convert to quantized model
        quantized_model = convert(model, inplace=True)
        
        return quantized_model
    
    def _quantization_aware_training(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization-aware training"""
        # Add quantization stubs
        model = self._add_quant_stubs(model)
        
        # Fuse modules
        model = self._fuse_modules(model)
        
        # Prepare for QAT
        model.qconfig = torch.quantization.get_default_qat_qconfig(
            self.config.quantization_backend
        )
        model.train()
        prepare_qat(model, inplace=True)
        
        return model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse conv-bn-relu modules for efficiency"""
        modules_to_fuse = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                # Check if followed by BatchNorm and ReLU
                modules_to_fuse.append([name, f"{name}_bn", f"{name}_relu"])
        
        if modules_to_fuse:
            torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
        
        return model
    
    def _add_quant_stubs(self, model: nn.Module) -> nn.Module:
        """Add quantization and dequantization stubs"""
        class QuantizedWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.quant = QuantStub()
                self.model = model
                self.dequant = DeQuantStub()
            
            def forward(self, x):
                x = self.quant(x)
                x = self.model(x)
                x = self.dequant(x)
                return x
        
        return QuantizedWrapper(model)
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return size_mb


class MobileArchitectureOptimizer:
    """Optimize architecture for mobile deployment"""
    
    def __init__(self, config: EdgeOptimizationConfig):
        self.config = config
    
    def create_mobile_model(self, original_config: ModelConfig) -> nn.Module:
        """Create mobile-optimized model architecture"""
        # Create mobile config
        mobile_config = self._create_mobile_config(original_config)
        
        # Build mobile model
        model = MobileECGNet(mobile_config)
        
        return model
    
    def _create_mobile_config(self, original_config: ModelConfig) -> ModelConfig:
        """Adapt configuration for mobile deployment"""
        mobile_config = ModelConfig(
            input_channels=original_config.input_channels,
            sequence_length=int(original_config.sequence_length * self.config.resolution_multiplier),
            num_classes=original_config.num_classes,
            
            # Reduced model capacity
            cnn_growth_rate=int(original_config.cnn_growth_rate * self.config.width_multiplier),
            cnn_num_blocks=(4, 8, 12, 8),  # Fewer blocks
            
            # Smaller RNN
            gru_hidden_dim=int(original_config.gru_hidden_dim * self.config.width_multiplier),
            gru_num_layers=2,  # Fewer layers
            
            # Smaller transformer
            transformer_d_model=int(original_config.transformer_d_model * self.config.width_multiplier),
            transformer_heads=4,  # Fewer heads
            transformer_layers=3,  # Fewer layers
            
            # Disable some features for efficiency
            use_frequency_attention=False,
            use_multi_head_attention=True,
            use_channel_attention=False
        )
        
        return mobile_config


class MobileECGNet(nn.Module):
    """Lightweight ECG model for mobile deployment"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Initial feature extraction with depthwise separable convolutions
        self.stem = nn.Sequential(
            DepthwiseSeparableConv1d(
                config.input_channels, 32, kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm1d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool1d(2)
        )
        
        # Mobile inverted bottleneck blocks
        self.blocks = nn.ModuleList([
            MobileInvertedBottleneck(32, 64, stride=2, expand_ratio=6),
            MobileInvertedBottleneck(64, 128, stride=2, expand_ratio=6),
            MobileInvertedBottleneck(128, 256, stride=2, expand_ratio=6)
        ])
        
        # Lightweight temporal modeling
        self.temporal = nn.GRU(
            256,
            config.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config.gru_hidden_dim * 2, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Feature extraction
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        # Temporal modeling
        x = x.transpose(1, 2)  # (B, T, C)
        x, _ = self.temporal(x)
        x = x.transpose(1, 2)  # (B, C, T)
        
        # Classification
        logits = self.classifier(x)
        
        return {
            'logits': logits,
            'predictions': torch.softmax(logits, dim=-1)
        }


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for efficiency"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileInvertedBottleneck(nn.Module):
    """Mobile inverted bottleneck block (MobileNetV2 style)"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 6
    ):
        super().__init__()
        
        hidden_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
                bias=False
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU6(inplace=True)
        ])
        
        # Projection
        layers.extend([
            nn.Conv1d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        ])
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


class ModelExporter:
    """Export models to various formats for deployment"""
    
    def __init__(self, config: EdgeOptimizationConfig):
        self.config = config
    
    def export_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_dir: Union[str, Path],
        model_name: str = "ecg_model"
    ) -> Dict[str, Path]:
        """Export model to multiple formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_paths = {}
        
        # TorchScript
        if self.config.export_torchscript:
            path = self._export_torchscript(model, sample_input, output_dir, model_name)
            exported_paths['torchscript'] = path
        
        # ONNX
        if self.config.export_onnx and ONNX_AVAILABLE:
            path = self._export_onnx(model, sample_input, output_dir, model_name)
            exported_paths['onnx'] = path
        
        # TensorFlow Lite
        if self.config.export_tflite:
            path = self._export_tflite(model, sample_input, output_dir, model_name)
            exported_paths['tflite'] = path
        
        # Core ML (iOS)
        if self.config.export_coreml and COREML_AVAILABLE:
            path = self._export_coreml(model, sample_input, output_dir, model_name)
            exported_paths['coreml'] = path
        
        return exported_paths
    
    def _export_torchscript(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_dir: Path,
        model_name: str
    ) -> Path:
        """Export to TorchScript"""
        model.eval()
        
        # Trace the model
        traced_model = torch.jit.trace(model, sample_input)
        
        # Optimize for mobile
        traced_model = torch.jit.optimize_for_mobile(traced_model)
        
        # Save
        output_path = output_dir / f"{model_name}.pt"
        traced_model.save(str(output_path))
        
        # Also save mobile version
        mobile_path = output_dir / f"{model_name}_mobile.ptl"
        traced_model._save_for_lite_interpreter(str(mobile_path))
        
        logger.info(f"Exported TorchScript model to {output_path}")
        
        return output_path
    
    def _export_onnx(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_dir: Path,
        model_name: str
    ) -> Path:
        """Export to ONNX"""
        model.eval()
        
        output_path = output_dir / f"{model_name}.onnx"
        
        # Export to ONNX
        torch.onnx.export(
            model,
            sample_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['ecg_signal'],
            output_names=['predictions'],
            dynamic_axes={
                'ecg_signal': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        # Optimize ONNX model
        from onnx import optimizer
        optimized_model = optimizer.optimize(onnx_model)
        onnx.save(optimized_model, str(output_path))
        
        logger.info(f"Exported ONNX model to {output_path}")
        
        return output_path
    
    def _export_tflite(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_dir: Path,
        model_name: str
    ) -> Path:
        """Export to TensorFlow Lite"""
        # First export to ONNX
        onnx_path = output_dir / f"{model_name}_temp.onnx"
        
        if not onnx_path.exists():
            self._export_onnx(model, sample_input, output_dir, f"{model_name}_temp")
        
        try:
            import tensorflow as tf
            import onnx2tf
            
            # Convert ONNX to TensorFlow
            tf_model = onnx2tf.convert(str(onnx_path))
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_dataset_gen(sample_input)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            tflite_model = converter.convert()
            
            # Save
            output_path = output_dir / f"{model_name}.tflite"
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Exported TFLite model to {output_path}")
            
            return output_path
            
        except ImportError:
            logger.warning("TensorFlow not available for TFLite export")
            return None
    
    def _export_coreml(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_dir: Path,
        model_name: str
    ) -> Path:
        """Export to Core ML for iOS"""
        model.eval()
        
        # Trace the model
        traced_model = torch.jit.trace(model, sample_input)
        
        # Convert to Core ML
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=sample_input.shape)],
            convert_to="neuralnetwork",
            minimum_deployment_target=ct.target.iOS14
        )
        
        # Add metadata
        mlmodel.author = "CardioAI Pro"
        mlmodel.short_description = "ECG Analysis Model"
        mlmodel.version = "1.0"
        
        # Save
        output_path = output_dir / f"{model_name}.mlmodel"
        mlmodel.save(str(output_path))
        
        logger.info(f"Exported Core ML model to {output_path}")
        
        return output_path
    
    def _representative_dataset_gen(self, sample_input: torch.Tensor):
        """Generate representative dataset for TFLite quantization"""
        def gen():
            for _ in range(100):
                # Generate random variations of sample input
                noise = torch.randn_like(sample_input) * 0.1
                yield [sample_input.numpy() + noise.numpy()]
        return gen


class EdgeInferenceEngine:
    """Optimized inference engine for edge devices"""
    
    def __init__(self, model_path: str, backend: str = "pytorch"):
        self.model_path = model_path
        self.backend = backend
        self.model = self._load_model()
    
    def _load_model(self):
        """Load model based on backend"""
        if self.backend == "pytorch":
            return torch.jit.load(self.model_path)
        elif self.backend == "onnx" and ONNX_AVAILABLE:
            return ort.InferenceSession(self.model_path)
        elif self.backend == "tflite":
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def predict(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Run inference on ECG signal"""
        if self.backend == "pytorch":
            with torch.no_grad():
                input_tensor = torch.from_numpy(ecg_signal).float()
                output = self.model(input_tensor)
                return output['predictions'].numpy()
        
        elif self.backend == "onnx":
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            output = self.model.run([output_name], {input_name: ecg_signal})
            return output[0]
        
        elif self.backend == "tflite":
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            
            self.model.set_tensor(input_details[0]['index'], ecg_signal)
            self.model.invoke()
            
            output = self.model.get_tensor(output_details[0]['index'])
            return output
    
    def benchmark(self, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference performance"""
        import time
        
        # Generate sample input
        sample_input = np.random.randn(1, 12, 5000).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.predict(sample_input)
        
        # Benchmark
        latencies = []
        
        for _ in range(num_runs):
            start = time.time()
            self.predict(sample_input)
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }


def create_edge_optimized_model(
    original_model: HybridECGModel,
    config: Optional[EdgeOptimizationConfig] = None,
    validation_data: Optional[DataLoader] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Create edge-optimized version of ECG model
    
    Args:
        original_model: Original hybrid ECG model
        config: Edge optimization configuration
        validation_data: Data for calibration and validation
        
    Returns:
        Optimized model and optimization metrics
    """
    if config is None:
        config = EdgeOptimizationConfig()
    
    metrics = {}
    
    # 1. Create mobile architecture
    optimizer = MobileArchitectureOptimizer(config)
    model = optimizer.create_mobile_model(original_model.config)
    
    # 2. Transfer knowledge from original model
    # (In practice, use full knowledge distillation training)
    model.load_state_dict(original_model.state_dict(), strict=False)
    
    # 3. Apply pruning
    if config.enable_pruning:
        pruner = ModelPruner(config)
        model, prune_metrics = pruner.prune_model(model)
        metrics.update({'pruning': prune_metrics})
    
    # 4. Apply quantization
    if config.enable_quantization:
        quantizer = ModelQuantizer(config)
        model, quant_metrics = quantizer.quantize_model(model, validation_data)
        metrics.update({'quantization': quant_metrics})
    
    # 5. Export model
    exporter = ModelExporter(config)
    sample_input = torch.randn(1, 12, 5000)
    exported_paths = exporter.export_model(model, sample_input, "edge_models")
    metrics.update({'exported_formats': list(exported_paths.keys())})
    
    # 6. Benchmark performance
    engine = EdgeInferenceEngine(str(exported_paths['torchscript']), backend="pytorch")
    benchmark_results = engine.benchmark()
    metrics.update({'benchmark': benchmark_results})
    
    logger.info(f"Edge optimization complete. Metrics: {metrics}")
    
    return model, metrics
