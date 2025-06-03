"""
TensorRT Optimization Service for ECG Models
Converts ONNX models to TensorRT for sub-100ms inference
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available. Falling back to ONNX Runtime.")


class TensorRTOptimizer:
    """TensorRT optimization service for ECG models"""
    
    def __init__(self) -> None:
        self.logger_trt = trt.Logger(trt.Logger.WARNING) if TENSORRT_AVAILABLE else None
        self.engines: Dict[str, Any] = {}
        self.contexts: Dict[str, Any] = {}
        self.cuda_inputs: Dict[str, Any] = {}
        self.cuda_outputs: Dict[str, Any] = {}
        self.stream: Optional[Any] = None
        
        if TENSORRT_AVAILABLE:
            self.stream = cuda.Stream()
            
    def optimize_onnx_model(
        self, 
        onnx_path: str, 
        engine_path: str,
        max_batch_size: int = 32,
        workspace_size: int = 1 << 30,  # 1GB
        precision: str = "fp16"
    ) -> bool:
        """Convert ONNX model to optimized TensorRT engine"""
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available, skipping optimization")
            return False
            
        try:
            builder = trt.Builder(self.logger_trt)
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size
            
            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Enabled FP16 precision for faster inference")
            elif precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("Enabled INT8 precision for maximum speed")
                
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self.logger_trt)
            
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                    return False
                    
            profile = builder.create_optimization_profile()
            
            input_tensor = network.get_input(0)
            input_shape = input_tensor.shape
            
            if input_shape[0] == -1:
                profile.set_shape(
                    input_tensor.name,
                    (1, input_shape[1], input_shape[2]),  # min
                    (max_batch_size // 2, input_shape[1], input_shape[2]),  # opt
                    (max_batch_size, input_shape[1], input_shape[2])  # max
                )
                config.add_optimization_profile(profile)
                
            engine = builder.build_engine(network, config)
            if engine:
                with open(engine_path, 'wb') as f:
                    f.write(engine.serialize())
                logger.info(f"TensorRT engine saved to {engine_path}")
                return True
            else:
                logger.error("Failed to build TensorRT engine")
                return False
                
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            return False
            
    def load_engine(self, engine_path: str, model_name: str) -> bool:
        """Load TensorRT engine for inference"""
        if not TENSORRT_AVAILABLE:
            return False
            
        try:
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(self.logger_trt)
                engine = runtime.deserialize_cuda_engine(f.read())
                
            if engine is None:
                logger.error(f"Failed to load engine from {engine_path}")
                return False
                
            context = engine.create_execution_context()
            self.engines[model_name] = engine
            self.contexts[model_name] = context
            
            self._allocate_buffers(model_name)
            
            logger.info(f"TensorRT engine loaded: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            return False
            
    def _allocate_buffers(self, model_name: str) -> None:
        """Allocate GPU memory buffers for inference"""
        engine = self.engines[model_name]
        
        inputs = []
        outputs = []
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                
        self.cuda_inputs[model_name] = inputs
        self.cuda_outputs[model_name] = outputs
        
    def infer(
        self, 
        model_name: str, 
        input_data: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Run inference using TensorRT engine"""
        if not TENSORRT_AVAILABLE or model_name not in self.engines:
            raise RuntimeError(f"TensorRT engine not available for {model_name}")
            
        try:
            context = self.contexts[model_name]
            inputs = self.cuda_inputs[model_name]
            outputs = self.cuda_outputs[model_name]
            
            np.copyto(inputs[0]['host'], input_data.ravel())
            cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], self.stream)
            
            context.execute_async_v2(
                bindings=[inputs[0]['device'], outputs[0]['device']],
                stream_handle=self.stream.handle
            )
            
            cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], self.stream)
            self.stream.synchronize()
            
            engine = self.engines[model_name]
            output_shape = engine.get_binding_shape(1)  # Assuming single output
            return outputs[0]['host'].reshape(output_shape)
            
        except Exception as e:
            logger.error(f"TensorRT inference failed: {e}")
            raise
            
    def benchmark_model(
        self, 
        model_name: str, 
        input_shape: tuple[int, ...],
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark TensorRT model performance"""
        if not TENSORRT_AVAILABLE or model_name not in self.engines:
            return {"error": "TensorRT not available"}
            
        import time
        
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        for _ in range(10):
            self.infer(model_name, input_data)
            
        start_time = time.time()
        for _ in range(num_iterations):
            self.infer(model_name, input_data)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_fps": throughput,
            "total_time_s": total_time,
            "iterations": num_iterations
        }
        
    def get_engine_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about loaded TensorRT engine"""
        if model_name not in self.engines:
            return {"error": "Engine not loaded"}
            
        engine = self.engines[model_name]
        
        info = {
            "max_batch_size": engine.max_batch_size,
            "num_bindings": engine.num_bindings,
            "has_implicit_batch_dimension": engine.has_implicit_batch_dimension,
            "bindings": []
        }
        
        for i in range(engine.num_bindings):
            binding_info = {
                "name": engine.get_binding_name(i),
                "shape": engine.get_binding_shape(i),
                "dtype": str(engine.get_binding_dtype(i)),
                "is_input": engine.binding_is_input(i)
            }
            info["bindings"].append(binding_info)
            
        return info
        
    def cleanup(self) -> None:
        """Clean up GPU resources"""
        if TENSORRT_AVAILABLE:
            for model_name in list(self.cuda_inputs.keys()):
                for inp in self.cuda_inputs[model_name]:
                    inp['device'].free()
                for out in self.cuda_outputs[model_name]:
                    out['device'].free()
                    
            self.cuda_inputs.clear()
            self.cuda_outputs.clear()
            self.engines.clear()
            self.contexts.clear()
            
            if self.stream:
                self.stream.synchronize()


class TensorRTModelManager:
    """Manager for TensorRT optimized ECG models"""
    
    def __init__(self, models_dir: str = "/app/models") -> None:
        self.models_dir = Path(models_dir)
        self.tensorrt_dir = self.models_dir / "tensorrt"
        self.tensorrt_dir.mkdir(exist_ok=True)
        
        self.optimizer = TensorRTOptimizer()
        self.model_configs = {
            "ecg_classifier": {
                "onnx_path": "ecg_classifier.onnx",
                "input_shape": (1, 12, 5000),
                "precision": "fp16"
            },
            "rhythm_detector": {
                "onnx_path": "rhythm_detector.onnx", 
                "input_shape": (1, 12, 5000),
                "precision": "fp16"
            },
            "quality_assessor": {
                "onnx_path": "quality_assessor.onnx",
                "input_shape": (1, 12, 5000),
                "precision": "fp16"
            }
        }
        
    def optimize_all_models(self) -> Dict[str, bool]:
        """Optimize all ECG models to TensorRT"""
        results = {}
        
        for model_name, config in self.model_configs.items():
            onnx_path = self.models_dir / config["onnx_path"]
            engine_path = self.tensorrt_dir / f"{model_name}.trt"
            
            if onnx_path.exists():
                success = self.optimizer.optimize_onnx_model(
                    str(onnx_path),
                    str(engine_path),
                    precision=config["precision"]
                )
                results[model_name] = success
                
                if success:
                    self.optimizer.load_engine(str(engine_path), model_name)
            else:
                logger.warning(f"ONNX model not found: {onnx_path}")
                results[model_name] = False
                
        return results
        
    def load_optimized_models(self) -> Dict[str, bool]:
        """Load pre-optimized TensorRT engines"""
        results = {}
        
        for model_name in self.model_configs.keys():
            engine_path = self.tensorrt_dir / f"{model_name}.trt"
            
            if engine_path.exists():
                success = self.optimizer.load_engine(str(engine_path), model_name)
                results[model_name] = success
            else:
                results[model_name] = False
                
        return results
        
    def benchmark_all_models(self) -> Dict[str, Dict[str, float]]:
        """Benchmark all loaded TensorRT models"""
        results = {}
        
        for model_name, config in self.model_configs.items():
            if model_name in self.optimizer.engines:
                benchmark = self.optimizer.benchmark_model(
                    model_name, 
                    config["input_shape"]
                )
                results[model_name] = benchmark
                
        return results
        
    def get_status(self) -> Dict[str, Any]:
        """Get status of TensorRT optimization"""
        return {
            "tensorrt_available": TENSORRT_AVAILABLE,
            "models_dir": str(self.models_dir),
            "tensorrt_dir": str(self.tensorrt_dir),
            "loaded_engines": list(self.optimizer.engines.keys()),
            "model_configs": self.model_configs
        }
