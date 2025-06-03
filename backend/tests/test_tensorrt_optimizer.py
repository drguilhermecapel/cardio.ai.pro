"""
Tests for TensorRT Optimization Service
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from app.services.tensorrt_optimizer import TensorRTOptimizer, TensorRTModelManager


class TestTensorRTOptimizer:
    """Test TensorRT optimization functionality"""

    def test_init_without_tensorrt(self):
        """Test initialization when TensorRT is not available"""
        with patch('app.services.tensorrt_optimizer.TENSORRT_AVAILABLE', False):
            optimizer = TensorRTOptimizer()
            assert optimizer.logger_trt is None
            assert optimizer.stream is None

    @patch('app.services.tensorrt_optimizer.TENSORRT_AVAILABLE', True)
    def test_init_with_tensorrt(self):
        """Test initialization when TensorRT is available"""
        with patch('app.services.tensorrt_optimizer.trt') as mock_trt, \
             patch('app.services.tensorrt_optimizer.cuda') as mock_cuda:
            
            mock_trt.Logger.return_value = Mock()
            mock_cuda.Stream.return_value = Mock()
            
            optimizer = TensorRTOptimizer()
            assert optimizer.logger_trt is not None
            assert optimizer.stream is not None

    @patch('app.services.tensorrt_optimizer.TENSORRT_AVAILABLE', False)
    def test_optimize_onnx_model_without_tensorrt(self):
        """Test ONNX optimization when TensorRT is not available"""
        optimizer = TensorRTOptimizer()
        result = optimizer.optimize_onnx_model("test.onnx", "test.trt")
        assert result is False

    @patch('app.services.tensorrt_optimizer.TENSORRT_AVAILABLE', True)
    def test_optimize_onnx_model_success(self):
        """Test successful ONNX to TensorRT optimization"""
        with patch('app.services.tensorrt_optimizer.trt') as mock_trt, \
             patch('builtins.open', create=True) as mock_open:
            
            mock_builder = Mock()
            mock_config = Mock()
            mock_network = Mock()
            mock_parser = Mock()
            mock_engine = Mock()
            
            mock_trt.Builder.return_value = mock_builder
            mock_builder.create_builder_config.return_value = mock_config
            mock_builder.create_network.return_value = mock_network
            mock_builder.platform_has_fast_fp16 = True
            mock_builder.build_engine.return_value = mock_engine
            
            mock_trt.OnnxParser.return_value = mock_parser
            mock_parser.parse.return_value = True
            mock_parser.num_errors = 0
            
            mock_engine.serialize.return_value = b"serialized_engine"
            
            mock_file = Mock()
            mock_file.read.return_value = b"onnx_model_data"
            mock_open.return_value.__enter__.return_value = mock_file
            
            optimizer = TensorRTOptimizer()
            result = optimizer.optimize_onnx_model("test.onnx", "test.trt")
            
            assert result is True
            mock_builder.create_builder_config.assert_called_once()
            mock_parser.parse.assert_called_once()

    @patch('app.services.tensorrt_optimizer.TENSORRT_AVAILABLE', True)
    def test_optimize_onnx_model_parse_failure(self):
        """Test ONNX optimization with parsing failure"""
        with patch('app.services.tensorrt_optimizer.trt') as mock_trt, \
             patch('builtins.open', create=True):
            
            mock_builder = Mock()
            mock_config = Mock()
            mock_network = Mock()
            mock_parser = Mock()
            
            mock_trt.Builder.return_value = mock_builder
            mock_builder.create_builder_config.return_value = mock_config
            mock_builder.create_network.return_value = mock_network
            
            mock_trt.OnnxParser.return_value = mock_parser
            mock_parser.parse.return_value = False
            mock_parser.num_errors = 1
            mock_parser.get_error.return_value = "Parse error"
            
            optimizer = TensorRTOptimizer()
            result = optimizer.optimize_onnx_model("test.onnx", "test.trt")
            
            assert result is False

    @patch('app.services.tensorrt_optimizer.TENSORRT_AVAILABLE', False)
    def test_load_engine_without_tensorrt(self):
        """Test engine loading when TensorRT is not available"""
        optimizer = TensorRTOptimizer()
        result = optimizer.load_engine("test.trt", "test_model")
        assert result is False

    @patch('app.services.tensorrt_optimizer.TENSORRT_AVAILABLE', True)
    def test_load_engine_success(self):
        """Test successful engine loading"""
        with patch('app.services.tensorrt_optimizer.trt') as mock_trt, \
             patch('builtins.open', create=True) as mock_open:
            
            mock_runtime = Mock()
            mock_engine = Mock()
            mock_context = Mock()
            
            mock_trt.Runtime.return_value = mock_runtime
            mock_runtime.deserialize_cuda_engine.return_value = mock_engine
            mock_engine.create_execution_context.return_value = mock_context
            
            mock_engine.__iter__ = Mock(return_value=iter(['input', 'output']))
            mock_engine.get_binding_shape.return_value = [1, 12, 5000]
            mock_engine.max_batch_size = 1
            mock_engine.get_binding_dtype.return_value = Mock()
            mock_engine.binding_is_input.side_effect = lambda x: x == 'input'
            
            mock_trt.volume.return_value = 60000
            mock_trt.nptype.return_value = np.float32
            
            with patch('app.services.tensorrt_optimizer.cuda') as mock_cuda:
                mock_cuda.pagelocked_empty.return_value = np.zeros(60000, dtype=np.float32)
                mock_cuda.mem_alloc.return_value = Mock()
                
                optimizer = TensorRTOptimizer()
                result = optimizer.load_engine("test.trt", "test_model")
                
                assert result is True
                assert "test_model" in optimizer.engines
                assert "test_model" in optimizer.contexts

    def test_benchmark_model_without_tensorrt(self):
        """Test model benchmarking when TensorRT is not available"""
        optimizer = TensorRTOptimizer()
        result = optimizer.benchmark_model("test_model", (1, 12, 5000))
        assert "error" in result

    def test_get_engine_info_not_loaded(self):
        """Test getting engine info for non-loaded model"""
        optimizer = TensorRTOptimizer()
        result = optimizer.get_engine_info("test_model")
        assert "error" in result

    @patch('app.services.tensorrt_optimizer.TENSORRT_AVAILABLE', True)
    def test_cleanup(self):
        """Test GPU resource cleanup"""
        with patch('app.services.tensorrt_optimizer.cuda'):
            optimizer = TensorRTOptimizer()
            
            mock_device_mem = Mock()
            optimizer.cuda_inputs = {"test": [{"device": mock_device_mem}]}
            optimizer.cuda_outputs = {"test": [{"device": mock_device_mem}]}
            optimizer.engines = {"test": Mock()}
            optimizer.contexts = {"test": Mock()}
            optimizer.stream = Mock()
            
            optimizer.cleanup()
            
            assert len(optimizer.cuda_inputs) == 0
            assert len(optimizer.cuda_outputs) == 0
            assert len(optimizer.engines) == 0
            assert len(optimizer.contexts) == 0


class TestTensorRTModelManager:
    """Test TensorRT model manager functionality"""

    def test_init(self):
        """Test model manager initialization"""
        with patch('pathlib.Path.mkdir'):
            manager = TensorRTModelManager("/test/models")
            assert str(manager.models_dir) == "/test/models"
            assert "ecg_classifier" in manager.model_configs

    def test_optimize_all_models_no_files(self):
        """Test optimizing models when ONNX files don't exist"""
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=False):
            
            manager = TensorRTModelManager("/test/models")
            results = manager.optimize_all_models()
            
            for model_name in manager.model_configs.keys():
                assert results[model_name] is False

    def test_optimize_all_models_success(self):
        """Test successful model optimization"""
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=True):
            
            manager = TensorRTModelManager("/test/models")
            manager.optimizer.optimize_onnx_model = Mock(return_value=True)
            manager.optimizer.load_engine = Mock(return_value=True)
            
            results = manager.optimize_all_models()
            
            for model_name in manager.model_configs.keys():
                assert results[model_name] is True

    def test_load_optimized_models_no_engines(self):
        """Test loading models when engine files don't exist"""
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=False):
            
            manager = TensorRTModelManager("/test/models")
            results = manager.load_optimized_models()
            
            for model_name in manager.model_configs.keys():
                assert results[model_name] is False

    def test_load_optimized_models_success(self):
        """Test successful model loading"""
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=True):
            
            manager = TensorRTModelManager("/test/models")
            manager.optimizer.load_engine = Mock(return_value=True)
            
            results = manager.load_optimized_models()
            
            for model_name in manager.model_configs.keys():
                assert results[model_name] is True

    def test_benchmark_all_models(self):
        """Test benchmarking all models"""
        with patch('pathlib.Path.mkdir'):
            manager = TensorRTModelManager("/test/models")
            manager.optimizer.engines = {"ecg_classifier": Mock()}
            manager.optimizer.benchmark_model = Mock(return_value={
                "avg_inference_time_ms": 25.0,
                "throughput_fps": 40.0
            })
            
            results = manager.benchmark_all_models()
            
            assert "ecg_classifier" in results
            assert results["ecg_classifier"]["avg_inference_time_ms"] == 25.0

    def test_get_status(self):
        """Test getting manager status"""
        with patch('pathlib.Path.mkdir'):
            manager = TensorRTModelManager("/test/models")
            manager.optimizer.engines = {"test_model": Mock()}
            
            status = manager.get_status()
            
            assert "tensorrt_available" in status
            assert "loaded_engines" in status
            assert "model_configs" in status
            assert status["loaded_engines"] == ["test_model"]
