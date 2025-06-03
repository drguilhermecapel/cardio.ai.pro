"""
Tests for ECG Foundation Model
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.services.ecg_foundation_model import (
    ECGFoundationModel, 
    ECGTransformer, 
    ECGConvNet, 
    ECGDataset
)


class TestECGDataset:
    """Test ECG dataset functionality"""

    def test_dataset_creation(self):
        """Test dataset creation with ECG data"""
        ecg_data = np.random.randn(10, 5000, 12).astype(np.float32)
        labels = np.random.randint(0, 71, 10).astype(np.int64)
        
        dataset = ECGDataset(ecg_data, labels)
        
        assert len(dataset) == 10
        
        sample = dataset[0]
        assert "ecg" in sample
        assert "label" in sample
        assert sample["ecg"].shape == (5000, 12)

    def test_dataset_without_labels(self):
        """Test dataset creation without labels"""
        ecg_data = np.random.randn(5, 5000, 12).astype(np.float32)
        
        dataset = ECGDataset(ecg_data)
        
        assert len(dataset) == 5
        
        sample = dataset[0]
        assert "ecg" in sample
        assert "label" not in sample


@patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', True)
class TestECGTransformer:
    """Test ECG Transformer model"""

    def test_transformer_initialization(self):
        """Test transformer model initialization"""
        with patch('app.services.ecg_foundation_model.torch') as mock_torch:
            mock_torch.nn = Mock()
            mock_torch.zeros.return_value = Mock()
            mock_torch.arange.return_value = Mock()
            mock_torch.exp.return_value = Mock()
            mock_torch.sin.return_value = Mock()
            mock_torch.cos.return_value = Mock()
            
            model = ECGTransformer(
                input_dim=12,
                sequence_length=5000,
                d_model=512,
                nhead=8,
                num_layers=6,
                num_classes=71
            )
            
            assert model.input_dim == 12
            assert model.sequence_length == 5000
            assert model.d_model == 512

    def test_transformer_forward_pass(self):
        """Test transformer forward pass"""
        with patch('app.services.ecg_foundation_model.torch') as mock_torch:
            mock_tensor = Mock()
            mock_tensor.shape = (2, 5000, 12)
            mock_tensor.device = "cpu"
            mock_tensor.mean.return_value = Mock()
            
            mock_torch.zeros.return_value = Mock()
            mock_torch.arange.return_value = Mock()
            mock_torch.exp.return_value = Mock()
            mock_torch.sin.return_value = Mock()
            mock_torch.cos.return_value = Mock()
            
            model = ECGTransformer()
            
            model.input_projection = Mock(return_value=mock_tensor)
            model.transformer = Mock(return_value=mock_tensor)
            model.classifier = Mock(return_value=mock_tensor)
            model.positional_encoding = Mock()
            model.positional_encoding.device = "cpu"
            model.positional_encoding.to.return_value = model.positional_encoding
            
            result = model.forward(mock_tensor)
            
            assert "logits" in result
            assert "embeddings" in result
            assert "encoded_sequence" in result


@patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', True)
class TestECGConvNet:
    """Test ECG ConvNet model"""

    def test_convnet_initialization(self):
        """Test ConvNet model initialization"""
        with patch('app.services.ecg_foundation_model.torch') as mock_torch:
            mock_torch.nn = Mock()
            
            model = ECGConvNet(
                input_channels=12,
                num_classes=71
            )
            
            assert len(model.conv_blocks) == 5

    def test_convnet_forward_pass(self):
        """Test ConvNet forward pass"""
        with patch('app.services.ecg_foundation_model.torch') as mock_torch:
            mock_tensor = Mock()
            mock_tensor.transpose.return_value = mock_tensor
            mock_tensor.squeeze.return_value = mock_tensor
            
            model = ECGConvNet()
            
            for i, conv_block in enumerate(model.conv_blocks):
                model.conv_blocks[i] = Mock(return_value=mock_tensor)
            
            model.global_pool = Mock(return_value=mock_tensor)
            model.classifier = Mock(return_value=mock_tensor)
            
            result = model.forward(mock_tensor)
            
            assert "logits" in result
            assert "embeddings" in result


class TestECGFoundationModel:
    """Test ECG Foundation Model functionality"""

    def test_init_without_torch(self):
        """Test initialization when PyTorch is not available"""
        with patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', False):
            model = ECGFoundationModel()
            
            assert model.model is None
            assert model.device == "cpu"

    @patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', True)
    def test_init_with_torch(self):
        """Test initialization when PyTorch is available"""
        with patch('app.services.ecg_foundation_model.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device.return_value = "cuda"
            
            model = ECGFoundationModel()
            
            assert model.model_type == "transformer"
            assert len(model.condition_names) == 71

    @patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', False)
    async def test_load_model_without_torch(self):
        """Test model loading when PyTorch is not available"""
        model = ECGFoundationModel()
        
        result = await model.load_model()
        
        assert result is False

    @patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', True)
    async def test_load_model_transformer(self):
        """Test loading transformer model"""
        with patch('app.services.ecg_foundation_model.torch') as mock_torch, \
             patch('app.services.ecg_foundation_model.ECGTransformer') as mock_transformer:
            
            mock_model = Mock()
            mock_transformer.return_value = mock_model
            mock_torch.device.return_value = "cpu"
            
            model = ECGFoundationModel(model_type="transformer")
            
            result = await model.load_model()
            
            assert result is True
            assert model.model is not None

    @patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', True)
    async def test_load_model_convnet(self):
        """Test loading ConvNet model"""
        with patch('app.services.ecg_foundation_model.torch') as mock_torch, \
             patch('app.services.ecg_foundation_model.ECGConvNet') as mock_convnet:
            
            mock_model = Mock()
            mock_convnet.return_value = mock_model
            mock_torch.device.return_value = "cpu"
            
            model = ECGFoundationModel(model_type="convnet")
            
            result = await model.load_model()
            
            assert result is True
            assert model.model is not None

    async def test_analyze_ecg_without_model(self):
        """Test ECG analysis when model is not loaded"""
        model = ECGFoundationModel()
        
        ecg_data = np.random.randn(1, 5000, 12).astype(np.float32)
        
        with pytest.raises(RuntimeError):
            await model.analyze_ecg(ecg_data)

    @patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', True)
    async def test_analyze_ecg_success(self):
        """Test successful ECG analysis"""
        with patch('app.services.ecg_foundation_model.torch') as mock_torch, \
             patch('app.services.ecg_foundation_model.DataLoader') as mock_dataloader:
            
            mock_model = Mock()
            mock_outputs = {
                "logits": Mock(),
                "embeddings": Mock()
            }
            mock_model.return_value = mock_outputs
            
            mock_logits = Mock()
            mock_logits.cpu.return_value.numpy.return_value = np.random.rand(1, 71)
            mock_outputs["logits"] = mock_logits
            
            mock_embeddings = Mock()
            mock_embeddings.cpu.return_value.numpy.return_value = np.random.rand(1, 512)
            mock_outputs["embeddings"] = mock_embeddings
            
            mock_torch.softmax.return_value = mock_logits
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            
            mock_batch = {"ecg": Mock()}
            mock_batch["ecg"].to.return_value = mock_batch["ecg"]
            mock_dataloader.return_value = [mock_batch]
            
            model = ECGFoundationModel()
            model.model = mock_model
            model.device = "cpu"
            
            ecg_data = np.random.randn(1, 5000, 12).astype(np.float32)
            
            result = await model.analyze_ecg(ecg_data)
            
            assert "predictions" in result
            assert "top_conditions" in result
            assert "overall_confidence" in result
            assert "processing_time" in result

    def test_get_model_info_without_torch(self):
        """Test getting model info when PyTorch is not available"""
        with patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', False):
            model = ECGFoundationModel()
            
            info = model.get_model_info()
            
            assert "error" in info

    @patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', True)
    def test_get_model_info_with_model(self):
        """Test getting model info when model is loaded"""
        with patch('app.services.ecg_foundation_model.torch'):
            model = ECGFoundationModel()
            
            mock_model = Mock()
            mock_param = Mock()
            mock_param.numel.return_value = 1000
            mock_param.requires_grad = True
            mock_model.parameters.return_value = [mock_param, mock_param]
            
            model.model = mock_model
            
            info = model.get_model_info()
            
            assert "total_parameters" in info
            assert "trainable_parameters" in info
            assert "model_size_mb" in info
            assert info["total_parameters"] == 2000
            assert info["trainable_parameters"] == 2000

    @patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', True)
    async def test_save_model(self):
        """Test model saving"""
        with patch('app.services.ecg_foundation_model.torch') as mock_torch:
            mock_model = Mock()
            mock_model.state_dict.return_value = {"param": "value"}
            
            model = ECGFoundationModel()
            model.model = mock_model
            
            result = await model.save_model("/tmp/test_model.pth")
            
            assert result is True
            mock_torch.save.assert_called_once()

    async def test_save_model_without_torch(self):
        """Test model saving when PyTorch is not available"""
        with patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', False):
            model = ECGFoundationModel()
            
            result = await model.save_model("/tmp/test_model.pth")
            
            assert result is False

    @patch('app.services.ecg_foundation_model.TORCH_AVAILABLE', True)
    async def test_extract_features(self):
        """Test feature extraction"""
        with patch('app.services.ecg_foundation_model.torch') as mock_torch:
            mock_model = Mock()
            mock_model.named_modules.return_value = [("layer1", Mock()), ("layer2", Mock())]
            
            mock_outputs = {"logits": Mock(), "embeddings": Mock()}
            mock_model.return_value = mock_outputs
            
            for key, value in mock_outputs.items():
                value.cpu.return_value.numpy.return_value = np.random.rand(1, 100)
            
            mock_torch.FloatTensor.return_value.to.return_value = Mock()
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            
            model = ECGFoundationModel()
            model.model = mock_model
            model.device = "cpu"
            
            ecg_data = np.random.randn(5000, 12).astype(np.float32)
            
            result = await model.extract_features(ecg_data)
            
            assert "features" in result
            assert "feature_names" in result
            assert "input_shape" in result
