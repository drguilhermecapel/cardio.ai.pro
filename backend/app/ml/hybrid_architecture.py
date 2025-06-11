"""
Hybrid CNN-BiLSTM-Transformer Architecture for ECG Analysis
Implements the scientific recommendation for advanced neural architecture
Based on Dr. Guilherme Capel's recommendations for CardioAI Pro
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the hybrid model"""
    input_channels: int = 12
    sequence_length: int = 5000
    num_classes: int = 71
    cnn_growth_rate: int = 32
    lstm_hidden_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    dropout_rate: float = 0.2
    ensemble_weights: list[float] | None = None

class FrequencyChannelAttention(nn.Module):
    """
    Frequency Channel Attention mechanism for CNN
    Enhances feature representation by focusing on important frequency components
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        attention = self.sigmoid(avg_out + max_out).unsqueeze(2)
        return x * attention.expand_as(x)

class DenseLayer(nn.Module):
    """Dense layer for DenseNet architecture"""

    def __init__(self, in_channels: int, growth_rate: int, dropout_rate: float = 0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm1d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.bn1(x)))
        new_features = self.conv2(self.relu2(self.bn2(new_features)))
        new_features = self.dropout(new_features)
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Module):
    """Dense block containing multiple dense layers"""

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int, dropout_rate: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate, dropout_rate)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    """Transition layer between dense blocks"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        return self.pool(x)

class DenseNetCNN(nn.Module):
    """
    DenseNet-based CNN for ECG feature extraction
    Target accuracy: 99.6% as specified in recommendations
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        growth_rate = config.cnn_growth_rate
        input_channels = config.input_channels
        dropout_rate = config.dropout_rate

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        num_features = 64

        self.dense_block1 = DenseBlock(num_features, growth_rate, 6, dropout_rate)
        num_features += 6 * growth_rate
        self.transition1 = TransitionLayer(num_features, num_features // 2)
        num_features = num_features // 2

        self.dense_block2 = DenseBlock(num_features, growth_rate, 12, dropout_rate)
        num_features += 12 * growth_rate
        self.transition2 = TransitionLayer(num_features, num_features // 2)
        num_features = num_features // 2

        self.dense_block3 = DenseBlock(num_features, growth_rate, 24, dropout_rate)
        num_features += 24 * growth_rate
        self.transition3 = TransitionLayer(num_features, num_features // 2)
        num_features = num_features // 2

        self.dense_block4 = DenseBlock(num_features, growth_rate, 16, dropout_rate)
        num_features += 16 * growth_rate

        self.attention = FrequencyChannelAttention(num_features)

        self.bn_final = nn.BatchNorm1d(num_features)
        self.relu_final = nn.ReLU(inplace=True)

        self.output_channels = num_features

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))

        x = self.dense_block1(x)
        x = self.transition1(x)

        x = self.dense_block2(x)
        x = self.transition2(x)

        x = self.dense_block3(x)
        x = self.transition3(x)

        x = self.dense_block4(x)

        x = self.attention(x)

        x = self.relu_final(self.bn_final(x))

        return x  # (batch, output_channels, reduced_sequence_length)

class BiLSTMTemporalAnalyzer(nn.Module):
    """
    Bidirectional LSTM for capturing temporal dependencies in ECG signals
    Processes CNN features to understand temporal patterns
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout_rate: float = 0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)

        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        return lstm_out, (hidden, cell)

class MultiHeadTransformerEncoder(nn.Module):
    """
    Multi-head Transformer encoder for spatial-temporal correlation
    8 attention heads as specified in recommendations
    """

    def __init__(self, d_model: int, num_heads: int = 8, num_layers: int = 4, dropout_rate: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.pos_encoding = PositionalEncoding(d_model, dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):

        x = self.pos_encoding(x)

        transformer_out = self.transformer(x, src_key_padding_mask=mask)

        return self.dropout(transformer_out)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class EnsembleVotingSystem(nn.Module):
    """
    Ensemble voting system with majority voting
    Combines predictions from CNN, BiLSTM, and Transformer
    """

    def __init__(self, input_dim: int, num_classes: int, ensemble_weights: list[float] | None = None):
        super().__init__()

        self.num_classes = num_classes

        self.cnn_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_classes)
        )

        self.lstm_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_classes)
        )

        self.transformer_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_classes)
        )

        if ensemble_weights is None:
            self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        else:
            self.register_buffer('ensemble_weights', torch.tensor(ensemble_weights))

        self.meta_classifier = nn.Sequential(
            nn.Linear(num_classes * 3, num_classes * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_classes * 2, num_classes)
        )

    def forward(self, cnn_features, lstm_features, transformer_features):
        cnn_logits = self.cnn_classifier(cnn_features)
        lstm_logits = self.lstm_classifier(lstm_features)
        transformer_logits = self.transformer_classifier(transformer_features)

        weighted_logits = (
            self.ensemble_weights[0] * cnn_logits +
            self.ensemble_weights[1] * lstm_logits +
            self.ensemble_weights[2] * transformer_logits
        )

        concatenated = torch.cat([cnn_logits, lstm_logits, transformer_logits], dim=1)
        meta_logits = self.meta_classifier(concatenated)

        return {
            'cnn_logits': cnn_logits,
            'lstm_logits': lstm_logits,
            'transformer_logits': transformer_logits,
            'weighted_logits': weighted_logits,
            'meta_logits': meta_logits,
            'ensemble_weights': self.ensemble_weights
        }

class MultimodalFusion(nn.Module):
    """
    Multimodal fusion for processing different signal representations
    Processes 1D signals + 2D spectrograms + wavelet representations
    """

    def __init__(self, signal_dim: int, spectrogram_dim: int, wavelet_dim: int, output_dim: int):
        super().__init__()

        self.signal_processor = nn.Sequential(
            nn.Linear(signal_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.spectrogram_processor = nn.Sequential(
            nn.Linear(spectrogram_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.wavelet_processor = nn.Sequential(
            nn.Linear(wavelet_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim)
        )

        self.attention = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, signal_features, spectrogram_features, wavelet_features):
        signal_out = self.signal_processor(signal_features)
        spectrogram_out = self.spectrogram_processor(spectrogram_features)
        wavelet_out = self.wavelet_processor(wavelet_features)

        concatenated = torch.cat([signal_out, spectrogram_out, wavelet_out], dim=1)

        attention_weights = self.attention(concatenated)

        weighted_signal = attention_weights[:, 0:1] * signal_out
        weighted_spectrogram = attention_weights[:, 1:2] * spectrogram_out
        weighted_wavelet = attention_weights[:, 2:3] * wavelet_out

        fused = torch.cat([weighted_signal, weighted_spectrogram, weighted_wavelet], dim=1)
        output = self.fusion(fused)

        return output, attention_weights

class HybridECGModel(nn.Module):
    """
    Complete Hybrid CNN-BiLSTM-Transformer Architecture
    Implements the full pipeline as specified in scientific recommendations
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.cnn = DenseNetCNN(config)

        self.bilstm = BiLSTMTemporalAnalyzer(
            input_dim=self.cnn.output_channels,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=2,
            dropout_rate=config.dropout_rate
        )

        self.transformer = MultiHeadTransformerEncoder(
            d_model=config.lstm_hidden_dim * 2,  # BiLSTM output is bidirectional
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
            dropout_rate=config.dropout_rate
        )

        self.multimodal_fusion = MultimodalFusion(
            signal_dim=config.lstm_hidden_dim * 2,
            spectrogram_dim=config.lstm_hidden_dim * 2,
            wavelet_dim=config.lstm_hidden_dim * 2,
            output_dim=config.lstm_hidden_dim * 2
        )

        self.ensemble = EnsembleVotingSystem(
            input_dim=config.lstm_hidden_dim * 2,
            num_classes=config.num_classes,
            ensemble_weights=config.ensemble_weights
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        self.final_classifier = nn.Linear(config.num_classes, config.num_classes)

    def forward(self, x, return_features=False):
        batch_size = x.size(0)

        cnn_features = self.cnn(x)  # (batch, channels, seq_len)

        cnn_features_transposed = cnn_features.transpose(1, 2)  # (batch, seq_len, channels)

        lstm_out, (hidden, cell) = self.bilstm(cnn_features_transposed)

        transformer_out = self.transformer(lstm_out)

        cnn_pooled_avg = self.global_avg_pool(cnn_features).squeeze(-1)
        cnn_pooled_max = self.global_max_pool(cnn_features).squeeze(-1)
        cnn_pooled = (cnn_pooled_avg + cnn_pooled_max) / 2

        lstm_features = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)

        transformer_features = transformer_out.mean(dim=1)  # (batch, hidden_dim * 2)

        fused_features, modality_weights = self.multimodal_fusion(
            transformer_features, transformer_features, transformer_features
        )

        ensemble_results = self.ensemble(cnn_pooled, lstm_features, transformer_features)

        final_logits = self.final_classifier(ensemble_results['meta_logits'])

        if return_features:
            return {
                'logits': final_logits,
                'cnn_logits': ensemble_results['cnn_logits'],
                'lstm_logits': ensemble_results['lstm_logits'],
                'transformer_logits': ensemble_results['transformer_logits'],
                'weighted_logits': ensemble_results['weighted_logits'],
                'meta_logits': ensemble_results['meta_logits'],
                'features': {
                    'cnn': cnn_pooled,
                    'lstm': lstm_features,
                    'transformer': transformer_features,
                    'fused': fused_features
                },
                'attention_weights': {
                    'ensemble': ensemble_results['ensemble_weights'],
                    'modality': modality_weights
                }
            }
        else:
            return final_logits

    def get_attention_maps(self, x):
        """Extract attention maps for interpretability"""
        with torch.no_grad():
            cnn_features = self.cnn(x)
            cnn_features_transposed = cnn_features.transpose(1, 2)
            lstm_out, _ = self.bilstm(cnn_features_transposed)
            transformer_out = self.transformer(lstm_out)

            attention_maps = {}

            cnn_attention = self.cnn.attention
            b, c, seq_len = cnn_features.size()
            avg_out = cnn_attention.fc(cnn_attention.avg_pool(cnn_features).view(b, c))
            max_out = cnn_attention.fc(cnn_attention.max_pool(cnn_features).view(b, c))
            channel_attention = torch.sigmoid(avg_out + max_out)

            attention_maps['cnn_channel_attention'] = channel_attention.cpu().numpy()

            transformer_attention = torch.mean(torch.abs(transformer_out), dim=-1)
            attention_maps['transformer_temporal_attention'] = transformer_attention.cpu().numpy()

            return attention_maps

    def count_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self):
        """Get model architecture summary"""
        total_params = self.count_parameters()

        summary = {
            'total_parameters': total_params,
            'cnn_parameters': sum(p.numel() for p in self.cnn.parameters() if p.requires_grad),
            'lstm_parameters': sum(p.numel() for p in self.bilstm.parameters() if p.requires_grad),
            'transformer_parameters': sum(p.numel() for p in self.transformer.parameters() if p.requires_grad),
            'ensemble_parameters': sum(p.numel() for p in self.ensemble.parameters() if p.requires_grad),
            'config': self.config
        }

        return summary

def create_hybrid_model(
    num_classes: int = 71,
    input_channels: int = 12,
    sequence_length: int = 5000,
    **kwargs
) -> HybridECGModel:
    """
    Factory function to create a hybrid ECG model

    Args:
        num_classes: Number of ECG conditions to classify (default: 71 SCP-ECG conditions)
        input_channels: Number of ECG leads (default: 12)
        sequence_length: Length of ECG signal (default: 5000 samples)
        **kwargs: Additional configuration parameters

    Returns:
        HybridECGModel: Configured hybrid model
    """

    config = ModelConfig(
        input_channels=input_channels,
        sequence_length=sequence_length,
        num_classes=num_classes,
        **kwargs
    )

    model = HybridECGModel(config)

    logger.info(f"Created hybrid ECG model with {model.count_parameters():,} parameters")
    logger.info(f"Model configuration: {config}")

    return model

def load_pretrained_model(checkpoint_path: str, config: ModelConfig | None = None) -> HybridECGModel:
    """
    Load a pretrained hybrid model

    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration (if None, will be loaded from checkpoint)

    Returns:
        HybridECGModel: Loaded model
    """

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if config is None:
        config = checkpoint.get('config', ModelConfig())

    model = HybridECGModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Loaded pretrained model from {checkpoint_path}")

    return model

if __name__ == "__main__":
    model = create_hybrid_model()

    summary = model.get_model_summary()
    print(f"Model created with {summary['total_parameters']:,} parameters")
    print(f"CNN: {summary['cnn_parameters']:,}")
    print(f"LSTM: {summary['lstm_parameters']:,}")
    print(f"Transformer: {summary['transformer_parameters']:,}")
    print(f"Ensemble: {summary['ensemble_parameters']:,}")

    batch_size = 2
    sequence_length = 5000
    input_channels = 12

    x = torch.randn(batch_size, input_channels, sequence_length)

    with torch.no_grad():
        output = model(x, return_features=True)

    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"CNN features shape: {output['features']['cnn'].shape}")
    print(f"LSTM features shape: {output['features']['lstm'].shape}")
    print(f"Transformer features shape: {output['features']['transformer'].shape}")

    attention_maps = model.get_attention_maps(x)
    print(f"CNN channel attention shape: {attention_maps['cnn_channel_attention'].shape}")
    print(f"Transformer temporal attention shape: {attention_maps['transformer_temporal_attention'].shape}")
