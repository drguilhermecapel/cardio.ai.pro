"""
ECG Foundation Model Integration
Implements large-scale pre-trained models for comprehensive ECG analysis
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Foundation model features disabled.")

try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Pre-trained model features limited.")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("TIMM not available. Vision transformer features limited.")


class ECGDataset(Dataset):
    """Dataset class for ECG data processing"""
    
    def __init__(
        self, 
        ecg_data: npt.NDArray[np.float32], 
        labels: Optional[npt.NDArray[np.int64]] = None,
        transform: Optional[Any] = None
    ):
        self.ecg_data = ecg_data
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.ecg_data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {
            "ecg": torch.FloatTensor(self.ecg_data[idx])
        }
        
        if self.labels is not None:
            sample["label"] = torch.LongTensor([self.labels[idx]])
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class ECGTransformer(nn.Module):
    """Transformer-based ECG analysis model"""
    
    def __init__(
        self,
        input_dim: int = 12,
        sequence_length: int = 5000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        num_classes: int = 71,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.positional_encoding = self._create_positional_encoding(sequence_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        batch_size, seq_len, input_dim = x.shape
        
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        if x.device != self.positional_encoding.device:
            self.positional_encoding = self.positional_encoding.to(x.device)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        encoded = self.transformer(x)  # [batch, seq_len, d_model]
        
        pooled = encoded.mean(dim=1)  # [batch, d_model]
        
        logits = self.classifier(pooled)  # [batch, num_classes]
        
        return {
            "logits": logits,
            "embeddings": pooled,
            "encoded_sequence": encoded
        }


class ECGConvNet(nn.Module):
    """Convolutional Neural Network for ECG analysis"""
    
    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 71,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(input_channels, 64, kernel_size=15, stride=2),
            self._make_conv_block(64, 128, kernel_size=11, stride=2),
            self._make_conv_block(128, 256, kernel_size=7, stride=2),
            self._make_conv_block(256, 512, kernel_size=5, stride=2),
            self._make_conv_block(512, 1024, kernel_size=3, stride=2),
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def _make_conv_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int
    ) -> nn.Module:
        """Create a convolutional block"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        x = x.transpose(1, 2)
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            
        pooled = self.global_pool(x).squeeze(-1)  # [batch, 1024]
        
        logits = self.classifier(pooled)
        
        return {
            "logits": logits,
            "embeddings": pooled
        }


class ECGFoundationModel:
    """Foundation model for comprehensive ECG analysis"""
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        device: str = "auto",
        model_type: str = "transformer"
    ):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model_type = model_type
        
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.config: Dict[str, Any] = {}
        
        self.model_configs = {
            "transformer": {
                "input_dim": 12,
                "sequence_length": 5000,
                "d_model": 512,
                "nhead": 8,
                "num_layers": 6,
                "num_classes": 71,
                "dropout": 0.1
            },
            "convnet": {
                "input_channels": 12,
                "num_classes": 71,
                "dropout": 0.1
            }
        }
        
        self.condition_names = [
            "NORM", "MI", "STTC", "CD", "HYP", "PAC", "PVC", "AFIB", "AFLT", "SVTAC",
            "PSVT", "BIGU", "TRIGU", "VT", "VFL", "VF", "ASYS", "BRADY", "TACHY", "SINUS",
            "SA", "SB", "SR", "PACE", "RBBB", "LBBB", "LAD", "RAD", "LAFB", "LPFB",
            "LVH", "RVH", "QWAVE", "LOWT", "NT", "PAD", "PRC", "LPR", "INVT", "LVOLT",
            "HVOLT", "TAB", "STE", "STD", "NST", "DIG", "LNGQT", "NORM", "IMI", "AMI",
            "LMI", "ILMI", "LAO", "LMO", "PMI", "IPLMI", "IPMI", "ALMI", "INJAS", "LVS",
            "INJAL", "ISCAL", "INJIN", "ISCIN", "INJLA", "ISCLA", "INJIL", "ISCIL", "ISCAN",
            "INJAS", "ISCAS"
        ]
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if not TORCH_AVAILABLE:
            return "cpu"
            
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
            
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load foundation model"""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for foundation model")
            return False
            
        try:
            model_path = model_path or self.model_path
            
            if self.model_type == "transformer":
                self.model = ECGTransformer(**self.model_configs["transformer"])
            elif self.model_type == "convnet":
                self.model = ECGConvNet(**self.model_configs["convnet"])
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    self.config = checkpoint.get("config", {})
                else:
                    self.model.load_state_dict(checkpoint)
                    
                logger.info(f"Loaded pre-trained model from {model_path}")
            else:
                logger.info(f"Initialized {self.model_type} model with random weights")
                
            self.model.to(self.device)
            self.model.eval()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load foundation model: {e}")
            return False
            
    async def analyze_ecg(
        self, 
        ecg_data: npt.NDArray[np.float32],
        return_embeddings: bool = False,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Analyze ECG using foundation model"""
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("Foundation model not available or loaded")
            
        try:
            start_time = time.time()
            
            if len(ecg_data.shape) == 2:
                ecg_data = ecg_data[np.newaxis, ...]  # Add batch dimension
                
            dataset = ECGDataset(ecg_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            all_predictions = []
            all_embeddings = []
            
            with torch.no_grad():
                for batch in dataloader:
                    ecg_batch = batch["ecg"].to(self.device)
                    
                    outputs = self.model(ecg_batch)
                    
                    logits = outputs["logits"]
                    probabilities = torch.softmax(logits, dim=-1)
                    
                    all_predictions.append(probabilities.cpu().numpy())
                    
                    if return_embeddings:
                        all_embeddings.append(outputs["embeddings"].cpu().numpy())
                        
            predictions = np.concatenate(all_predictions, axis=0)
            
            results = {
                "predictions": {},
                "confidence_scores": {},
                "top_conditions": [],
                "processing_time": time.time() - start_time,
                "model_type": self.model_type,
                "device": str(self.device)
            }
            
            for i, condition in enumerate(self.condition_names):
                if i < predictions.shape[1]:
                    results["predictions"][condition] = float(predictions[0, i])
                    
            top_indices = np.argsort(predictions[0])[-10:][::-1]
            for idx in top_indices:
                if idx < len(self.condition_names):
                    condition = self.condition_names[idx]
                    confidence = float(predictions[0, idx])
                    results["top_conditions"].append({
                        "condition": condition,
                        "confidence": confidence
                    })
                    
            max_confidence = float(np.max(predictions[0]))
            results["overall_confidence"] = max_confidence
            
            if return_embeddings and all_embeddings:
                embeddings = np.concatenate(all_embeddings, axis=0)
                results["embeddings"] = embeddings.tolist()
                
            return results
            
        except Exception as e:
            logger.error(f"Foundation model analysis failed: {e}")
            raise
            
    async def fine_tune(
        self,
        train_data: npt.NDArray[np.float32],
        train_labels: npt.NDArray[np.int64],
        val_data: Optional[npt.NDArray[np.float32]] = None,
        val_labels: Optional[npt.NDArray[np.int64]] = None,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Fine-tune the foundation model on custom data"""
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("Foundation model not available for fine-tuning")
            
        try:
            train_dataset = ECGDataset(train_data, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            val_loader = None
            if val_data is not None and val_labels is not None:
                val_dataset = ECGDataset(val_data, val_labels)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            
            training_history = {
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }
            
            for epoch in range(epochs):
                train_loss, train_acc = await self._train_epoch(
                    train_loader, optimizer, criterion
                )
                
                val_loss, val_acc = 0.0, 0.0
                if val_loader:
                    val_loss, val_acc = await self._validate_epoch(val_loader, criterion)
                    
                scheduler.step()
                
                training_history["train_loss"].append(train_loss)
                training_history["train_accuracy"].append(train_acc)
                training_history["val_loss"].append(val_loss)
                training_history["val_accuracy"].append(val_acc)
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
                
            self.model.eval()
            
            return {
                "status": "success",
                "epochs_completed": epochs,
                "final_train_loss": training_history["train_loss"][-1],
                "final_train_accuracy": training_history["train_accuracy"][-1],
                "final_val_loss": training_history["val_loss"][-1] if val_loader else None,
                "final_val_accuracy": training_history["val_accuracy"][-1] if val_loader else None,
                "training_history": training_history
            }
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise
            
    async def _train_epoch(
        self, 
        dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            ecg_batch = batch["ecg"].to(self.device)
            labels = batch["label"].squeeze().to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(ecg_batch)
            loss = criterion(outputs["logits"], labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs["logits"], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            await asyncio.sleep(0)
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
        
    async def _validate_epoch(
        self, 
        dataloader: DataLoader, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate for one epoch"""
        total_loss = 0.0
        correct = 0
        total = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                ecg_batch = batch["ecg"].to(self.device)
                labels = batch["label"].squeeze().to(self.device)
                
                outputs = self.model(ecg_batch)
                loss = criterion(outputs["logits"], labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs["logits"], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                await asyncio.sleep(0)
                
        self.model.train()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
        
    async def save_model(self, save_path: str) -> bool:
        """Save the foundation model"""
        if not TORCH_AVAILABLE or self.model is None:
            return False
            
        try:
            save_dict = {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "model_type": self.model_type,
                "condition_names": self.condition_names
            }
            
            torch.save(save_dict, save_path)
            logger.info(f"Model saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the foundation model"""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
            
        info = {
            "model_type": self.model_type,
            "device": str(self.device),
            "torch_available": TORCH_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "timm_available": TIMM_AVAILABLE,
            "num_conditions": len(self.condition_names),
            "model_loaded": self.model is not None
        }
        
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
                "config": self.config
            })
            
        return info
        
    async def extract_features(
        self, 
        ecg_data: npt.NDArray[np.float32],
        layer_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract features from intermediate layers"""
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("Foundation model not available")
            
        try:
            if len(ecg_data.shape) == 2:
                ecg_data = ecg_data[np.newaxis, ...]
                
            ecg_tensor = torch.FloatTensor(ecg_data).to(self.device)
            
            features = {}
            
            def hook_fn(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        features[name] = output.detach().cpu().numpy()
                    elif isinstance(output, dict):
                        for key, value in output.items():
                            if isinstance(value, torch.Tensor):
                                features[f"{name}_{key}"] = value.detach().cpu().numpy()
                return hook
                
            hooks = []
            
            for name, module in self.model.named_modules():
                if layer_name is None or layer_name in name:
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
                    
            with torch.no_grad():
                outputs = self.model(ecg_tensor)
                
            for hook in hooks:
                hook.remove()
                
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    features[f"output_{key}"] = value.cpu().numpy()
                    
            return {
                "features": features,
                "feature_names": list(features.keys()),
                "input_shape": ecg_data.shape
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
