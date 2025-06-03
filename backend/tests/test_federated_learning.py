"""
Tests for Federated Learning Service
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

import numpy as np

from app.services.federated_learning import (
    FederatedLearningStatus,
    ParticipantRole,
    DifferentialPrivacyConfig,
    FederatedLearningConfig,
    ModelUpdate,
    FederatedRound,
    DifferentialPrivacyMechanism,
    PrivacyAccountant,
    SecureCommunication,
    FederatedAggregator,
    FederatedLearningCoordinator,
    FederatedLearningParticipant,
    FederatedLearningService
)


class TestDifferentialPrivacyConfig:
    """Test differential privacy configuration"""
    
    def test_default_config(self):
        """Test default differential privacy configuration"""
        config = DifferentialPrivacyConfig()
        
        assert config.epsilon == 1.0
        assert config.delta == 1e-5
        assert config.noise_multiplier == 1.1
        assert config.max_grad_norm == 1.0
        assert config.enable_privacy is True
        
    def test_custom_config(self):
        """Test custom differential privacy configuration"""
        config = DifferentialPrivacyConfig(
            epsilon=0.5,
            delta=1e-6,
            noise_multiplier=2.0,
            max_grad_norm=0.5,
            enable_privacy=False
        )
        
        assert config.epsilon == 0.5
        assert config.delta == 1e-6
        assert config.noise_multiplier == 2.0
        assert config.max_grad_norm == 0.5
        assert config.enable_privacy is False


class TestFederatedLearningConfig:
    """Test federated learning configuration"""
    
    def test_default_config(self):
        """Test default federated learning configuration"""
        config = FederatedLearningConfig(round_id="test_round")
        
        assert config.round_id == "test_round"
        assert config.min_participants == 3
        assert config.max_participants == 10
        assert config.rounds_per_epoch == 5
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.local_epochs == 1
        assert config.aggregation_method == "fedavg"
        assert config.timeout_seconds == 3600
        assert config.model_architecture == "ecg_classifier"


class TestModelUpdate:
    """Test model update dataclass"""
    
    def test_model_update_creation(self):
        """Test creating model update"""
        weights = {"layer1": np.array([1.0, 2.0], dtype=np.float32)}
        timestamp = datetime.now(timezone.utc)
        
        update = ModelUpdate(
            participant_id="participant_1",
            round_id="round_1",
            model_weights=weights,
            training_samples=100,
            training_loss=0.5,
            validation_accuracy=0.85,
            timestamp=timestamp
        )
        
        assert update.participant_id == "participant_1"
        assert update.round_id == "round_1"
        assert np.array_equal(update.model_weights["layer1"], weights["layer1"])
        assert update.training_samples == 100
        assert update.training_loss == 0.5
        assert update.validation_accuracy == 0.85
        assert update.timestamp == timestamp


class TestDifferentialPrivacyMechanism:
    """Test differential privacy mechanism"""
    
    def setup_method(self):
        self.config = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5)
        self.dp_mechanism = DifferentialPrivacyMechanism(self.config)
        
    def test_add_noise_to_gradients_enabled(self):
        """Test adding noise to gradients when privacy is enabled"""
        gradients = {
            "layer1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "layer2": np.array([0.5, 1.5], dtype=np.float32)
        }
        
        noisy_gradients = self.dp_mechanism.add_noise_to_gradients(gradients)
        
        assert "layer1" in noisy_gradients
        assert "layer2" in noisy_gradients
        assert noisy_gradients["layer1"].shape == gradients["layer1"].shape
        assert noisy_gradients["layer2"].shape == gradients["layer2"].shape
        
        assert not np.array_equal(noisy_gradients["layer1"], gradients["layer1"])
        assert not np.array_equal(noisy_gradients["layer2"], gradients["layer2"])
        
    def test_add_noise_to_gradients_disabled(self):
        """Test adding noise to gradients when privacy is disabled"""
        self.config.enable_privacy = False
        dp_mechanism = DifferentialPrivacyMechanism(self.config)
        
        gradients = {
            "layer1": np.array([1.0, 2.0, 3.0], dtype=np.float32)
        }
        
        noisy_gradients = dp_mechanism.add_noise_to_gradients(gradients)
        
        assert np.array_equal(noisy_gradients["layer1"], gradients["layer1"])
        
    def test_add_noise_to_model_weights(self):
        """Test adding noise to model weights"""
        weights = {
            "layer1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "layer2": np.array([0.5, 1.5], dtype=np.float32)
        }
        
        noisy_weights = self.dp_mechanism.add_noise_to_model_weights(weights)
        
        assert "layer1" in noisy_weights
        assert "layer2" in noisy_weights
        assert noisy_weights["layer1"].shape == weights["layer1"].shape
        assert noisy_weights["layer2"].shape == weights["layer2"].shape


class TestPrivacyAccountant:
    """Test privacy budget accounting"""
    
    def setup_method(self):
        self.accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5)
        
    def test_initial_budget(self):
        """Test initial privacy budget"""
        remaining_epsilon, remaining_delta = self.accountant.get_remaining_budget()
        
        assert remaining_epsilon == 1.0
        assert remaining_delta == 1e-5
        assert self.accountant.consumed_epsilon == 0.0
        assert self.accountant.consumed_delta == 0.0
        
    def test_consume_budget_success(self):
        """Test successful budget consumption"""
        result = self.accountant.consume_budget(noise_multiplier=2.0)
        
        assert result is True
        assert self.accountant.consumed_epsilon > 0
        assert self.accountant.consumed_delta > 0
        assert len(self.accountant.privacy_history) == 1
        
    def test_consume_budget_exhausted(self):
        """Test budget consumption when exhausted"""
        for _ in range(10):
            self.accountant.consume_budget(noise_multiplier=1.1)
            
        result = self.accountant.consume_budget(noise_multiplier=0.1)
        
        assert result is False
        
    def test_reset_budget(self):
        """Test resetting privacy budget"""
        self.accountant.consume_budget(noise_multiplier=2.0)
        
        assert self.accountant.consumed_epsilon > 0
        assert self.accountant.consumed_delta > 0
        
        self.accountant.reset_budget()
        
        assert self.accountant.consumed_epsilon == 0.0
        assert self.accountant.consumed_delta == 0.0
        assert len(self.accountant.privacy_history) == 0


class TestSecureCommunication:
    """Test secure communication"""
    
    def setup_method(self):
        self.secure_comm = SecureCommunication()
        
    def test_serialize_model_update(self):
        """Test serializing model update"""
        weights = {"layer1": np.array([1.0, 2.0], dtype=np.float32)}
        timestamp = datetime.now(timezone.utc)
        
        update = ModelUpdate(
            participant_id="participant_1",
            round_id="round_1",
            model_weights=weights,
            training_samples=100,
            training_loss=0.5,
            validation_accuracy=0.85,
            timestamp=timestamp
        )
        
        serialized = self.secure_comm._serialize_model_update(update)
        
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
    def test_deserialize_model_update(self):
        """Test deserializing model update"""
        weights = {"layer1": np.array([1.0, 2.0], dtype=np.float32)}
        timestamp = datetime.now(timezone.utc)
        
        original_update = ModelUpdate(
            participant_id="participant_1",
            round_id="round_1",
            model_weights=weights,
            training_samples=100,
            training_loss=0.5,
            validation_accuracy=0.85,
            timestamp=timestamp
        )
        
        serialized = self.secure_comm._serialize_model_update(original_update)
        deserialized = self.secure_comm._deserialize_model_update(serialized)
        
        assert deserialized is not None
        assert deserialized.participant_id == original_update.participant_id
        assert deserialized.round_id == original_update.round_id
        assert deserialized.training_samples == original_update.training_samples
        assert deserialized.training_loss == original_update.training_loss
        assert deserialized.validation_accuracy == original_update.validation_accuracy
        assert np.array_equal(deserialized.model_weights["layer1"], weights["layer1"])
        
    def test_encrypt_decrypt_model_update(self):
        """Test encrypting and decrypting model update"""
        weights = {"layer1": np.array([1.0, 2.0], dtype=np.float32)}
        timestamp = datetime.now(timezone.utc)
        
        update = ModelUpdate(
            participant_id="participant_1",
            round_id="round_1",
            model_weights=weights,
            training_samples=100,
            training_loss=0.5,
            validation_accuracy=0.85,
            timestamp=timestamp
        )
        
        encrypted = self.secure_comm.encrypt_model_update(update)
        decrypted = self.secure_comm.decrypt_model_update(encrypted)
        
        assert isinstance(encrypted, bytes)
        assert decrypted is not None
        assert decrypted.participant_id == update.participant_id
        assert decrypted.round_id == update.round_id


class TestFederatedAggregator:
    """Test federated aggregation algorithms"""
    
    def setup_method(self):
        self.aggregator = FederatedAggregator("fedavg")
        
    def test_federated_averaging(self):
        """Test FedAvg aggregation"""
        weights1 = {"layer1": np.array([1.0, 2.0], dtype=np.float32)}
        weights2 = {"layer1": np.array([3.0, 4.0], dtype=np.float32)}
        
        update1 = ModelUpdate(
            participant_id="p1",
            round_id="r1",
            model_weights=weights1,
            training_samples=100,
            training_loss=0.5,
            validation_accuracy=0.85,
            timestamp=datetime.now(timezone.utc)
        )
        
        update2 = ModelUpdate(
            participant_id="p2",
            round_id="r1",
            model_weights=weights2,
            training_samples=200,
            training_loss=0.4,
            validation_accuracy=0.90,
            timestamp=datetime.now(timezone.utc)
        )
        
        aggregated = self.aggregator.aggregate_model_updates([update1, update2])
        
        assert "layer1" in aggregated
        assert aggregated["layer1"].shape == (2,)
        
        expected = np.array([7.0/3.0, 10.0/3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(aggregated["layer1"], expected, decimal=2)
        
    def test_aggregate_empty_updates(self):
        """Test aggregation with empty updates"""
        with pytest.raises(ValueError, match="No model updates to aggregate"):
            self.aggregator.aggregate_model_updates([])
            
    def test_unknown_aggregation_method(self):
        """Test unknown aggregation method falls back to FedAvg"""
        aggregator = FederatedAggregator("unknown_method")
        
        weights = {"layer1": np.array([1.0, 2.0], dtype=np.float32)}
        update = ModelUpdate(
            participant_id="p1",
            round_id="r1",
            model_weights=weights,
            training_samples=100,
            training_loss=0.5,
            validation_accuracy=0.85,
            timestamp=datetime.now(timezone.utc)
        )
        
        aggregated = aggregator.aggregate_model_updates([update])
        assert "layer1" in aggregated


class TestFederatedLearningCoordinator:
    """Test federated learning coordinator"""
    
    def setup_method(self):
        self.coordinator = FederatedLearningCoordinator("coordinator_1")
        
    @pytest.mark.asyncio
    async def test_start_federated_round(self):
        """Test starting a federated learning round"""
        config = FederatedLearningConfig(round_id="test_round")
        
        round_id = await self.coordinator.start_federated_round(config)
        
        assert round_id == "test_round"
        assert round_id in self.coordinator.active_rounds
        
        federated_round = self.coordinator.active_rounds[round_id]
        assert federated_round.status == FederatedLearningStatus.WAITING_FOR_PARTICIPANTS
        assert federated_round.config == config
        
    @pytest.mark.asyncio
    async def test_register_participant(self):
        """Test registering participants"""
        config = FederatedLearningConfig(round_id="test_round")
        round_id = await self.coordinator.start_federated_round(config)
        
        participant_info = {"node_id": "participant_1", "capabilities": ["ecg"]}
        
        result = await self.coordinator.register_participant(
            round_id, "participant_1", participant_info
        )
        
        assert result is True
        
        federated_round = self.coordinator.active_rounds[round_id]
        assert "participant_1" in federated_round.participants
        assert "participant_1" in self.coordinator.participants
        
    @pytest.mark.asyncio
    async def test_register_participant_nonexistent_round(self):
        """Test registering participant for non-existent round"""
        result = await self.coordinator.register_participant(
            "nonexistent_round", "participant_1", {}
        )
        
        assert result is False
        
    @pytest.mark.asyncio
    async def test_submit_model_update(self):
        """Test submitting model update"""
        config = FederatedLearningConfig(round_id="test_round")
        round_id = await self.coordinator.start_federated_round(config)
        
        await self.coordinator.register_participant(round_id, "p1", {})
        await self.coordinator.register_participant(round_id, "p2", {})
        await self.coordinator.register_participant(round_id, "p3", {})
        
        weights = {"layer1": np.array([1.0, 2.0], dtype=np.float32)}
        update = ModelUpdate(
            participant_id="p1",
            round_id=round_id,
            model_weights=weights,
            training_samples=100,
            training_loss=0.5,
            validation_accuracy=0.85,
            timestamp=datetime.now(timezone.utc)
        )
        
        result = await self.coordinator.submit_model_update(update)
        
        assert result is True
        
    def test_get_round_status(self):
        """Test getting round status"""
        status = self.coordinator.get_round_status("nonexistent")
        assert status is None
        
    @pytest.mark.asyncio
    async def test_get_round_status_active(self):
        """Test getting status of active round"""
        config = FederatedLearningConfig(round_id="test_round")
        round_id = await self.coordinator.start_federated_round(config)
        
        status = self.coordinator.get_round_status(round_id)
        
        assert status is not None
        assert status["round_id"] == round_id
        assert status["status"] == FederatedLearningStatus.WAITING_FOR_PARTICIPANTS.value
        assert status["participants"] == 0


class TestFederatedLearningParticipant:
    """Test federated learning participant"""
    
    def setup_method(self):
        self.participant = FederatedLearningParticipant("participant_1")
        
    @pytest.mark.asyncio
    async def test_join_federated_round(self):
        """Test joining a federated round"""
        result = await self.participant.join_federated_round(
            "http://coordinator:8000",
            "test_round",
            {"node_id": "participant_1"}
        )
        
        assert result is True
        assert hasattr(self.participant, 'current_round_id')
        assert self.participant.current_round_id == "test_round"
        
    @pytest.mark.asyncio
    async def test_train_local_model(self):
        """Test local model training"""
        training_data = np.random.random((100, 10)).astype(np.float32)
        training_labels = np.random.randint(0, 2, 100).astype(np.int64)
        
        global_weights = {
            "layer1": np.random.random((10, 5)).astype(np.float32),
            "layer2": np.random.random((5, 2)).astype(np.float32)
        }
        
        config = FederatedLearningConfig(round_id="test_round")
        
        model_update = await self.participant.train_local_model(
            training_data, training_labels, global_weights, config
        )
        
        assert model_update is not None
        assert model_update.participant_id == "participant_1"
        assert model_update.round_id == "test_round"
        assert model_update.training_samples == 100
        assert model_update.training_loss > 0
        assert model_update.validation_accuracy > 0
        assert "layer1" in model_update.model_weights
        assert "layer2" in model_update.model_weights
        
    @pytest.mark.asyncio
    async def test_train_local_model_with_privacy(self):
        """Test local model training with differential privacy"""
        training_data = np.random.random((50, 10)).astype(np.float32)
        training_labels = np.random.randint(0, 2, 50).astype(np.int64)
        
        global_weights = {
            "layer1": np.random.random((10, 5)).astype(np.float32)
        }
        
        dp_config = DifferentialPrivacyConfig(epsilon=1.0, enable_privacy=True)
        config = FederatedLearningConfig(
            round_id="test_round",
            differential_privacy=dp_config
        )
        
        model_update = await self.participant.train_local_model(
            training_data, training_labels, global_weights, config
        )
        
        assert model_update is not None
        assert model_update.privacy_budget_used > 0


class TestFederatedLearningService:
    """Test main federated learning service"""
    
    def setup_method(self):
        self.coordinator_service = FederatedLearningService(
            "coordinator_1", 
            ParticipantRole.COORDINATOR
        )
        self.participant_service = FederatedLearningService(
            "participant_1", 
            ParticipantRole.PARTICIPANT
        )
        
    @pytest.mark.asyncio
    async def test_create_federated_round_coordinator(self):
        """Test creating federated round as coordinator"""
        config = FederatedLearningConfig(round_id="test_round")
        
        round_id = await self.coordinator_service.create_federated_round(config)
        
        assert round_id == "test_round"
        assert self.coordinator_service.service_status["active_rounds"] == 1
        
    @pytest.mark.asyncio
    async def test_create_federated_round_participant_fails(self):
        """Test that participants cannot create federated rounds"""
        config = FederatedLearningConfig(round_id="test_round")
        
        with pytest.raises(ValueError, match="Only coordinators can create"):
            await self.participant_service.create_federated_round(config)
            
    @pytest.mark.asyncio
    async def test_join_federated_round_participant(self):
        """Test joining federated round as participant"""
        result = await self.participant_service.join_federated_round(
            "http://coordinator:8000",
            "test_round",
            {"capabilities": ["ecg"]}
        )
        
        assert result is True
        assert self.participant_service.service_status["active_rounds"] == 1
        
    @pytest.mark.asyncio
    async def test_join_federated_round_coordinator_fails(self):
        """Test that coordinators cannot join rounds as participants"""
        with pytest.raises(ValueError, match="Coordinators cannot join rounds"):
            await self.coordinator_service.join_federated_round(
                "http://coordinator:8000",
                "test_round"
            )
            
    @pytest.mark.asyncio
    async def test_participate_in_training(self):
        """Test participating in federated training"""
        training_data = np.random.random((50, 10)).astype(np.float32)
        training_labels = np.random.randint(0, 2, 50).astype(np.int64)
        global_weights = {"layer1": np.random.random((10, 5)).astype(np.float32)}
        config = FederatedLearningConfig(round_id="test_round")
        
        model_update = await self.participant_service.participate_in_training(
            "test_round", training_data, training_labels, global_weights, config
        )
        
        assert model_update is not None
        assert model_update.participant_id == "participant_1"
        
    def test_get_service_status_coordinator(self):
        """Test getting service status for coordinator"""
        status = self.coordinator_service.get_service_status()
        
        assert status["node_id"] == "coordinator_1"
        assert status["role"] == "coordinator"
        assert "coordinator_info" in status
        
    def test_get_service_status_participant(self):
        """Test getting service status for participant"""
        status = self.participant_service.get_service_status()
        
        assert status["node_id"] == "participant_1"
        assert status["role"] == "participant"
        assert "participant_info" in status
        
    def test_get_privacy_capabilities(self):
        """Test getting privacy capabilities"""
        capabilities = self.coordinator_service.get_privacy_capabilities()
        
        assert "differential_privacy" in capabilities
        assert "secure_communication" in capabilities
        assert "privacy_accounting" in capabilities
        assert "supported_aggregation_methods" in capabilities
        assert "encryption_available" in capabilities
        assert "pytorch_available" in capabilities
        
        assert capabilities["differential_privacy"] is True
        assert capabilities["privacy_accounting"] is True
        assert "fedavg" in capabilities["supported_aggregation_methods"]


@pytest.mark.asyncio
async def test_integration_workflow():
    """Test complete federated learning workflow"""
    coordinator = FederatedLearningService("coordinator", ParticipantRole.COORDINATOR)
    participant1 = FederatedLearningService("participant1", ParticipantRole.PARTICIPANT)
    participant2 = FederatedLearningService("participant2", ParticipantRole.PARTICIPANT)
    
    config = FederatedLearningConfig(
        round_id="integration_test_round",
        min_participants=2,
        max_participants=5,
        local_epochs=1
    )
    
    round_id = await coordinator.create_federated_round(config)
    assert round_id == "integration_test_round"
    
    result1 = await participant1.join_federated_round(
        "http://coordinator:8000",
        round_id,
        {"capabilities": ["ecg_classification"]}
    )
    result2 = await participant2.join_federated_round(
        "http://coordinator:8000", 
        round_id,
        {"capabilities": ["ecg_classification"]}
    )
    
    assert result1 is True
    assert result2 is True
    
    training_data1 = np.random.random((100, 20)).astype(np.float32)
    training_labels1 = np.random.randint(0, 3, 100).astype(np.int64)
    
    training_data2 = np.random.random((150, 20)).astype(np.float32)
    training_labels2 = np.random.randint(0, 3, 150).astype(np.int64)
    
    global_weights = {
        "fc1.weight": np.random.random((128, 20)).astype(np.float32),
        "fc1.bias": np.random.random((128,)).astype(np.float32),
        "fc2.weight": np.random.random((3, 128)).astype(np.float32),
        "fc2.bias": np.random.random((3,)).astype(np.float32)
    }
    
    update1 = await participant1.participate_in_training(
        round_id, training_data1, training_labels1, global_weights, config
    )
    update2 = await participant2.participate_in_training(
        round_id, training_data2, training_labels2, global_weights, config
    )
    
    assert update1 is not None
    assert update2 is not None
    assert update1.training_samples == 100
    assert update2.training_samples == 150
    
    coordinator_status = coordinator.get_service_status()
    participant1_status = participant1.get_service_status()
    
    assert coordinator_status["active_rounds"] == 1
    assert participant1_status["active_rounds"] == 1
    
    capabilities = coordinator.get_privacy_capabilities()
    assert capabilities["differential_privacy"] is True
    assert "fedavg" in capabilities["supported_aggregation_methods"]
