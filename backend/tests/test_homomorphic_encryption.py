"""
Tests for Homomorphic Encryption Service
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from app.services.homomorphic_encryption import (
    HomomorphicEncryptionService,
    EncryptionContext,
    EncryptedData,
    TenSEALScheme,
    PaillierScheme,
    SecureMultiPartyComputation,
    ZeroKnowledgeProof,
    TENSEAL_AVAILABLE,
    PHE_AVAILABLE
)

class TestEncryptionContext:
    """Test EncryptionContext dataclass"""
    
    def test_encryption_context_creation(self):
        """Test creating encryption context"""
        context = EncryptionContext(
            scheme="test_scheme",
            parameters={"key_length": 2048}
        )
        
        assert context.scheme == "test_scheme"
        assert context.parameters["key_length"] == 2048
        assert context.created_at is not None
        assert context.public_key is None
        assert context.private_key is None

class TestEncryptedData:
    """Test EncryptedData dataclass"""
    
    def test_encrypted_data_creation(self):
        """Test creating encrypted data container"""
        test_data = [1, 2, 3, 4, 5]
        metadata = {"original_shape": (5,), "dtype": "float64"}
        
        encrypted = EncryptedData(
            data=test_data,
            scheme="test_scheme",
            metadata=metadata,
            checksum="",
            timestamp=1234567890.0
        )
        
        assert encrypted.data == test_data
        assert encrypted.scheme == "test_scheme"
        assert encrypted.metadata == metadata
        assert encrypted.checksum != ""  # Should be auto-generated
        assert encrypted.timestamp == 1234567890.0
    
    def test_checksum_generation(self):
        """Test automatic checksum generation"""
        encrypted1 = EncryptedData(
            data=[1, 2, 3],
            scheme="scheme1",
            metadata={"test": "data"},
            checksum="",
            timestamp=1234567890.0
        )
        
        encrypted2 = EncryptedData(
            data=[1, 2, 3],
            scheme="scheme1",
            metadata={"test": "data"},
            checksum="",
            timestamp=1234567890.0
        )
        
        assert encrypted1.checksum == encrypted2.checksum
        
        encrypted3 = EncryptedData(
            data=[1, 2, 3],
            scheme="scheme2",  # Different scheme
            metadata={"test": "data"},
            checksum="",
            timestamp=1234567890.0
        )
        
        assert encrypted1.checksum != encrypted3.checksum

@pytest.mark.skipif(not TENSEAL_AVAILABLE, reason="TenSEAL not available")
class TestTenSEALScheme:
    """Test TenSEAL homomorphic encryption scheme"""
    
    @pytest.fixture
    def tenseal_scheme(self):
        """Create TenSEAL scheme instance"""
        return TenSEALScheme()
    
    @pytest.mark.asyncio
    async def test_generate_keys(self, tenseal_scheme):
        """Test TenSEAL key generation"""
        context = await tenseal_scheme.generate_keys(
            poly_modulus_degree=4096,
            global_scale=2**30
        )
        
        assert context.scheme == "tenseal_ckks"
        assert context.context is not None
        assert context.parameters["poly_modulus_degree"] == 4096
        assert context.parameters["global_scale"] == 2**30
    
    @pytest.mark.asyncio
    async def test_encrypt_decrypt(self, tenseal_scheme):
        """Test TenSEAL encryption and decryption"""
        context = await tenseal_scheme.generate_keys()
        
        test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        encrypted = await tenseal_scheme.encrypt(test_data, context)
        
        assert encrypted.scheme == "tenseal_ckks"
        assert encrypted.metadata["original_shape"] == test_data.shape
        assert encrypted.metadata["size"] == test_data.size
        
        decrypted = await tenseal_scheme.decrypt(encrypted, context)
        
        assert decrypted.shape == test_data.shape
        np.testing.assert_allclose(decrypted, test_data, rtol=1e-3)
    
    @pytest.mark.asyncio
    async def test_homomorphic_addition(self, tenseal_scheme):
        """Test homomorphic addition"""
        context = await tenseal_scheme.generate_keys()
        
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([4.0, 5.0, 6.0])
        
        encrypted1 = await tenseal_scheme.encrypt(data1, context)
        encrypted2 = await tenseal_scheme.encrypt(data2, context)
        
        result = await tenseal_scheme.add(encrypted1, encrypted2)
        
        assert result.scheme == "tenseal_ckks"
        assert result.metadata["operation"] == "addition"
        
        decrypted_result = await tenseal_scheme.decrypt(result, context)
        expected = data1 + data2
        
        np.testing.assert_allclose(decrypted_result, expected, rtol=1e-3)
    
    @pytest.mark.asyncio
    async def test_homomorphic_scalar_multiplication(self, tenseal_scheme):
        """Test homomorphic scalar multiplication"""
        context = await tenseal_scheme.generate_keys()
        
        data = np.array([1.0, 2.0, 3.0])
        scalar = 2.5
        
        encrypted = await tenseal_scheme.encrypt(data, context)
        
        result = await tenseal_scheme.multiply(encrypted, scalar)
        
        assert result.scheme == "tenseal_ckks"
        assert result.metadata["operation"] == "scalar_multiplication"
        assert result.metadata["scalar"] == scalar
        
        decrypted_result = await tenseal_scheme.decrypt(result, context)
        expected = data * scalar
        
        np.testing.assert_allclose(decrypted_result, expected, rtol=1e-3)

@pytest.mark.skipif(not PHE_AVAILABLE, reason="phe library not available")
class TestPaillierScheme:
    """Test Paillier homomorphic encryption scheme"""
    
    @pytest.fixture
    def paillier_scheme(self):
        """Create Paillier scheme instance"""
        return PaillierScheme()
    
    @pytest.mark.asyncio
    async def test_generate_keys(self, paillier_scheme):
        """Test Paillier key generation"""
        context = await paillier_scheme.generate_keys(key_length=1024)
        
        assert context.scheme == "paillier"
        assert context.public_key is not None
        assert context.private_key is not None
        assert context.parameters["key_length"] == 1024
    
    @pytest.mark.asyncio
    async def test_encrypt_decrypt(self, paillier_scheme):
        """Test Paillier encryption and decryption"""
        context = await paillier_scheme.generate_keys(key_length=1024)
        
        test_data = np.array([[1.5, 2.7, 3.2], [4.1, 5.9, 6.3]])
        
        encrypted = await paillier_scheme.encrypt(test_data, context)
        
        assert encrypted.scheme == "paillier"
        assert encrypted.metadata["original_shape"] == test_data.shape
        assert "scale_factor" in encrypted.metadata
        
        decrypted = await paillier_scheme.decrypt(encrypted, context)
        
        assert decrypted.shape == test_data.shape
        np.testing.assert_allclose(decrypted, test_data, rtol=1e-3)
    
    @pytest.mark.asyncio
    async def test_homomorphic_addition(self, paillier_scheme):
        """Test Paillier homomorphic addition"""
        context = await paillier_scheme.generate_keys(key_length=1024)
        
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([4.0, 5.0, 6.0])
        
        encrypted1 = await paillier_scheme.encrypt(data1, context)
        encrypted2 = await paillier_scheme.encrypt(data2, context)
        
        result = await paillier_scheme.add(encrypted1, encrypted2)
        
        assert result.scheme == "paillier"
        assert result.metadata["operation"] == "addition"
        
        decrypted_result = await paillier_scheme.decrypt(result, context)
        expected = data1 + data2
        
        np.testing.assert_allclose(decrypted_result, expected, rtol=1e-3)
    
    @pytest.mark.asyncio
    async def test_homomorphic_scalar_multiplication(self, paillier_scheme):
        """Test Paillier homomorphic scalar multiplication"""
        context = await paillier_scheme.generate_keys(key_length=1024)
        
        data = np.array([1.0, 2.0, 3.0])
        scalar = 3.0
        
        encrypted = await paillier_scheme.encrypt(data, context)
        
        result = await paillier_scheme.multiply(encrypted, scalar)
        
        assert result.scheme == "paillier"
        assert result.metadata["operation"] == "scalar_multiplication"
        assert result.metadata["scalar"] == scalar
        
        decrypted_result = await paillier_scheme.decrypt(result, context)
        expected = data * scalar
        
        np.testing.assert_allclose(decrypted_result, expected, rtol=1e-3)

class TestSecureMultiPartyComputation:
    """Test secure multi-party computation"""
    
    @pytest.fixture
    def mock_scheme(self):
        """Create mock homomorphic scheme"""
        scheme = Mock()
        scheme.generate_keys = AsyncMock()
        scheme.add = AsyncMock()
        scheme.multiply = AsyncMock()
        return scheme
    
    @pytest.fixture
    def smc(self, mock_scheme):
        """Create SMC instance with mock scheme"""
        return SecureMultiPartyComputation(mock_scheme)
    
    @pytest.mark.asyncio
    async def test_register_participant(self, smc, mock_scheme):
        """Test participant registration"""
        mock_context = EncryptionContext(scheme="test")
        mock_scheme.generate_keys.return_value = mock_context
        
        context = await smc.register_participant("participant1", key_length=2048)
        
        assert context == mock_context
        assert "participant1" in smc.participants
        mock_scheme.generate_keys.assert_called_once_with(key_length=2048)
    
    @pytest.mark.asyncio
    async def test_secure_aggregation(self, smc, mock_scheme):
        """Test secure aggregation"""
        encrypted1 = EncryptedData([1, 2, 3], "test", {}, "", 0)
        encrypted2 = EncryptedData([4, 5, 6], "test", {}, "", 0)
        encrypted3 = EncryptedData([7, 8, 9], "test", {}, "", 0)
        
        contributions = {
            "participant1": encrypted1,
            "participant2": encrypted2,
            "participant3": encrypted3
        }
        
        intermediate_result = EncryptedData([5, 7, 9], "test", {}, "", 0)
        final_result = EncryptedData([12, 15, 18], "test", {}, "", 0)
        
        mock_scheme.add.side_effect = [intermediate_result, final_result]
        
        result = await smc.secure_aggregation(contributions)
        
        assert result == final_result
        assert len(smc.computation_history) == 1
        assert smc.computation_history[0]["operation"] == "secure_aggregation"
        assert len(smc.computation_history[0]["participants"]) == 3
    
    @pytest.mark.asyncio
    async def test_secure_average(self, smc, mock_scheme):
        """Test secure average computation"""
        encrypted1 = EncryptedData([3, 6, 9], "test", {}, "", 0)
        encrypted2 = EncryptedData([6, 12, 18], "test", {}, "", 0)
        
        contributions = {
            "participant1": encrypted1,
            "participant2": encrypted2
        }
        
        aggregated = EncryptedData([9, 18, 27], "test", {}, "", 0)
        averaged = EncryptedData([4.5, 9, 13.5], "test", {}, "", 0)
        
        mock_scheme.add.return_value = aggregated
        mock_scheme.multiply.return_value = averaged
        
        result = await smc.secure_average(contributions)
        
        assert result == averaged
        mock_scheme.multiply.assert_called_once_with(aggregated, 0.5)  # 1/2 participants
        
        assert len(smc.computation_history) == 2  # aggregation + average
        assert smc.computation_history[1]["operation"] == "secure_average"
    
    @pytest.mark.asyncio
    async def test_empty_contributions_error(self, smc):
        """Test error handling for empty contributions"""
        with pytest.raises(ValueError, match="No contributions provided"):
            await smc.secure_aggregation({})

class TestZeroKnowledgeProof:
    """Test zero-knowledge proof system"""
    
    @pytest.fixture
    def zkp(self):
        """Create ZKP instance"""
        return ZeroKnowledgeProof()
    
    @pytest.mark.asyncio
    async def test_range_proof_generation(self, zkp):
        """Test range proof generation"""
        value = 75.0
        min_range = 60.0
        max_range = 100.0
        proof_id = "test_range_proof"
        
        proof = await zkp.generate_range_proof(value, min_range, max_range, proof_id)
        
        assert proof["range"]["min"] == min_range
        assert proof["range"]["max"] == max_range
        assert proof["valid"] is True
        assert "commitment" in proof
        assert "timestamp" in proof
        assert proof_id in zkp.proofs
    
    @pytest.mark.asyncio
    async def test_range_proof_verification(self, zkp):
        """Test range proof verification"""
        proof_id = "test_proof"
        
        await zkp.generate_range_proof(75.0, 60.0, 100.0, proof_id)
        
        is_valid = await zkp.verify_range_proof(proof_id)
        assert is_valid is True
        
        is_valid = await zkp.verify_range_proof("non_existent")
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_range_proof_invalid_value(self, zkp):
        """Test range proof with invalid value"""
        value = 150.0  # Outside range
        min_range = 60.0
        max_range = 100.0
        proof_id = "invalid_range_proof"
        
        proof = await zkp.generate_range_proof(value, min_range, max_range, proof_id)
        
        assert proof["valid"] is False
        
        is_valid = await zkp.verify_range_proof(proof_id)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_membership_proof(self, zkp):
        """Test membership proof generation"""
        value = "atrial_fibrillation"
        valid_set = ["normal", "atrial_fibrillation", "ventricular_tachycardia"]
        proof_id = "membership_test"
        
        proof = await zkp.generate_membership_proof(value, valid_set, proof_id)
        
        assert proof["set_size"] == 3
        assert proof["valid"] is True
        assert "commitment" in proof
        assert proof_id in zkp.proofs
    
    @pytest.mark.asyncio
    async def test_membership_proof_invalid(self, zkp):
        """Test membership proof with invalid value"""
        value = "unknown_condition"
        valid_set = ["normal", "atrial_fibrillation", "ventricular_tachycardia"]
        proof_id = "invalid_membership"
        
        proof = await zkp.generate_membership_proof(value, valid_set, proof_id)
        
        assert proof["valid"] is False

class TestHomomorphicEncryptionService:
    """Test main homomorphic encryption service"""
    
    @pytest.fixture
    def service(self):
        """Create service instance"""
        return HomomorphicEncryptionService()
    
    def test_service_initialization(self, service):
        """Test service initialization"""
        assert isinstance(service.schemes, dict)
        assert isinstance(service.zkp, ZeroKnowledgeProof)
        assert service.smc is None
        assert len(service.active_contexts) == 0
        
        if TENSEAL_AVAILABLE:
            assert "tenseal" in service.schemes
        if PHE_AVAILABLE:
            assert "paillier" in service.schemes
    
    @pytest.mark.asyncio
    async def test_create_encryption_context(self, service):
        """Test encryption context creation"""
        if not service.schemes:
            pytest.skip("No encryption schemes available")
        
        scheme_name = list(service.schemes.keys())[0]
        context_id = "test_context"
        
        context = await service.create_encryption_context(
            scheme_name, context_id, key_length=1024
        )
        
        assert context_id in service.active_contexts
        assert service.active_contexts[context_id] == context
    
    @pytest.mark.asyncio
    async def test_unsupported_scheme_error(self, service):
        """Test error for unsupported encryption scheme"""
        with pytest.raises(ValueError, match="Unsupported encryption scheme"):
            await service.create_encryption_context("unsupported", "test", key_length=1024)
    
    @pytest.mark.asyncio
    async def test_setup_secure_computation(self, service):
        """Test SMC setup"""
        if not service.schemes:
            pytest.skip("No encryption schemes available")
        
        scheme_name = list(service.schemes.keys())[0]
        smc = await service.setup_secure_computation(scheme_name)
        
        assert service.smc == smc
        assert isinstance(smc, SecureMultiPartyComputation)
    
    @pytest.mark.asyncio
    async def test_validate_ecg_quality(self, service):
        """Test ECG quality validation with ZKP"""
        ecg_metrics = {
            "heart_rate": 75.0,
            "qrs_duration": 100.0,
            "pr_interval": 160.0
        }
        
        quality_thresholds = {
            "heart_rate": (60.0, 100.0),
            "qrs_duration": (80.0, 120.0),
            "pr_interval": (120.0, 200.0)
        }
        
        results = await service.validate_ecg_quality(ecg_metrics, quality_thresholds)
        
        assert len(results) == 3
        assert all(results.values())  # All should be valid
        
        invalid_metrics = {
            "heart_rate": 150.0,  # Too high
            "qrs_duration": 50.0   # Too low
        }
        
        invalid_results = await service.validate_ecg_quality(invalid_metrics, quality_thresholds)
        
        assert not invalid_results["heart_rate"]
        assert not invalid_results["qrs_duration"]
    
    @pytest.mark.asyncio
    async def test_get_encryption_stats(self, service):
        """Test encryption statistics"""
        stats = await service.get_encryption_stats()
        
        assert "available_schemes" in stats
        assert "active_contexts" in stats
        assert "smc_initialized" in stats
        assert "computation_history" in stats
        assert "zkp_proofs" in stats
        
        assert stats["active_contexts"] == 0
        assert stats["smc_initialized"] is False
        assert stats["computation_history"] == 0

class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_federated_learning_scenario(self):
        """Test complete federated learning scenario"""
        service = HomomorphicEncryptionService()
        
        if not service.schemes:
            pytest.skip("No encryption schemes available")
        
        scheme_name = list(service.schemes.keys())[0]
        smc = await service.setup_secure_computation(scheme_name)
        
        mock_gradients = {}
        for i in range(3):
            participant_id = f"hospital_{i+1}"
            gradient_data = np.random.randn(10)  # Mock gradient vector
            
            context = await service.create_encryption_context(
                scheme_name, participant_id, key_length=1024
            )
            
            encrypted_gradient = await service.encrypt_ecg_data(gradient_data, participant_id)
            mock_gradients[participant_id] = encrypted_gradient
        
        averaged_gradients = await service.federated_model_update(mock_gradients)
        
        assert averaged_gradients is not None
        assert len(smc.computation_history) >= 2  # aggregation + average
    
    @pytest.mark.asyncio
    async def test_privacy_preserving_ecg_analysis(self):
        """Test privacy-preserving ECG analysis workflow"""
        service = HomomorphicEncryptionService()
        
        if not service.schemes:
            pytest.skip("No encryption schemes available")
        
        scheme_name = list(service.schemes.keys())[0]
        
        context_id = "ecg_analysis"
        await service.create_encryption_context(scheme_name, context_id)
        
        ecg_data = np.random.randn(12, 5000)  # 12-lead ECG, 5000 samples
        
        encrypted_ecg = await service.encrypt_ecg_data(ecg_data, context_id)
        
        assert encrypted_ecg.scheme == service.active_contexts[context_id].scheme
        assert encrypted_ecg.metadata["original_shape"] == ecg_data.shape
        
        decrypted_ecg = await service.decrypt_ecg_data(encrypted_ecg, context_id)
        
        assert decrypted_ecg.shape == ecg_data.shape
        
        ecg_metrics = {
            "signal_quality": 0.85,
            "noise_level": 0.15
        }
        
        quality_thresholds = {
            "signal_quality": (0.7, 1.0),
            "noise_level": (0.0, 0.3)
        }
        
        validation_results = await service.validate_ecg_quality(ecg_metrics, quality_thresholds)
        
        assert all(validation_results.values())

if __name__ == "__main__":
    pytest.main([__file__])
