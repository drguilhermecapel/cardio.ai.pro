"""
Comprehensive tests for blockchain service
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from app.services.blockchain_service import (
    BlockchainService,
    BlockchainNetwork,
    ECGAnalysisRecord,
    PatientIdentity,
    ComplianceContract,
    AuditTrail,
    EthereumProvider,
    HyperledgerProvider,
    MockBlockchainProvider,
    IPFSStorage,
    blockchain_service
)

class TestECGAnalysisRecord:
    """Test ECG analysis record functionality"""
    
    def test_ecg_analysis_record_creation(self):
        """Test creating ECG analysis record"""
        record = ECGAnalysisRecord(
            analysis_id="test_analysis_001",
            patient_id="patient_123",
            timestamp=time.time(),
            ecg_hash="abc123def456",
            analysis_results={"heart_rate": 75, "rhythm": "normal"},
            physician_id="dr_smith",
            facility_id="hospital_001",
            compliance_flags=["REVIEWED"],
            data_integrity_hash="integrity_hash_123"
        )
        
        assert record.analysis_id == "test_analysis_001"
        assert record.patient_id == "patient_123"
        assert record.analysis_results["heart_rate"] == 75
        assert "REVIEWED" in record.compliance_flags
    
    def test_ecg_analysis_record_to_dict(self):
        """Test converting ECG analysis record to dictionary"""
        record = ECGAnalysisRecord(
            analysis_id="test_analysis_001",
            patient_id="patient_123",
            timestamp=1234567890.0,
            ecg_hash="abc123def456",
            analysis_results={"heart_rate": 75},
            physician_id="dr_smith",
            facility_id="hospital_001",
            compliance_flags=["REVIEWED"],
            data_integrity_hash="integrity_hash_123"
        )
        
        record_dict = record.to_dict()
        
        assert isinstance(record_dict, dict)
        assert record_dict["analysis_id"] == "test_analysis_001"
        assert record_dict["patient_id"] == "patient_123"
        assert record_dict["timestamp"] == 1234567890.0
    
    def test_ecg_analysis_record_compute_hash(self):
        """Test computing hash of ECG analysis record"""
        record = ECGAnalysisRecord(
            analysis_id="test_analysis_001",
            patient_id="patient_123",
            timestamp=1234567890.0,
            ecg_hash="abc123def456",
            analysis_results={"heart_rate": 75},
            physician_id="dr_smith",
            facility_id="hospital_001",
            compliance_flags=["REVIEWED"],
            data_integrity_hash="integrity_hash_123"
        )
        
        hash1 = record.compute_hash()
        hash2 = record.compute_hash()
        
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex length
        assert hash1 == hash2  # Should be deterministic

class TestPatientIdentity:
    """Test patient identity functionality"""
    
    def test_patient_identity_creation(self):
        """Test creating patient identity"""
        identity = PatientIdentity(
            patient_id="patient_123",
            public_key="public_key_abc",
            identity_hash="identity_hash_123",
            consent_records=[{"consent_type": "data_sharing", "granted": True}],
            access_permissions={"physicians": ["dr_smith", "dr_jones"]},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        assert identity.patient_id == "patient_123"
        assert identity.public_key == "public_key_abc"
        assert len(identity.consent_records) == 1
        assert "dr_smith" in identity.access_permissions["physicians"]
    
    def test_patient_identity_to_dict(self):
        """Test converting patient identity to dictionary"""
        identity = PatientIdentity(
            patient_id="patient_123",
            public_key="public_key_abc",
            identity_hash="identity_hash_123",
            consent_records=[],
            access_permissions={},
            created_at=1234567890.0,
            updated_at=1234567890.0
        )
        
        identity_dict = identity.to_dict()
        
        assert isinstance(identity_dict, dict)
        assert identity_dict["patient_id"] == "patient_123"
        assert identity_dict["public_key"] == "public_key_abc"

class TestAuditTrail:
    """Test audit trail functionality"""
    
    def test_audit_trail_creation(self):
        """Test creating audit trail"""
        trail = AuditTrail(
            trail_id="trail_001",
            action="store_ecg_analysis",
            actor_id="dr_smith",
            resource_id="analysis_001",
            timestamp=time.time(),
            details={"operation": "store"},
            previous_hash="prev_hash",
            current_hash="current_hash"
        )
        
        assert trail.trail_id == "trail_001"
        assert trail.action == "store_ecg_analysis"
        assert trail.actor_id == "dr_smith"
    
    def test_audit_trail_compute_hash(self):
        """Test computing audit trail hash"""
        trail = AuditTrail(
            trail_id="trail_001",
            action="store_ecg_analysis",
            actor_id="dr_smith",
            resource_id="analysis_001",
            timestamp=1234567890.0,
            details={"operation": "store"},
            previous_hash="prev_hash",
            current_hash=""
        )
        
        computed_hash = trail.compute_hash()
        
        assert isinstance(computed_hash, str)
        assert len(computed_hash) == 64  # SHA256 hex length

class TestMockBlockchainProvider:
    """Test mock blockchain provider"""
    
    @pytest.fixture
    def mock_provider(self):
        """Create mock blockchain provider"""
        return MockBlockchainProvider()
    
    @pytest.mark.asyncio
    async def test_mock_provider_connect(self, mock_provider):
        """Test mock provider connection"""
        result = await mock_provider.connect()
        assert result is True
        assert mock_provider.connected is True
    
    @pytest.mark.asyncio
    async def test_mock_provider_store_record(self, mock_provider):
        """Test storing record with mock provider"""
        await mock_provider.connect()
        
        record = ECGAnalysisRecord(
            analysis_id="test_analysis_001",
            patient_id="patient_123",
            timestamp=time.time(),
            ecg_hash="abc123def456",
            analysis_results={"heart_rate": 75},
            physician_id="dr_smith",
            facility_id="hospital_001",
            compliance_flags=["REVIEWED"],
            data_integrity_hash="integrity_hash_123"
        )
        
        record_id = await mock_provider.store_record(record)
        
        assert isinstance(record_id, str)
        assert record_id.startswith("mock_")
        assert record_id in mock_provider.records
    
    @pytest.mark.asyncio
    async def test_mock_provider_retrieve_record(self, mock_provider):
        """Test retrieving record with mock provider"""
        await mock_provider.connect()
        
        record = ECGAnalysisRecord(
            analysis_id="test_analysis_001",
            patient_id="patient_123",
            timestamp=time.time(),
            ecg_hash="abc123def456",
            analysis_results={"heart_rate": 75},
            physician_id="dr_smith",
            facility_id="hospital_001",
            compliance_flags=["REVIEWED"],
            data_integrity_hash="integrity_hash_123"
        )
        
        record_id = await mock_provider.store_record(record)
        retrieved_record = await mock_provider.retrieve_record(record_id)
        
        assert retrieved_record is not None
        assert retrieved_record.analysis_id == "test_analysis_001"
        assert retrieved_record.patient_id == "patient_123"
    
    @pytest.mark.asyncio
    async def test_mock_provider_create_identity(self, mock_provider):
        """Test creating identity with mock provider"""
        await mock_provider.connect()
        
        identity = PatientIdentity(
            patient_id="patient_123",
            public_key="public_key_abc",
            identity_hash="identity_hash_123",
            consent_records=[],
            access_permissions={},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        identity_id = await mock_provider.create_identity(identity)
        
        assert isinstance(identity_id, str)
        assert identity_id.startswith("mock_identity_")
        assert identity_id in mock_provider.identities
    
    @pytest.mark.asyncio
    async def test_mock_provider_deploy_contract(self, mock_provider):
        """Test deploying contract with mock provider"""
        await mock_provider.connect()
        
        contract = ComplianceContract(
            contract_id="contract_001",
            contract_address="",
            rules={"min_confidence": 0.8},
            regulatory_framework="FDA",
            created_at=time.time(),
            is_active=True
        )
        
        contract_address = await mock_provider.deploy_contract(contract)
        
        assert isinstance(contract_address, str)
        assert contract_address.startswith("mock_contract_")
        assert contract_address in mock_provider.contracts
    
    @pytest.mark.asyncio
    async def test_mock_provider_execute_contract(self, mock_provider):
        """Test executing contract with mock provider"""
        await mock_provider.connect()
        
        result = await mock_provider.execute_contract(
            "mock_contract_address",
            "validateECGData",
            '{"heart_rate": 75}'
        )
        
        assert isinstance(result, str)
        assert result.startswith("mock_execution_")

class TestBlockchainService:
    """Test main blockchain service"""
    
    @pytest.fixture
    def service(self):
        """Create blockchain service instance"""
        return BlockchainService(network=BlockchainNetwork.PRIVATE)
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test blockchain service initialization"""
        config = {
            "enable_ipfs": False,
            "mock_mode": True
        }
        
        result = await service.initialize(config)
        
        assert result is True
        assert service.provider is not None
        assert isinstance(service.provider, MockBlockchainProvider)
    
    @pytest.mark.asyncio
    async def test_store_ecg_analysis(self, service):
        """Test storing ECG analysis"""
        await service.initialize({"enable_ipfs": False})
        
        ecg_data = b"mock_ecg_data_bytes"
        analysis_results = {
            "heart_rate": 75,
            "rhythm": "normal",
            "confidence": 0.95
        }
        
        tx_id = await service.store_ecg_analysis(
            analysis_id="analysis_001",
            patient_id="patient_123",
            ecg_data=ecg_data,
            analysis_results=analysis_results,
            physician_id="dr_smith",
            facility_id="hospital_001"
        )
        
        assert isinstance(tx_id, str)
        assert "analysis_001" in service.analysis_records
        assert len(service.audit_trails) > 0
    
    @pytest.mark.asyncio
    async def test_retrieve_ecg_analysis(self, service):
        """Test retrieving ECG analysis"""
        await service.initialize({"enable_ipfs": False})
        
        ecg_data = b"mock_ecg_data_bytes"
        analysis_results = {"heart_rate": 75, "rhythm": "normal"}
        
        await service.store_ecg_analysis(
            analysis_id="analysis_001",
            patient_id="patient_123",
            ecg_data=ecg_data,
            analysis_results=analysis_results,
            physician_id="dr_smith",
            facility_id="hospital_001"
        )
        
        retrieved_record = await service.retrieve_ecg_analysis("analysis_001")
        
        assert retrieved_record is not None
        assert retrieved_record.analysis_id == "analysis_001"
        assert retrieved_record.patient_id == "patient_123"
        assert retrieved_record.analysis_results["heart_rate"] == 75
    
    @pytest.mark.asyncio
    async def test_create_patient_identity(self, service):
        """Test creating patient identity"""
        await service.initialize({"enable_ipfs": False})
        
        consent_records = [{"consent_type": "data_sharing", "granted": True}]
        access_permissions = {"physicians": ["dr_smith"]}
        
        tx_id = await service.create_patient_identity(
            patient_id="patient_123",
            public_key="public_key_abc",
            consent_records=consent_records,
            access_permissions=access_permissions
        )
        
        assert isinstance(tx_id, str)
        assert "patient_123" in service.patient_identities
        assert len(service.audit_trails) > 0
    
    @pytest.mark.asyncio
    async def test_deploy_compliance_contract(self, service):
        """Test deploying compliance contract"""
        await service.initialize({"enable_ipfs": False})
        
        rules = {
            "min_confidence": 0.8,
            "required_review": True
        }
        
        contract_address = await service.deploy_compliance_contract(
            contract_id="contract_001",
            rules=rules,
            regulatory_framework="FDA"
        )
        
        assert isinstance(contract_address, str)
        assert "contract_001" in service.compliance_contracts
        assert len(service.audit_trails) > 0
    
    @pytest.mark.asyncio
    async def test_check_compliance(self, service):
        """Test checking compliance"""
        await service.initialize({"enable_ipfs": False})
        
        rules = {"min_confidence": 0.8}
        await service.deploy_compliance_contract(
            contract_id="contract_001",
            rules=rules,
            regulatory_framework="FDA"
        )
        
        analysis_results = {
            "heart_rate": 75,
            "confidence": 0.95
        }
        
        compliance_result = await service.check_compliance(
            analysis_results=analysis_results,
            contract_id="contract_001"
        )
        
        assert isinstance(compliance_result, dict)
        assert "contract_id" in compliance_result
        assert "is_compliant" in compliance_result
        assert "checked_at" in compliance_result
    
    @pytest.mark.asyncio
    async def test_verify_data_integrity(self, service):
        """Test verifying data integrity"""
        await service.initialize({"enable_ipfs": False})
        
        ecg_data = b"mock_ecg_data_bytes"
        analysis_results = {"heart_rate": 75}
        
        await service.store_ecg_analysis(
            analysis_id="analysis_001",
            patient_id="patient_123",
            ecg_data=ecg_data,
            analysis_results=analysis_results,
            physician_id="dr_smith",
            facility_id="hospital_001"
        )
        
        is_valid = await service.verify_data_integrity("analysis_001")
        
        assert isinstance(is_valid, bool)
        assert is_valid is True  # Should be valid for newly stored data
    
    @pytest.mark.asyncio
    async def test_get_audit_trail(self, service):
        """Test getting audit trail"""
        await service.initialize({"enable_ipfs": False})
        
        ecg_data = b"mock_ecg_data_bytes"
        analysis_results = {"heart_rate": 75}
        
        await service.store_ecg_analysis(
            analysis_id="analysis_001",
            patient_id="patient_123",
            ecg_data=ecg_data,
            analysis_results=analysis_results,
            physician_id="dr_smith",
            facility_id="hospital_001"
        )
        
        audit_trails = await service.get_audit_trail("analysis_001")
        
        assert isinstance(audit_trails, list)
        assert len(audit_trails) > 0
        assert all(trail.resource_id == "analysis_001" for trail in audit_trails)
    
    @pytest.mark.asyncio
    async def test_get_service_stats(self, service):
        """Test getting service statistics"""
        await service.initialize({"enable_ipfs": False})
        
        stats = await service.get_service_stats()
        
        assert isinstance(stats, dict)
        assert "network" in stats
        assert "provider_connected" in stats
        assert "analysis_records_count" in stats
        assert "patient_identities_count" in stats
        assert "compliance_contracts_count" in stats
        assert "audit_trails_count" in stats
    
    def test_check_compliance_flags(self, service):
        """Test compliance flags checking"""
        analysis_results = {"confidence": 0.5}
        flags = service._check_compliance(analysis_results)
        assert "LOW_CONFIDENCE" in flags
        
        analysis_results = {"critical_findings": ["arrhythmia"]}
        flags = service._check_compliance(analysis_results)
        assert "CRITICAL_FINDINGS" in flags
        
        analysis_results = {"physician_reviewed": False}
        flags = service._check_compliance(analysis_results)
        assert "PENDING_REVIEW" in flags
        
        analysis_results = {
            "confidence": 0.95,
            "physician_reviewed": True
        }
        flags = service._check_compliance(analysis_results)
        assert len(flags) == 0

class TestEthereumProvider:
    """Test Ethereum provider (mocked)"""
    
    @pytest.fixture
    def ethereum_provider(self):
        """Create Ethereum provider with mocked dependencies"""
        with patch('app.services.blockchain_service.WEB3_AVAILABLE', True):
            with patch('app.services.blockchain_service.Web3') as mock_web3:
                with patch('app.services.blockchain_service.Account') as mock_account:
                    mock_web3_instance = Mock()
                    mock_web3.return_value = mock_web3_instance
                    mock_web3_instance.is_connected.return_value = True
                    
                    mock_account_instance = Mock()
                    mock_account_instance.address = "0x123456789"
                    mock_account.from_key.return_value = mock_account_instance
                    
                    provider = EthereumProvider(
                        rpc_url="http://localhost:8545",
                        private_key="0xprivatekey"
                    )
                    provider.w3 = mock_web3_instance
                    provider.account = mock_account_instance
                    
                    return provider
    
    @pytest.mark.asyncio
    async def test_ethereum_provider_connect(self, ethereum_provider):
        """Test Ethereum provider connection"""
        result = await ethereum_provider.connect()
        assert result is True

class TestHyperledgerProvider:
    """Test Hyperledger provider"""
    
    @pytest.fixture
    def hyperledger_provider(self):
        """Create Hyperledger provider"""
        return HyperledgerProvider(network_config={})
    
    @pytest.mark.asyncio
    async def test_hyperledger_provider_connect(self, hyperledger_provider):
        """Test Hyperledger provider connection"""
        result = await hyperledger_provider.connect()
        assert result is True
        assert hyperledger_provider.connected is True
    
    @pytest.mark.asyncio
    async def test_hyperledger_provider_store_record(self, hyperledger_provider):
        """Test storing record with Hyperledger provider"""
        await hyperledger_provider.connect()
        
        record = ECGAnalysisRecord(
            analysis_id="test_analysis_001",
            patient_id="patient_123",
            timestamp=time.time(),
            ecg_hash="abc123def456",
            analysis_results={"heart_rate": 75},
            physician_id="dr_smith",
            facility_id="hospital_001",
            compliance_flags=["REVIEWED"],
            data_integrity_hash="integrity_hash_123"
        )
        
        record_id = await hyperledger_provider.store_record(record)
        
        assert isinstance(record_id, str)
        assert record_id.startswith("hlf_")

class TestIPFSStorage:
    """Test IPFS storage (mocked)"""
    
    @pytest.fixture
    def ipfs_storage(self):
        """Create IPFS storage with mocked client"""
        with patch('app.services.blockchain_service.IPFS_AVAILABLE', True):
            with patch('app.services.blockchain_service.ipfshttpclient') as mock_ipfs:
                mock_client = Mock()
                mock_ipfs.connect.return_value = mock_client
                
                storage = IPFSStorage()
                storage.client = mock_client
                
                return storage
    
    @pytest.mark.asyncio
    async def test_ipfs_storage_connect(self, ipfs_storage):
        """Test IPFS storage connection"""
        result = await ipfs_storage.connect()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_ipfs_storage_store_data(self, ipfs_storage):
        """Test storing data on IPFS"""
        ipfs_storage.client.add_bytes.return_value = "QmTestHash123"
        
        ecg_data = b"mock_ecg_data_bytes"
        ipfs_hash = await ipfs_storage.store_ecg_data(ecg_data)
        
        assert ipfs_hash == "QmTestHash123"
        ipfs_storage.client.add_bytes.assert_called_once_with(ecg_data)
    
    @pytest.mark.asyncio
    async def test_ipfs_storage_retrieve_data(self, ipfs_storage):
        """Test retrieving data from IPFS"""
        expected_data = b"mock_ecg_data_bytes"
        ipfs_storage.client.cat.return_value = expected_data
        
        retrieved_data = await ipfs_storage.retrieve_ecg_data("QmTestHash123")
        
        assert retrieved_data == expected_data
        ipfs_storage.client.cat.assert_called_once_with("QmTestHash123")

class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_ecg_analysis_workflow(self):
        """Test complete ECG analysis workflow with blockchain"""
        service = BlockchainService(network=BlockchainNetwork.PRIVATE)
        await service.initialize({"enable_ipfs": False})
        
        await service.create_patient_identity(
            patient_id="patient_123",
            public_key="public_key_abc",
            consent_records=[{"consent_type": "data_sharing", "granted": True}],
            access_permissions={"physicians": ["dr_smith"]}
        )
        
        await service.deploy_compliance_contract(
            contract_id="fda_contract",
            rules={"min_confidence": 0.8},
            regulatory_framework="FDA"
        )
        
        ecg_data = b"mock_ecg_data_bytes"
        analysis_results = {
            "heart_rate": 75,
            "rhythm": "normal",
            "confidence": 0.95,
            "physician_reviewed": True
        }
        
        tx_id = await service.store_ecg_analysis(
            analysis_id="analysis_001",
            patient_id="patient_123",
            ecg_data=ecg_data,
            analysis_results=analysis_results,
            physician_id="dr_smith",
            facility_id="hospital_001"
        )
        
        compliance_result = await service.check_compliance(
            analysis_results=analysis_results,
            contract_id="fda_contract"
        )
        
        is_valid = await service.verify_data_integrity("analysis_001")
        
        audit_trails = await service.get_audit_trail("analysis_001")
        
        assert isinstance(tx_id, str)
        assert compliance_result["is_compliant"] is True
        assert is_valid is True
        assert len(audit_trails) > 0
        assert len(service.analysis_records) == 1
        assert len(service.patient_identities) == 1
        assert len(service.compliance_contracts) == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling scenarios"""
        service = BlockchainService(network=BlockchainNetwork.PRIVATE)
        
        with pytest.raises(RuntimeError, match="Blockchain service not initialized"):
            await service.store_ecg_analysis(
                analysis_id="analysis_001",
                patient_id="patient_123",
                ecg_data=b"data",
                analysis_results={},
                physician_id="dr_smith",
                facility_id="hospital_001"
            )
        
        await service.initialize({"enable_ipfs": False})
        
        with pytest.raises(ValueError, match="Compliance contract .* not found"):
            await service.check_compliance(
                analysis_results={"heart_rate": 75},
                contract_id="non_existent_contract"
            )
    
    @pytest.mark.asyncio
    async def test_global_service_instance(self):
        """Test global blockchain service instance"""
        assert blockchain_service is not None
        assert isinstance(blockchain_service, BlockchainService)
        
        result = await blockchain_service.initialize({"enable_ipfs": False})
        assert result is True
        
        stats = await blockchain_service.get_service_stats()
        assert isinstance(stats, dict)
        assert "network" in stats
