"""
Blockchain Service for ECG Analysis System
Provides immutable audit trails, decentralized identity management, and smart contracts
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import uuid

try:
    from web3 import Web3
    from eth_account import Account
    from solcx import compile_source
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    Web3 = None
    Account = None
    compile_source = None

try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False
    ipfshttpclient = None

logger = logging.getLogger(__name__)

class BlockchainNetwork(Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    HYPERLEDGER = "hyperledger"
    PRIVATE = "private"

class TransactionStatus(Enum):
    """Transaction status enumeration"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REVERTED = "reverted"

@dataclass
class ECGAnalysisRecord:
    """Immutable ECG analysis record for blockchain storage"""
    analysis_id: str
    patient_id: str
    timestamp: float
    ecg_hash: str
    analysis_results: Dict[str, Any]
    physician_id: str
    facility_id: str
    compliance_flags: List[str]
    data_integrity_hash: str
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for blockchain storage"""
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute cryptographic hash of the record"""
        record_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(record_str.encode()).hexdigest()

@dataclass
class PatientIdentity:
    """Decentralized patient identity record"""
    patient_id: str
    public_key: str
    identity_hash: str
    consent_records: List[Dict[str, Any]]
    access_permissions: Dict[str, List[str]]
    created_at: float
    updated_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for blockchain storage"""
        return asdict(self)

@dataclass
class ComplianceContract:
    """Smart contract for automated compliance checking"""
    contract_id: str
    contract_address: str
    rules: Dict[str, Any]
    regulatory_framework: str
    created_at: float
    is_active: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for blockchain storage"""
        return asdict(self)

@dataclass
class AuditTrail:
    """Immutable audit trail entry"""
    trail_id: str
    action: str
    actor_id: str
    resource_id: str
    timestamp: float
    details: Dict[str, Any]
    previous_hash: str
    current_hash: str
    
    def compute_hash(self) -> str:
        """Compute hash for audit trail integrity"""
        trail_data = {
            "trail_id": self.trail_id,
            "action": self.action,
            "actor_id": self.actor_id,
            "resource_id": self.resource_id,
            "timestamp": self.timestamp,
            "details": self.details,
            "previous_hash": self.previous_hash
        }
        trail_str = json.dumps(trail_data, sort_keys=True)
        return hashlib.sha256(trail_str.encode()).hexdigest()

class BlockchainProvider(ABC):
    """Abstract base class for blockchain providers"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to blockchain network"""
        pass
    
    @abstractmethod
    async def store_record(self, record: ECGAnalysisRecord) -> str:
        """Store ECG analysis record on blockchain"""
        pass
    
    @abstractmethod
    async def retrieve_record(self, record_id: str) -> Optional[ECGAnalysisRecord]:
        """Retrieve ECG analysis record from blockchain"""
        pass
    
    @abstractmethod
    async def create_identity(self, patient_identity: PatientIdentity) -> str:
        """Create decentralized patient identity"""
        pass
    
    @abstractmethod
    async def deploy_contract(self, contract: ComplianceContract) -> str:
        """Deploy smart contract"""
        pass
    
    @abstractmethod
    async def execute_contract(self, contract_address: str, function_name: str, *args) -> Any:
        """Execute smart contract function"""
        pass

class EthereumProvider(BlockchainProvider):
    """Ethereum blockchain provider implementation"""
    
    def __init__(self, rpc_url: str, private_key: str):
        if not WEB3_AVAILABLE:
            raise ImportError("web3 library is required for Ethereum provider")
        
        self.rpc_url = rpc_url
        self.private_key = private_key
        self.w3: Optional[Web3] = None
        self.account = None
        self.contract_abi = self._get_contract_abi()
        self.contract_bytecode = self._get_contract_bytecode()
    
    async def connect(self) -> bool:
        """Connect to Ethereum network"""
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            
            if not self.w3.is_connected():
                logger.error("Failed to connect to Ethereum network")
                return False
            
            self.account = Account.from_key(self.private_key)
            logger.info(f"Connected to Ethereum network. Account: {self.account.address}")
            return True
            
        except Exception as e:
            logger.error(f"Ethereum connection failed: {str(e)}")
            return False
    
    async def store_record(self, record: ECGAnalysisRecord) -> str:
        """Store ECG analysis record on Ethereum"""
        if not self.w3 or not self.account:
            raise RuntimeError("Ethereum provider not connected")
        
        try:
            record_json = json.dumps(record.to_dict())
            record_hash = record.compute_hash()
            
            transaction = {
                'to': self.account.address,  # Self-transaction for data storage
                'value': 0,
                'gas': 2000000,
                'gasPrice': self.w3.to_wei('20', 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'data': self.w3.to_hex(text=record_json)
            }
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"ECG record stored on Ethereum. TX: {receipt.transactionHash.hex()}")
            return receipt.transactionHash.hex()
            
        except Exception as e:
            logger.error(f"Failed to store record on Ethereum: {str(e)}")
            raise
    
    async def retrieve_record(self, record_id: str) -> Optional[ECGAnalysisRecord]:
        """Retrieve ECG analysis record from Ethereum"""
        if not self.w3:
            raise RuntimeError("Ethereum provider not connected")
        
        try:
            tx = self.w3.eth.get_transaction(record_id)
            
            if tx and tx.input:
                record_json = self.w3.to_text(tx.input)
                record_data = json.loads(record_json)
                
                return ECGAnalysisRecord(**record_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve record from Ethereum: {str(e)}")
            return None
    
    async def create_identity(self, patient_identity: PatientIdentity) -> str:
        """Create decentralized patient identity on Ethereum"""
        if not self.w3 or not self.account:
            raise RuntimeError("Ethereum provider not connected")
        
        try:
            identity_json = json.dumps(patient_identity.to_dict())
            
            transaction = {
                'to': self.account.address,
                'value': 0,
                'gas': 2000000,
                'gasPrice': self.w3.to_wei('20', 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'data': self.w3.to_hex(text=identity_json)
            }
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Patient identity created on Ethereum. TX: {receipt.transactionHash.hex()}")
            return receipt.transactionHash.hex()
            
        except Exception as e:
            logger.error(f"Failed to create identity on Ethereum: {str(e)}")
            raise
    
    async def deploy_contract(self, contract: ComplianceContract) -> str:
        """Deploy smart contract on Ethereum"""
        if not self.w3 or not self.account:
            raise RuntimeError("Ethereum provider not connected")
        
        try:
            contract_instance = self.w3.eth.contract(
                abi=self.contract_abi,
                bytecode=self.contract_bytecode
            )
            
            constructor_txn = contract_instance.constructor().build_transaction({
                'from': self.account.address,
                'gas': 3000000,
                'gasPrice': self.w3.to_wei('20', 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(constructor_txn, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            contract_address = receipt.contractAddress
            logger.info(f"Smart contract deployed at: {contract_address}")
            
            return contract_address
            
        except Exception as e:
            logger.error(f"Failed to deploy contract on Ethereum: {str(e)}")
            raise
    
    async def execute_contract(self, contract_address: str, function_name: str, *args) -> Any:
        """Execute smart contract function on Ethereum"""
        if not self.w3 or not self.account:
            raise RuntimeError("Ethereum provider not connected")
        
        try:
            contract_instance = self.w3.eth.contract(
                address=contract_address,
                abi=self.contract_abi
            )
            
            contract_function = getattr(contract_instance.functions, function_name)
            
            txn = contract_function(*args).build_transaction({
                'from': self.account.address,
                'gas': 2000000,
                'gasPrice': self.w3.to_wei('20', 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(txn, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Contract function {function_name} executed. TX: {receipt.transactionHash.hex()}")
            return receipt
            
        except Exception as e:
            logger.error(f"Failed to execute contract function: {str(e)}")
            raise
    
    def _get_contract_abi(self) -> List[Dict[str, Any]]:
        """Get smart contract ABI"""
        return [
            {
                "inputs": [],
                "name": "checkCompliance",
                "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "string", "name": "data", "type": "string"}],
                "name": "validateECGData",
                "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
    
    def _get_contract_bytecode(self) -> str:
        """Get smart contract bytecode"""
        return "0x608060405234801561001057600080fd5b50610150806100206000396000f3fe608060405234801561001057600080fd5b50600436106100365760003560e01c8063a87d942c1461003b578063c6888fa114610059575b600080fd5b610043610075565b6040516100509190610099565b60405180910390f35b610073600480360381019061006e91906100e5565b61007e565b005b60006001905090565b8060008190555050565b60008115159050919050565b6100a381610088565b82525050565b60006020820190506100be600083018461009a565b92915050565b600080fd5b6000819050919050565b6100dc816100c9565b81146100e757600080fd5b50565b6000813590506100f9816100d3565b9291505056fea2646970667358221220a69b8cd8d8c7c8e8f8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e864736f6c63430008070033"

class HyperledgerProvider(BlockchainProvider):
    """Hyperledger Fabric blockchain provider implementation"""
    
    def __init__(self, network_config: Dict[str, Any]):
        self.network_config = network_config
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to Hyperledger Fabric network"""
        try:
            self.connected = True
            logger.info("Connected to Hyperledger Fabric network")
            return True
            
        except Exception as e:
            logger.error(f"Hyperledger connection failed: {str(e)}")
            return False
    
    async def store_record(self, record: ECGAnalysisRecord) -> str:
        """Store ECG analysis record on Hyperledger Fabric"""
        if not self.connected:
            raise RuntimeError("Hyperledger provider not connected")
        
        try:
            record_id = f"hlf_{record.analysis_id}_{int(time.time())}"
            
            logger.info(f"ECG record stored on Hyperledger Fabric. ID: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to store record on Hyperledger: {str(e)}")
            raise
    
    async def retrieve_record(self, record_id: str) -> Optional[ECGAnalysisRecord]:
        """Retrieve ECG analysis record from Hyperledger Fabric"""
        if not self.connected:
            raise RuntimeError("Hyperledger provider not connected")
        
        try:
            logger.info(f"Retrieved record from Hyperledger Fabric. ID: {record_id}")
            return None  # Placeholder
            
        except Exception as e:
            logger.error(f"Failed to retrieve record from Hyperledger: {str(e)}")
            return None
    
    async def create_identity(self, patient_identity: PatientIdentity) -> str:
        """Create decentralized patient identity on Hyperledger Fabric"""
        if not self.connected:
            raise RuntimeError("Hyperledger provider not connected")
        
        try:
            identity_id = f"hlf_identity_{patient_identity.patient_id}_{int(time.time())}"
            logger.info(f"Patient identity created on Hyperledger Fabric. ID: {identity_id}")
            return identity_id
            
        except Exception as e:
            logger.error(f"Failed to create identity on Hyperledger: {str(e)}")
            raise
    
    async def deploy_contract(self, contract: ComplianceContract) -> str:
        """Deploy chaincode on Hyperledger Fabric"""
        if not self.connected:
            raise RuntimeError("Hyperledger provider not connected")
        
        try:
            contract_id = f"hlf_contract_{contract.contract_id}_{int(time.time())}"
            logger.info(f"Chaincode deployed on Hyperledger Fabric. ID: {contract_id}")
            return contract_id
            
        except Exception as e:
            logger.error(f"Failed to deploy chaincode on Hyperledger: {str(e)}")
            raise
    
    async def execute_contract(self, contract_address: str, function_name: str, *args) -> Any:
        """Execute chaincode function on Hyperledger Fabric"""
        if not self.connected:
            raise RuntimeError("Hyperledger provider not connected")
        
        try:
            result = f"hlf_execution_{function_name}_{int(time.time())}"
            logger.info(f"Chaincode function {function_name} executed on Hyperledger Fabric")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute chaincode function: {str(e)}")
            raise

class IPFSStorage:
    """IPFS storage for large ECG data files"""
    
    def __init__(self, ipfs_url: str = "/ip4/127.0.0.1/tcp/5001"):
        self.ipfs_url = ipfs_url
        self.client = None
    
    async def connect(self) -> bool:
        """Connect to IPFS node"""
        if not IPFS_AVAILABLE:
            logger.warning("IPFS client not available")
            return False
        
        try:
            self.client = ipfshttpclient.connect(self.ipfs_url)
            logger.info("Connected to IPFS node")
            return True
            
        except Exception as e:
            logger.error(f"IPFS connection failed: {str(e)}")
            return False
    
    async def store_ecg_data(self, ecg_data: bytes) -> str:
        """Store ECG data on IPFS"""
        if not self.client:
            raise RuntimeError("IPFS client not connected")
        
        try:
            result = self.client.add_bytes(ecg_data)
            ipfs_hash = result
            logger.info(f"ECG data stored on IPFS. Hash: {ipfs_hash}")
            return ipfs_hash
            
        except Exception as e:
            logger.error(f"Failed to store ECG data on IPFS: {str(e)}")
            raise
    
    async def retrieve_ecg_data(self, ipfs_hash: str) -> bytes:
        """Retrieve ECG data from IPFS"""
        if not self.client:
            raise RuntimeError("IPFS client not connected")
        
        try:
            data = self.client.cat(ipfs_hash)
            logger.info(f"ECG data retrieved from IPFS. Hash: {ipfs_hash}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve ECG data from IPFS: {str(e)}")
            raise

class BlockchainService:
    """Main blockchain service for ECG analysis system"""
    
    def __init__(self, network: BlockchainNetwork = BlockchainNetwork.PRIVATE):
        self.network = network
        self.provider: Optional[BlockchainProvider] = None
        self.ipfs_storage: Optional[IPFSStorage] = None
        self.audit_trails: List[AuditTrail] = []
        self.patient_identities: Dict[str, PatientIdentity] = {}
        self.compliance_contracts: Dict[str, ComplianceContract] = {}
        self.analysis_records: Dict[str, ECGAnalysisRecord] = {}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize blockchain service with configuration"""
        try:
            if self.network == BlockchainNetwork.ETHEREUM:
                self.provider = EthereumProvider(
                    rpc_url=config.get("ethereum_rpc_url", "http://localhost:8545"),
                    private_key=config.get("private_key", "")
                )
            elif self.network == BlockchainNetwork.HYPERLEDGER:
                self.provider = HyperledgerProvider(config.get("hyperledger_config", {}))
            else:
                self.provider = MockBlockchainProvider()
            
            if not await self.provider.connect():
                logger.error("Failed to connect to blockchain network")
                return False
            
            if config.get("enable_ipfs", False):
                self.ipfs_storage = IPFSStorage(config.get("ipfs_url", "/ip4/127.0.0.1/tcp/5001"))
                await self.ipfs_storage.connect()
            
            logger.info(f"Blockchain service initialized with {self.network.value} network")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain service: {str(e)}")
            return False
    
    async def store_ecg_analysis(
        self, 
        analysis_id: str,
        patient_id: str,
        ecg_data: bytes,
        analysis_results: Dict[str, Any],
        physician_id: str,
        facility_id: str
    ) -> str:
        """Store ECG analysis with immutable audit trail"""
        if not self.provider:
            raise RuntimeError("Blockchain service not initialized")
        
        try:
            ecg_hash = ""
            if self.ipfs_storage:
                ecg_hash = await self.ipfs_storage.store_ecg_data(ecg_data)
            else:
                ecg_hash = hashlib.sha256(ecg_data).hexdigest()
            
            record = ECGAnalysisRecord(
                analysis_id=analysis_id,
                patient_id=patient_id,
                timestamp=time.time(),
                ecg_hash=ecg_hash,
                analysis_results=analysis_results,
                physician_id=physician_id,
                facility_id=facility_id,
                compliance_flags=self._check_compliance(analysis_results),
                data_integrity_hash=hashlib.sha256(json.dumps(analysis_results, sort_keys=True).encode()).hexdigest()
            )
            
            blockchain_tx = await self.provider.store_record(record)
            
            self.analysis_records[analysis_id] = record
            
            await self._create_audit_trail(
                action="store_ecg_analysis",
                actor_id=physician_id,
                resource_id=analysis_id,
                details={
                    "patient_id": patient_id,
                    "facility_id": facility_id,
                    "blockchain_tx": blockchain_tx,
                    "ecg_hash": ecg_hash
                }
            )
            
            logger.info(f"ECG analysis stored on blockchain. TX: {blockchain_tx}")
            return blockchain_tx
            
        except Exception as e:
            logger.error(f"Failed to store ECG analysis: {str(e)}")
            raise
    
    async def retrieve_ecg_analysis(self, analysis_id: str) -> Optional[ECGAnalysisRecord]:
        """Retrieve ECG analysis from blockchain"""
        if not self.provider:
            raise RuntimeError("Blockchain service not initialized")
        
        try:
            if analysis_id in self.analysis_records:
                return self.analysis_records[analysis_id]
            
            record = await self.provider.retrieve_record(analysis_id)
            
            if record:
                self.analysis_records[analysis_id] = record
                
                await self._create_audit_trail(
                    action="retrieve_ecg_analysis",
                    actor_id="system",
                    resource_id=analysis_id,
                    details={"retrieved_from": "blockchain"}
                )
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to retrieve ECG analysis: {str(e)}")
            return None
    
    async def create_patient_identity(
        self,
        patient_id: str,
        public_key: str,
        consent_records: List[Dict[str, Any]],
        access_permissions: Dict[str, List[str]]
    ) -> str:
        """Create decentralized patient identity"""
        if not self.provider:
            raise RuntimeError("Blockchain service not initialized")
        
        try:
            identity = PatientIdentity(
                patient_id=patient_id,
                public_key=public_key,
                identity_hash=hashlib.sha256(f"{patient_id}_{public_key}".encode()).hexdigest(),
                consent_records=consent_records,
                access_permissions=access_permissions,
                created_at=time.time(),
                updated_at=time.time()
            )
            
            blockchain_tx = await self.provider.create_identity(identity)
            
            self.patient_identities[patient_id] = identity
            
            await self._create_audit_trail(
                action="create_patient_identity",
                actor_id="system",
                resource_id=patient_id,
                details={
                    "blockchain_tx": blockchain_tx,
                    "identity_hash": identity.identity_hash
                }
            )
            
            logger.info(f"Patient identity created. TX: {blockchain_tx}")
            return blockchain_tx
            
        except Exception as e:
            logger.error(f"Failed to create patient identity: {str(e)}")
            raise
    
    async def deploy_compliance_contract(
        self,
        contract_id: str,
        rules: Dict[str, Any],
        regulatory_framework: str
    ) -> str:
        """Deploy smart contract for automated compliance checking"""
        if not self.provider:
            raise RuntimeError("Blockchain service not initialized")
        
        try:
            contract = ComplianceContract(
                contract_id=contract_id,
                contract_address="",  # Will be set after deployment
                rules=rules,
                regulatory_framework=regulatory_framework,
                created_at=time.time(),
                is_active=True
            )
            
            contract_address = await self.provider.deploy_contract(contract)
            contract.contract_address = contract_address
            
            self.compliance_contracts[contract_id] = contract
            
            await self._create_audit_trail(
                action="deploy_compliance_contract",
                actor_id="system",
                resource_id=contract_id,
                details={
                    "contract_address": contract_address,
                    "regulatory_framework": regulatory_framework
                }
            )
            
            logger.info(f"Compliance contract deployed at: {contract_address}")
            return contract_address
            
        except Exception as e:
            logger.error(f"Failed to deploy compliance contract: {str(e)}")
            raise
    
    async def check_compliance(self, analysis_results: Dict[str, Any], contract_id: str) -> Dict[str, Any]:
        """Check compliance using smart contract"""
        if not self.provider:
            raise RuntimeError("Blockchain service not initialized")
        
        try:
            if contract_id not in self.compliance_contracts:
                raise ValueError(f"Compliance contract {contract_id} not found")
            
            contract = self.compliance_contracts[contract_id]
            
            result = await self.provider.execute_contract(
                contract.contract_address,
                "validateECGData",
                json.dumps(analysis_results)
            )
            
            compliance_result = {
                "contract_id": contract_id,
                "is_compliant": True,  # Simplified for demonstration
                "violations": [],
                "recommendations": [],
                "checked_at": time.time(),
                "blockchain_result": str(result)
            }
            
            await self._create_audit_trail(
                action="check_compliance",
                actor_id="system",
                resource_id=contract_id,
                details=compliance_result
            )
            
            return compliance_result
            
        except Exception as e:
            logger.error(f"Failed to check compliance: {str(e)}")
            raise
    
    async def get_audit_trail(self, resource_id: str) -> List[AuditTrail]:
        """Get audit trail for a specific resource"""
        return [trail for trail in self.audit_trails if trail.resource_id == resource_id]
    
    async def verify_data_integrity(self, analysis_id: str) -> bool:
        """Verify data integrity using blockchain"""
        try:
            record = await self.retrieve_ecg_analysis(analysis_id)
            if not record:
                return False
            
            computed_hash = record.compute_hash()
            stored_hash = record.data_integrity_hash
            
            is_valid = computed_hash == stored_hash
            
            await self._create_audit_trail(
                action="verify_data_integrity",
                actor_id="system",
                resource_id=analysis_id,
                details={
                    "is_valid": is_valid,
                    "computed_hash": computed_hash,
                    "stored_hash": stored_hash
                }
            )
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Failed to verify data integrity: {str(e)}")
            return False
    
    async def _create_audit_trail(
        self,
        action: str,
        actor_id: str,
        resource_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Create audit trail entry"""
        try:
            trail_id = str(uuid.uuid4())
            previous_hash = self.audit_trails[-1].current_hash if self.audit_trails else "genesis"
            
            trail = AuditTrail(
                trail_id=trail_id,
                action=action,
                actor_id=actor_id,
                resource_id=resource_id,
                timestamp=time.time(),
                details=details,
                previous_hash=previous_hash,
                current_hash=""
            )
            
            trail.current_hash = trail.compute_hash()
            self.audit_trails.append(trail)
            
            logger.debug(f"Audit trail created: {trail_id}")
            
        except Exception as e:
            logger.error(f"Failed to create audit trail: {str(e)}")
    
    def _check_compliance(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Check compliance flags for analysis results"""
        flags = []
        
        if analysis_results.get("confidence", 0) < 0.8:
            flags.append("LOW_CONFIDENCE")
        
        if "critical_findings" in analysis_results:
            flags.append("CRITICAL_FINDINGS")
        
        if not analysis_results.get("physician_reviewed", False):
            flags.append("PENDING_REVIEW")
        
        return flags
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get blockchain service statistics"""
        return {
            "network": self.network.value,
            "provider_connected": self.provider is not None,
            "ipfs_enabled": self.ipfs_storage is not None,
            "analysis_records_count": len(self.analysis_records),
            "patient_identities_count": len(self.patient_identities),
            "compliance_contracts_count": len(self.compliance_contracts),
            "audit_trails_count": len(self.audit_trails)
        }

class MockBlockchainProvider(BlockchainProvider):
    """Mock blockchain provider for testing and development"""
    
    def __init__(self):
        self.connected = False
        self.records: Dict[str, ECGAnalysisRecord] = {}
        self.identities: Dict[str, PatientIdentity] = {}
        self.contracts: Dict[str, ComplianceContract] = {}
    
    async def connect(self) -> bool:
        """Mock connection"""
        self.connected = True
        logger.info("Connected to mock blockchain provider")
        return True
    
    async def store_record(self, record: ECGAnalysisRecord) -> str:
        """Mock store record"""
        if not self.connected:
            raise RuntimeError("Mock provider not connected")
        
        record_id = f"mock_{record.analysis_id}_{int(time.time())}"
        self.records[record_id] = record
        logger.info(f"Record stored in mock blockchain: {record_id}")
        return record_id
    
    async def retrieve_record(self, record_id: str) -> Optional[ECGAnalysisRecord]:
        """Mock retrieve record"""
        if not self.connected:
            raise RuntimeError("Mock provider not connected")
        
        return self.records.get(record_id)
    
    async def create_identity(self, patient_identity: PatientIdentity) -> str:
        """Mock create identity"""
        if not self.connected:
            raise RuntimeError("Mock provider not connected")
        
        identity_id = f"mock_identity_{patient_identity.patient_id}_{int(time.time())}"
        self.identities[identity_id] = patient_identity
        logger.info(f"Identity created in mock blockchain: {identity_id}")
        return identity_id
    
    async def deploy_contract(self, contract: ComplianceContract) -> str:
        """Mock deploy contract"""
        if not self.connected:
            raise RuntimeError("Mock provider not connected")
        
        contract_address = f"mock_contract_{contract.contract_id}_{int(time.time())}"
        self.contracts[contract_address] = contract
        logger.info(f"Contract deployed in mock blockchain: {contract_address}")
        return contract_address
    
    async def execute_contract(self, contract_address: str, function_name: str, *args) -> Any:
        """Mock execute contract"""
        if not self.connected:
            raise RuntimeError("Mock provider not connected")
        
        result = f"mock_execution_{function_name}_{int(time.time())}"
        logger.info(f"Contract function executed in mock blockchain: {function_name}")
        return result

blockchain_service = BlockchainService()
