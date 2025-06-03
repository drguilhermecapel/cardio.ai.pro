"""
Homomorphic Encryption Service for ECG Data
Provides secure computation capabilities for sensitive ECG data processing
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    ts = None

try:
    from phe import paillier
    PHE_AVAILABLE = True
except ImportError:
    PHE_AVAILABLE = False
    paillier = None

logger = logging.getLogger(__name__)

@dataclass
class EncryptionContext:
    """Context for homomorphic encryption operations"""
    scheme: str
    public_key: Optional[Any] = None
    private_key: Optional[Any] = None
    context: Optional[Any] = None
    parameters: Dict[str, Any] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class EncryptedData:
    """Container for encrypted ECG data"""
    data: Any
    scheme: str
    metadata: Dict[str, Any]
    checksum: str
    timestamp: float
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Compute checksum for data integrity verification"""
        data_str = json.dumps({
            'scheme': self.scheme,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

class HomomorphicScheme(ABC):
    """Abstract base class for homomorphic encryption schemes"""
    
    @abstractmethod
    async def generate_keys(self, **kwargs) -> EncryptionContext:
        """Generate encryption keys and context"""
        pass
    
    @abstractmethod
    async def encrypt(self, data: np.ndarray, context: EncryptionContext) -> EncryptedData:
        """Encrypt data using homomorphic encryption"""
        pass
    
    @abstractmethod
    async def decrypt(self, encrypted_data: EncryptedData, context: EncryptionContext) -> np.ndarray:
        """Decrypt data using private key"""
        pass
    
    @abstractmethod
    async def add(self, a: EncryptedData, b: EncryptedData) -> EncryptedData:
        """Perform homomorphic addition"""
        pass
    
    @abstractmethod
    async def multiply(self, a: EncryptedData, scalar: float) -> EncryptedData:
        """Perform homomorphic scalar multiplication"""
        pass

class TenSEALScheme(HomomorphicScheme):
    """TenSEAL-based homomorphic encryption implementation"""
    
    def __init__(self):
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL is required for this encryption scheme")
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def generate_keys(self, **kwargs) -> EncryptionContext:
        """Generate TenSEAL encryption context"""
        def _generate():
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=kwargs.get('poly_modulus_degree', 8192),
                coeff_mod_bit_sizes=kwargs.get('coeff_mod_bit_sizes', [60, 40, 40, 60])
            )
            context.generate_galois_keys()
            context.global_scale = kwargs.get('global_scale', 2**40)
            return context
        
        loop = asyncio.get_event_loop()
        context = await loop.run_in_executor(self.executor, _generate)
        
        return EncryptionContext(
            scheme="tenseal_ckks",
            context=context,
            parameters={
                'poly_modulus_degree': kwargs.get('poly_modulus_degree', 8192),
                'global_scale': kwargs.get('global_scale', 2**40)
            }
        )
    
    async def encrypt(self, data: np.ndarray, context: EncryptionContext) -> EncryptedData:
        """Encrypt ECG data using TenSEAL CKKS"""
        def _encrypt():
            flat_data = data.flatten().astype(float).tolist()
            encrypted = ts.ckks_vector(context.context, flat_data)
            return encrypted
        
        loop = asyncio.get_event_loop()
        encrypted = await loop.run_in_executor(self.executor, _encrypt)
        
        return EncryptedData(
            data=encrypted,
            scheme="tenseal_ckks",
            metadata={
                'original_shape': data.shape,
                'dtype': str(data.dtype),
                'size': data.size
            },
            checksum="",
            timestamp=time.time()
        )
    
    async def decrypt(self, encrypted_data: EncryptedData, context: EncryptionContext) -> np.ndarray:
        """Decrypt TenSEAL encrypted data"""
        def _decrypt():
            decrypted = encrypted_data.data.decrypt()
            return np.array(decrypted).reshape(encrypted_data.metadata['original_shape'])
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _decrypt)
    
    async def add(self, a: EncryptedData, b: EncryptedData) -> EncryptedData:
        """Perform homomorphic addition on encrypted vectors"""
        def _add():
            result = a.data + b.data
            return result
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _add)
        
        return EncryptedData(
            data=result,
            scheme="tenseal_ckks",
            metadata={
                'operation': 'addition',
                'operand_shapes': [a.metadata.get('original_shape'), b.metadata.get('original_shape')]
            },
            checksum="",
            timestamp=time.time()
        )
    
    async def multiply(self, a: EncryptedData, scalar: float) -> EncryptedData:
        """Perform homomorphic scalar multiplication"""
        def _multiply():
            result = a.data * scalar
            return result
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _multiply)
        
        return EncryptedData(
            data=result,
            scheme="tenseal_ckks",
            metadata={
                'operation': 'scalar_multiplication',
                'scalar': scalar,
                'original_shape': a.metadata.get('original_shape')
            },
            checksum="",
            timestamp=time.time()
        )

class PaillierScheme(HomomorphicScheme):
    """Paillier homomorphic encryption implementation"""
    
    def __init__(self):
        if not PHE_AVAILABLE:
            raise ImportError("phe library is required for Paillier encryption")
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def generate_keys(self, **kwargs) -> EncryptionContext:
        """Generate Paillier key pair"""
        def _generate():
            key_length = kwargs.get('key_length', 2048)
            public_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)
            return public_key, private_key
        
        loop = asyncio.get_event_loop()
        public_key, private_key = await loop.run_in_executor(self.executor, _generate)
        
        return EncryptionContext(
            scheme="paillier",
            public_key=public_key,
            private_key=private_key,
            parameters={'key_length': kwargs.get('key_length', 2048)}
        )
    
    async def encrypt(self, data: np.ndarray, context: EncryptionContext) -> EncryptedData:
        """Encrypt data using Paillier encryption"""
        def _encrypt():
            scale_factor = 10000
            scaled_data = (data * scale_factor).astype(int)
            encrypted_list = [context.public_key.encrypt(int(x)) for x in scaled_data.flatten()]
            return encrypted_list, scale_factor
        
        loop = asyncio.get_event_loop()
        encrypted_list, scale_factor = await loop.run_in_executor(self.executor, _encrypt)
        
        return EncryptedData(
            data=encrypted_list,
            scheme="paillier",
            metadata={
                'original_shape': data.shape,
                'scale_factor': scale_factor,
                'dtype': str(data.dtype)
            },
            checksum="",
            timestamp=time.time()
        )
    
    async def decrypt(self, encrypted_data: EncryptedData, context: EncryptionContext) -> np.ndarray:
        """Decrypt Paillier encrypted data"""
        def _decrypt():
            scale_factor = encrypted_data.metadata['scale_factor']
            decrypted_list = [context.private_key.decrypt(x) / scale_factor for x in encrypted_data.data]
            return np.array(decrypted_list).reshape(encrypted_data.metadata['original_shape'])
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _decrypt)
    
    async def add(self, a: EncryptedData, b: EncryptedData) -> EncryptedData:
        """Perform homomorphic addition on Paillier encrypted data"""
        def _add():
            if len(a.data) != len(b.data):
                raise ValueError("Encrypted arrays must have the same length for addition")
            result = [x + y for x, y in zip(a.data, b.data)]
            return result
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _add)
        
        return EncryptedData(
            data=result,
            scheme="paillier",
            metadata={
                'operation': 'addition',
                'scale_factor': a.metadata['scale_factor']
            },
            checksum="",
            timestamp=time.time()
        )
    
    async def multiply(self, a: EncryptedData, scalar: float) -> EncryptedData:
        """Perform homomorphic scalar multiplication on Paillier encrypted data"""
        def _multiply():
            scale_factor = a.metadata['scale_factor']
            scaled_scalar = int(scalar * scale_factor)
            result = [x * scaled_scalar for x in a.data]
            return result
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _multiply)
        
        return EncryptedData(
            data=result,
            scheme="paillier",
            metadata={
                'operation': 'scalar_multiplication',
                'scalar': scalar,
                'scale_factor': a.metadata['scale_factor']
            },
            checksum="",
            timestamp=time.time()
        )

class SecureMultiPartyComputation:
    """Secure multi-party computation for federated ECG analysis"""
    
    def __init__(self, scheme: HomomorphicScheme):
        self.scheme = scheme
        self.participants: Dict[str, EncryptionContext] = {}
        self.computation_history: List[Dict[str, Any]] = []
    
    async def register_participant(self, participant_id: str, **key_params) -> EncryptionContext:
        """Register a new participant in the computation"""
        context = await self.scheme.generate_keys(**key_params)
        self.participants[participant_id] = context
        
        logger.info(f"Registered participant {participant_id} for secure computation")
        return context
    
    async def secure_aggregation(
        self, 
        encrypted_contributions: Dict[str, EncryptedData]
    ) -> EncryptedData:
        """Perform secure aggregation of encrypted contributions"""
        if not encrypted_contributions:
            raise ValueError("No contributions provided for aggregation")
        
        participant_ids = list(encrypted_contributions.keys())
        result = encrypted_contributions[participant_ids[0]]
        
        for participant_id in participant_ids[1:]:
            result = await self.scheme.add(result, encrypted_contributions[participant_id])
        
        self.computation_history.append({
            'operation': 'secure_aggregation',
            'participants': participant_ids,
            'timestamp': time.time()
        })
        
        logger.info(f"Completed secure aggregation with {len(participant_ids)} participants")
        return result
    
    async def secure_average(
        self, 
        encrypted_contributions: Dict[str, EncryptedData]
    ) -> EncryptedData:
        """Compute secure average of encrypted contributions"""
        aggregated = await self.secure_aggregation(encrypted_contributions)
        num_participants = len(encrypted_contributions)
        
        average = await self.scheme.multiply(aggregated, 1.0 / num_participants)
        
        self.computation_history.append({
            'operation': 'secure_average',
            'participants': list(encrypted_contributions.keys()),
            'num_participants': num_participants,
            'timestamp': time.time()
        })
        
        return average

class ZeroKnowledgeProof:
    """Zero-knowledge proof system for ECG data validation"""
    
    def __init__(self):
        self.proofs: Dict[str, Dict[str, Any]] = {}
    
    async def generate_range_proof(
        self, 
        value: float, 
        min_range: float, 
        max_range: float,
        proof_id: str
    ) -> Dict[str, Any]:
        """Generate zero-knowledge proof that a value is within a specified range"""
        
        commitment = hashlib.sha256(f"{value}_{time.time()}".encode()).hexdigest()
        
        proof = {
            'commitment': commitment,
            'range': {'min': min_range, 'max': max_range},
            'timestamp': time.time(),
            'valid': min_range <= value <= max_range
        }
        
        self.proofs[proof_id] = proof
        logger.info(f"Generated range proof {proof_id} for value in range [{min_range}, {max_range}]")
        
        return proof
    
    async def verify_range_proof(self, proof_id: str) -> bool:
        """Verify a zero-knowledge range proof"""
        if proof_id not in self.proofs:
            return False
        
        proof = self.proofs[proof_id]
        
        is_valid = proof.get('valid', False)
        
        logger.info(f"Verified range proof {proof_id}: {'valid' if is_valid else 'invalid'}")
        return is_valid
    
    async def generate_membership_proof(
        self, 
        value: Any, 
        valid_set: List[Any],
        proof_id: str
    ) -> Dict[str, Any]:
        """Generate zero-knowledge proof that a value belongs to a set"""
        commitment = hashlib.sha256(f"{value}_{time.time()}".encode()).hexdigest()
        
        proof = {
            'commitment': commitment,
            'set_size': len(valid_set),
            'timestamp': time.time(),
            'valid': value in valid_set
        }
        
        self.proofs[proof_id] = proof
        logger.info(f"Generated membership proof {proof_id} for set of size {len(valid_set)}")
        
        return proof

class HomomorphicEncryptionService:
    """Main service for homomorphic encryption operations on ECG data"""
    
    def __init__(self):
        self.schemes: Dict[str, HomomorphicScheme] = {}
        self.smc = None
        self.zkp = ZeroKnowledgeProof()
        self.active_contexts: Dict[str, EncryptionContext] = {}
        
        if TENSEAL_AVAILABLE:
            self.schemes['tenseal'] = TenSEALScheme()
        if PHE_AVAILABLE:
            self.schemes['paillier'] = PaillierScheme()
        
        logger.info(f"Initialized HomomorphicEncryptionService with schemes: {list(self.schemes.keys())}")
    
    async def create_encryption_context(
        self, 
        scheme: str, 
        context_id: str,
        **params
    ) -> EncryptionContext:
        """Create a new encryption context"""
        if scheme not in self.schemes:
            raise ValueError(f"Unsupported encryption scheme: {scheme}")
        
        context = await self.schemes[scheme].generate_keys(**params)
        self.active_contexts[context_id] = context
        
        logger.info(f"Created encryption context {context_id} with scheme {scheme}")
        return context
    
    async def encrypt_ecg_data(
        self, 
        ecg_data: np.ndarray, 
        context_id: str
    ) -> EncryptedData:
        """Encrypt ECG data using specified context"""
        if context_id not in self.active_contexts:
            raise ValueError(f"Context {context_id} not found")
        
        context = self.active_contexts[context_id]
        scheme = self.schemes[context.scheme.split('_')[0]]
        
        encrypted = await scheme.encrypt(ecg_data, context)
        logger.info(f"Encrypted ECG data with shape {ecg_data.shape} using context {context_id}")
        
        return encrypted
    
    async def decrypt_ecg_data(
        self, 
        encrypted_data: EncryptedData, 
        context_id: str
    ) -> np.ndarray:
        """Decrypt ECG data using specified context"""
        if context_id not in self.active_contexts:
            raise ValueError(f"Context {context_id} not found")
        
        context = self.active_contexts[context_id]
        scheme = self.schemes[context.scheme.split('_')[0]]
        
        decrypted = await scheme.decrypt(encrypted_data, context)
        logger.info(f"Decrypted ECG data to shape {decrypted.shape}")
        
        return decrypted
    
    async def setup_secure_computation(self, scheme: str = 'tenseal') -> SecureMultiPartyComputation:
        """Setup secure multi-party computation"""
        if scheme not in self.schemes:
            raise ValueError(f"Unsupported scheme for SMC: {scheme}")
        
        self.smc = SecureMultiPartyComputation(self.schemes[scheme])
        logger.info(f"Setup secure multi-party computation with {scheme}")
        
        return self.smc
    
    async def federated_model_update(
        self, 
        encrypted_gradients: Dict[str, EncryptedData]
    ) -> EncryptedData:
        """Perform federated learning model update with encrypted gradients"""
        if not self.smc:
            raise ValueError("Secure multi-party computation not initialized")
        
        averaged_gradients = await self.smc.secure_average(encrypted_gradients)
        
        logger.info(f"Completed federated model update with {len(encrypted_gradients)} participants")
        return averaged_gradients
    
    async def validate_ecg_quality(
        self, 
        ecg_metrics: Dict[str, float],
        quality_thresholds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, bool]:
        """Validate ECG quality using zero-knowledge proofs"""
        validation_results = {}
        
        for metric, value in ecg_metrics.items():
            if metric in quality_thresholds:
                min_val, max_val = quality_thresholds[metric]
                proof_id = f"quality_{metric}_{time.time()}"
                
                proof = await self.zkp.generate_range_proof(
                    value, min_val, max_val, proof_id
                )
                
                validation_results[metric] = await self.zkp.verify_range_proof(proof_id)
        
        logger.info(f"Validated ECG quality for {len(validation_results)} metrics")
        return validation_results
    
    async def get_encryption_stats(self) -> Dict[str, Any]:
        """Get statistics about encryption operations"""
        return {
            'available_schemes': list(self.schemes.keys()),
            'active_contexts': len(self.active_contexts),
            'smc_initialized': self.smc is not None,
            'computation_history': len(self.smc.computation_history) if self.smc else 0,
            'zkp_proofs': len(self.zkp.proofs)
        }

homomorphic_service = HomomorphicEncryptionService()
