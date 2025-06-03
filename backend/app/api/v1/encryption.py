"""
API endpoints for homomorphic encryption operations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
import numpy as np
from pydantic import BaseModel, Field

from app.services.homomorphic_encryption import homomorphic_service
from app.core.security import get_current_user
from app.schemas.user import User

router = APIRouter()

class EncryptionContextRequest(BaseModel):
    """Request model for creating encryption context"""
    scheme: str = Field(..., description="Encryption scheme (tenseal, paillier)")
    context_id: str = Field(..., description="Unique identifier for the context")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Scheme-specific parameters")

class EncryptionContextResponse(BaseModel):
    """Response model for encryption context"""
    context_id: str
    scheme: str
    created_at: float
    parameters: Dict[str, Any]

class ECGEncryptionRequest(BaseModel):
    """Request model for ECG data encryption"""
    ecg_data: List[List[float]] = Field(..., description="ECG data as nested list")
    context_id: str = Field(..., description="Encryption context ID")

class ECGEncryptionResponse(BaseModel):
    """Response model for encrypted ECG data"""
    encrypted_data_id: str
    scheme: str
    metadata: Dict[str, Any]
    timestamp: float

class ECGQualityValidationRequest(BaseModel):
    """Request model for ECG quality validation"""
    ecg_metrics: Dict[str, float] = Field(..., description="ECG quality metrics")
    quality_thresholds: Dict[str, List[float]] = Field(..., description="Quality thresholds as [min, max]")

class ECGQualityValidationResponse(BaseModel):
    """Response model for ECG quality validation"""
    validation_results: Dict[str, bool]
    proof_ids: List[str]

class FederatedLearningRequest(BaseModel):
    """Request model for federated learning"""
    participant_gradients: Dict[str, str] = Field(..., description="Encrypted gradient data IDs by participant")

class FederatedLearningResponse(BaseModel):
    """Response model for federated learning"""
    averaged_gradients_id: str
    participants: List[str]
    computation_history_count: int

encrypted_data_store: Dict[str, Any] = {}

@router.post("/context", response_model=EncryptionContextResponse)
async def create_encryption_context(
    request: EncryptionContextRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new homomorphic encryption context"""
    try:
        context = await homomorphic_service.create_encryption_context(
            scheme=request.scheme,
            context_id=request.context_id,
            **request.parameters
        )
        
        return EncryptionContextResponse(
            context_id=request.context_id,
            scheme=context.scheme,
            created_at=context.created_at,
            parameters=context.parameters or {}
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create encryption context: {str(e)}")

@router.post("/encrypt", response_model=ECGEncryptionResponse)
async def encrypt_ecg_data(
    request: ECGEncryptionRequest,
    current_user: User = Depends(get_current_user)
):
    """Encrypt ECG data using homomorphic encryption"""
    try:
        ecg_array = np.array(request.ecg_data, dtype=np.float64)
        
        encrypted_data = await homomorphic_service.encrypt_ecg_data(
            ecg_data=ecg_array,
            context_id=request.context_id
        )
        
        encrypted_data_id = f"encrypted_{request.context_id}_{encrypted_data.timestamp}"
        encrypted_data_store[encrypted_data_id] = encrypted_data
        
        return ECGEncryptionResponse(
            encrypted_data_id=encrypted_data_id,
            scheme=encrypted_data.scheme,
            metadata=encrypted_data.metadata,
            timestamp=encrypted_data.timestamp
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to encrypt ECG data: {str(e)}")

@router.post("/decrypt/{encrypted_data_id}")
async def decrypt_ecg_data(
    encrypted_data_id: str,
    context_id: str,
    current_user: User = Depends(get_current_user)
):
    """Decrypt ECG data using homomorphic encryption"""
    try:
        if encrypted_data_id not in encrypted_data_store:
            raise HTTPException(status_code=404, detail="Encrypted data not found")
        
        encrypted_data = encrypted_data_store[encrypted_data_id]
        
        decrypted_data = await homomorphic_service.decrypt_ecg_data(
            encrypted_data=encrypted_data,
            context_id=context_id
        )
        
        return {
            "decrypted_data": decrypted_data.tolist(),
            "original_shape": encrypted_data.metadata.get("original_shape"),
            "context_id": context_id
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decrypt ECG data: {str(e)}")

@router.post("/validate-quality", response_model=ECGQualityValidationResponse)
async def validate_ecg_quality(
    request: ECGQualityValidationRequest,
    current_user: User = Depends(get_current_user)
):
    """Validate ECG quality using zero-knowledge proofs"""
    try:
        quality_thresholds = {
            metric: tuple(thresholds) 
            for metric, thresholds in request.quality_thresholds.items()
        }
        
        validation_results = await homomorphic_service.validate_ecg_quality(
            ecg_metrics=request.ecg_metrics,
            quality_thresholds=quality_thresholds
        )
        
        proof_ids = list(homomorphic_service.zkp.proofs.keys())
        
        return ECGQualityValidationResponse(
            validation_results=validation_results,
            proof_ids=proof_ids[-len(validation_results):]  # Get recent proof IDs
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate ECG quality: {str(e)}")

@router.post("/setup-smc")
async def setup_secure_computation(
    scheme: str = "tenseal",
    current_user: User = Depends(get_current_user)
):
    """Setup secure multi-party computation"""
    try:
        smc = await homomorphic_service.setup_secure_computation(scheme)
        
        return {
            "message": "Secure multi-party computation initialized",
            "scheme": scheme,
            "participants": len(smc.participants)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to setup SMC: {str(e)}")

@router.post("/federated-update", response_model=FederatedLearningResponse)
async def federated_model_update(
    request: FederatedLearningRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Perform federated learning model update with encrypted gradients"""
    try:
        encrypted_gradients = {}
        for participant_id, gradient_id in request.participant_gradients.items():
            if gradient_id not in encrypted_data_store:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Encrypted gradient data not found for participant {participant_id}"
                )
            encrypted_gradients[participant_id] = encrypted_data_store[gradient_id]
        
        averaged_gradients = await homomorphic_service.federated_model_update(encrypted_gradients)
        
        averaged_gradients_id = f"averaged_gradients_{averaged_gradients.timestamp}"
        encrypted_data_store[averaged_gradients_id] = averaged_gradients
        
        return FederatedLearningResponse(
            averaged_gradients_id=averaged_gradients_id,
            participants=list(request.participant_gradients.keys()),
            computation_history_count=len(homomorphic_service.smc.computation_history)
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform federated update: {str(e)}")

@router.get("/stats")
async def get_encryption_stats(
    current_user: User = Depends(get_current_user)
):
    """Get homomorphic encryption service statistics"""
    try:
        stats = await homomorphic_service.get_encryption_stats()
        
        stats.update({
            "encrypted_data_count": len(encrypted_data_store),
            "service_status": "active"
        })
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get encryption stats: {str(e)}")

@router.delete("/context/{context_id}")
async def delete_encryption_context(
    context_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete an encryption context"""
    try:
        if context_id in homomorphic_service.active_contexts:
            del homomorphic_service.active_contexts[context_id]
            
            keys_to_remove = [
                key for key in encrypted_data_store.keys() 
                if context_id in key
            ]
            for key in keys_to_remove:
                del encrypted_data_store[key]
            
            return {
                "message": f"Encryption context {context_id} deleted successfully",
                "cleaned_data_count": len(keys_to_remove)
            }
        else:
            raise HTTPException(status_code=404, detail="Encryption context not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete encryption context: {str(e)}")
