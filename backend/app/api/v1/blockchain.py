"""
API endpoints for blockchain operations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
import json
from pydantic import BaseModel, Field

from app.services.blockchain_service import blockchain_service
from app.core.security import get_current_user
from app.schemas.user import User

router = APIRouter()

class ECGAnalysisStoreRequest(BaseModel):
    """Request model for storing ECG analysis on blockchain"""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    patient_id: str = Field(..., description="Patient identifier")
    ecg_data: str = Field(..., description="Base64 encoded ECG data")
    analysis_results: Dict[str, Any] = Field(..., description="Analysis results")
    physician_id: str = Field(..., description="Physician identifier")
    facility_id: str = Field(..., description="Facility identifier")

class ECGAnalysisStoreResponse(BaseModel):
    """Response model for stored ECG analysis"""
    transaction_id: str
    analysis_id: str
    blockchain_hash: str
    timestamp: float

class PatientIdentityRequest(BaseModel):
    """Request model for creating patient identity"""
    patient_id: str = Field(..., description="Patient identifier")
    public_key: str = Field(..., description="Patient's public key")
    consent_records: List[Dict[str, Any]] = Field(default_factory=list, description="Consent records")
    access_permissions: Dict[str, List[str]] = Field(default_factory=dict, description="Access permissions")

class PatientIdentityResponse(BaseModel):
    """Response model for patient identity"""
    transaction_id: str
    patient_id: str
    identity_hash: str
    created_at: float

class ComplianceContractRequest(BaseModel):
    """Request model for deploying compliance contract"""
    contract_id: str = Field(..., description="Contract identifier")
    rules: Dict[str, Any] = Field(..., description="Compliance rules")
    regulatory_framework: str = Field(..., description="Regulatory framework (FDA, ANVISA, etc.)")

class ComplianceContractResponse(BaseModel):
    """Response model for compliance contract"""
    contract_address: str
    contract_id: str
    regulatory_framework: str
    deployed_at: float

class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking"""
    analysis_results: Dict[str, Any] = Field(..., description="Analysis results to check")
    contract_id: str = Field(..., description="Compliance contract ID")

class ComplianceCheckResponse(BaseModel):
    """Response model for compliance check"""
    contract_id: str
    is_compliant: bool
    violations: List[str]
    recommendations: List[str]
    checked_at: float

class AuditTrailResponse(BaseModel):
    """Response model for audit trail"""
    trail_id: str
    action: str
    actor_id: str
    resource_id: str
    timestamp: float
    details: Dict[str, Any]
    current_hash: str

@router.post("/store-analysis", response_model=ECGAnalysisStoreResponse)
async def store_ecg_analysis(
    request: ECGAnalysisStoreRequest,
    current_user: User = Depends(get_current_user)
):
    """Store ECG analysis on blockchain with immutable audit trail"""
    try:
        import base64
        ecg_data = base64.b64decode(request.ecg_data.encode())
        
        transaction_id = await blockchain_service.store_ecg_analysis(
            analysis_id=request.analysis_id,
            patient_id=request.patient_id,
            ecg_data=ecg_data,
            analysis_results=request.analysis_results,
            physician_id=request.physician_id,
            facility_id=request.facility_id
        )
        
        stored_record = await blockchain_service.retrieve_ecg_analysis(request.analysis_id)
        
        return ECGAnalysisStoreResponse(
            transaction_id=transaction_id,
            analysis_id=request.analysis_id,
            blockchain_hash=stored_record.data_integrity_hash if stored_record else "",
            timestamp=stored_record.timestamp if stored_record else 0.0
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store ECG analysis: {str(e)}")

@router.get("/retrieve-analysis/{analysis_id}")
async def retrieve_ecg_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user)
):
    """Retrieve ECG analysis from blockchain"""
    try:
        record = await blockchain_service.retrieve_ecg_analysis(analysis_id)
        
        if not record:
            raise HTTPException(status_code=404, detail="ECG analysis not found")
        
        return {
            "analysis_id": record.analysis_id,
            "patient_id": record.patient_id,
            "timestamp": record.timestamp,
            "ecg_hash": record.ecg_hash,
            "analysis_results": record.analysis_results,
            "physician_id": record.physician_id,
            "facility_id": record.facility_id,
            "compliance_flags": record.compliance_flags,
            "data_integrity_hash": record.data_integrity_hash,
            "version": record.version
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve ECG analysis: {str(e)}")

@router.post("/create-identity", response_model=PatientIdentityResponse)
async def create_patient_identity(
    request: PatientIdentityRequest,
    current_user: User = Depends(get_current_user)
):
    """Create decentralized patient identity"""
    try:
        transaction_id = await blockchain_service.create_patient_identity(
            patient_id=request.patient_id,
            public_key=request.public_key,
            consent_records=request.consent_records,
            access_permissions=request.access_permissions
        )
        
        identity = blockchain_service.patient_identities.get(request.patient_id)
        
        return PatientIdentityResponse(
            transaction_id=transaction_id,
            patient_id=request.patient_id,
            identity_hash=identity.identity_hash if identity else "",
            created_at=identity.created_at if identity else 0.0
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create patient identity: {str(e)}")

@router.post("/deploy-contract", response_model=ComplianceContractResponse)
async def deploy_compliance_contract(
    request: ComplianceContractRequest,
    current_user: User = Depends(get_current_user)
):
    """Deploy smart contract for automated compliance checking"""
    try:
        contract_address = await blockchain_service.deploy_compliance_contract(
            contract_id=request.contract_id,
            rules=request.rules,
            regulatory_framework=request.regulatory_framework
        )
        
        contract = blockchain_service.compliance_contracts.get(request.contract_id)
        
        return ComplianceContractResponse(
            contract_address=contract_address,
            contract_id=request.contract_id,
            regulatory_framework=request.regulatory_framework,
            deployed_at=contract.created_at if contract else 0.0
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deploy compliance contract: {str(e)}")

@router.post("/check-compliance", response_model=ComplianceCheckResponse)
async def check_compliance(
    request: ComplianceCheckRequest,
    current_user: User = Depends(get_current_user)
):
    """Check compliance using smart contract"""
    try:
        compliance_result = await blockchain_service.check_compliance(
            analysis_results=request.analysis_results,
            contract_id=request.contract_id
        )
        
        return ComplianceCheckResponse(
            contract_id=compliance_result["contract_id"],
            is_compliant=compliance_result["is_compliant"],
            violations=compliance_result.get("violations", []),
            recommendations=compliance_result.get("recommendations", []),
            checked_at=compliance_result["checked_at"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check compliance: {str(e)}")

@router.get("/audit-trail/{resource_id}", response_model=List[AuditTrailResponse])
async def get_audit_trail(
    resource_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get audit trail for a specific resource"""
    try:
        audit_trails = await blockchain_service.get_audit_trail(resource_id)
        
        return [
            AuditTrailResponse(
                trail_id=trail.trail_id,
                action=trail.action,
                actor_id=trail.actor_id,
                resource_id=trail.resource_id,
                timestamp=trail.timestamp,
                details=trail.details,
                current_hash=trail.current_hash
            )
            for trail in audit_trails
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get audit trail: {str(e)}")

@router.post("/verify-integrity/{analysis_id}")
async def verify_data_integrity(
    analysis_id: str,
    current_user: User = Depends(get_current_user)
):
    """Verify data integrity using blockchain"""
    try:
        is_valid = await blockchain_service.verify_data_integrity(analysis_id)
        
        return {
            "analysis_id": analysis_id,
            "is_valid": is_valid,
            "verified_at": __import__("time").time()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify data integrity: {str(e)}")

@router.get("/stats")
async def get_blockchain_stats(
    current_user: User = Depends(get_current_user)
):
    """Get blockchain service statistics"""
    try:
        stats = await blockchain_service.get_service_stats()
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get blockchain stats: {str(e)}")

@router.post("/initialize")
async def initialize_blockchain_service(
    config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Initialize blockchain service with configuration"""
    try:
        result = await blockchain_service.initialize(config)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to initialize blockchain service")
        
        return {
            "message": "Blockchain service initialized successfully",
            "network": blockchain_service.network.value,
            "provider_connected": blockchain_service.provider is not None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize blockchain service: {str(e)}")
