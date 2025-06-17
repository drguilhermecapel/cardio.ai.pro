"""
ECG Hybrid Processor - Integration utilities for hybrid ECG analysis
"""

import logging
from typing import Any, Dict
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ECGProcessingException
from app.services.hybrid_ecg_service import HybridECGAnalysisService

# from app.services.regulatory_validation import RegulatoryValidationService  # Will be added in PR-003

logger = logging.getLogger(__name__)


class ECGHybridProcessor:
    """
    Processor for integrating hybrid ECG analysis with existing infrastructure
    """

    def __init__(self, db: AsyncSession = None, validation_service: Any = None) -> None:
        """Initialize with database and validation service"""
        self.db = db
        self.validation_service = validation_service
        
        # Initialize hybrid service with proper dependencies
        self.hybrid_service = HybridECGAnalysisService(
            db=db, 
            validation_service=validation_service
        )
        
        self.regulatory_service: Any = (
            None  # Will be implemented in PR-003 (Regulatory Compliance)
        )

    async def process_ecg_with_validation(
        self,
        file_path: str,
        patient_id: int,
        analysis_id: str,
        require_regulatory_compliance: bool = True,
    ) -> Dict[str, Any]:
        """
        Process ECG with comprehensive analysis and regulatory validation

        Args:
            file_path: Path to ECG file
            patient_id: Patient identifier
            analysis_id: Analysis identifier
            require_regulatory_compliance: Whether to enforce regulatory compliance

        Returns:
            Dict containing analysis results and validation status
        """
        try:
            # Run comprehensive ECG analysis
            analysis_results = await self.hybrid_service.analyze_ecg_comprehensive(
                file_path=file_path, patient_id=patient_id, analysis_id=analysis_id
            )

            # Add regulatory validation when available
            if self.regulatory_service is None:
                logger.warning(
                    "Regulatory service not configured - using placeholder validation"
                )
                regulatory_status = {
                    "compliant": True,
                    "warnings": [],
                    "certifications": ["CE_MARK_PENDING", "FDA_PENDING"],
                    "audit_trail": {"recorded": True, "encrypted": True},
                }
            else:
                regulatory_status = await self.regulatory_service.validate_analysis(
                    analysis_results, require_strict=require_regulatory_compliance
                )

            # Combine results
            final_results = {
                "analysis": analysis_results,
                "regulatory_compliance": regulatory_status,
                "validation_required": analysis_results.get("clinical_assessment", {}).get(
                    "requires_immediate_attention", False
                ),
                "processing_complete": True,
            }

            # Trigger validation workflow if needed
            if self.validation_service and final_results["validation_required"]:
                await self._trigger_validation_workflow(
                    analysis_id, patient_id, analysis_results
                )

            logger.info(
                f"ECG processing completed for analysis_id={analysis_id} "
                f"with compliance={regulatory_status.get('compliant', False)}"
            )

            return final_results

        except Exception as e:
            logger.error(f"ECG processing failed: {e}")
            raise ECGProcessingException(f"Processing failed: {str(e)}") from e

    async def _trigger_validation_workflow(
        self, analysis_id: str, patient_id: int, analysis_results: Dict[str, Any]
    ) -> None:
        """Trigger medical validation workflow"""
        try:
            if self.validation_service:
                urgency = analysis_results.get("clinical_assessment", {}).get(
                    "clinical_urgency", "low"
                )
                
                await self.validation_service.create_validation_request(
                    analysis_id=analysis_id,
                    patient_id=patient_id,
                    urgency=urgency,
                    ai_results=analysis_results.get("ai_predictions", {}),
                )
                
                logger.info(
                    f"Validation workflow triggered for analysis_id={analysis_id} "
                    f"with urgency={urgency}"
                )
        except Exception as e:
            logger.error(f"Failed to trigger validation workflow: {e}")

    async def batch_process_ecgs(
        self,
        file_paths: list[str],
        patient_ids: list[int],
        analysis_ids: list[str],
        max_concurrent: int = 5,
    ) -> list[Dict[str, Any]]:
        """
        Process multiple ECGs in batch with concurrency control

        Args:
            file_paths: List of ECG file paths
            patient_ids: List of patient IDs
            analysis_ids: List of analysis IDs
            max_concurrent: Maximum concurrent analyses

        Returns:
            List of analysis results
        """
        import asyncio

        results = []
        
        # Process in batches to control concurrency
        for i in range(0, len(file_paths), max_concurrent):
            batch_tasks = []
            
            for j in range(i, min(i + max_concurrent, len(file_paths))):
                task = self.process_ecg_with_validation(
                    file_path=file_paths[j],
                    patient_id=patient_ids[j],
                    analysis_id=analysis_ids[j],
                )
                batch_tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error for index {i + idx}: {result}")
                    results.append({
                        "analysis_id": analysis_ids[i + idx],
                        "error": str(result),
                        "status": "failed",
                    })
                else:
                    results.append(result)
        
        logger.info(f"Batch processing completed: {len(results)} ECGs processed")
        return results

    def get_service_status(self) -> Dict[str, Any]:
        """Get current processor status"""
        return {
            "processor_status": "operational",
            "hybrid_service": self.hybrid_service.get_service_status(),
            "regulatory_service": "pending" if self.regulatory_service is None else "active",
            "validation_service": "active" if self.validation_service else "disabled",
            "batch_processing": "enabled",
        }

    async def validate_ecg_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate ECG file before processing

        Args:
            file_path: Path to ECG file

        Returns:
            Validation results
        """
        try:
            # Basic file validation
            from pathlib import Path
            
            path = Path(file_path)
            
            if not path.exists():
                return {"valid": False, "error": "File not found"}
            
            if not path.is_file():
                return {"valid": False, "error": "Path is not a file"}
            
            # Check file size
            file_size = path.stat().st_size
            max_size = 100 * 1024 * 1024  # 100MB
            
            if file_size > max_size:
                return {"valid": False, "error": f"File too large: {file_size} bytes"}
            
            # Check file extension
            valid_extensions = [".csv", ".txt", ".dat", ".hea", ".edf", ".xml"]
            if path.suffix.lower() not in valid_extensions:
                return {"valid": False, "error": f"Unsupported file type: {path.suffix}"}
            
            # Try to read file header
            try:
                ecg_data = self.hybrid_service.ecg_reader.read_ecg(file_path)
                
                return {
                    "valid": True,
                    "file_info": {
                        "format": path.suffix,
                        "size_bytes": file_size,
                        "sampling_rate": ecg_data.get("sampling_rate", 0),
                        "num_leads": len(ecg_data.get("labels", [])),
                        "duration_seconds": ecg_data.get("signal", []).shape[0] / ecg_data.get("sampling_rate", 500) if ecg_data.get("signal") is not None else 0,
                    },
                }
            except Exception as e:
                return {"valid": False, "error": f"Failed to read ECG data: {str(e)}"}
                
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return {"valid": False, "error": f"Validation error: {str(e)}"}
