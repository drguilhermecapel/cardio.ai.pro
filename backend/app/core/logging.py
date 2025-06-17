# Adicione estes métodos à classe AuditLogger em backend/app/core/logging.py

class AuditLogger:
    # ... código existente ...
    
    def log_data_access(
        self,
        user_id: int,
        resource_type: str,
        resource_id: str,
        action: str,
        ip_address: str | None = None,
        additional_context: dict | None = None,
    ) -> None:
        """Log data access event for audit trail."""
        self._log_audit_event(
            event_type="DATA_ACCESS",
            user_id=user_id,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
                "ip_address": ip_address,
                "additional_context": additional_context or {},
            },
        )

    def log_medical_action(
        self,
        user_id: int,
        action_type: str,
        patient_id: int | None = None,
        analysis_id: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Log medical action for compliance."""
        self._log_audit_event(
            event_type="MEDICAL_ACTION",
            user_id=user_id,
            details={
                "action_type": action_type,
                "patient_id": patient_id,
                "analysis_id": analysis_id,
                "details": details or {},
            },
        )

    def _log_audit_event(
        self,
        event_type: str,
        user_id: int,
        details: dict,
    ) -> None:
        """Internal method to log audit events."""
        import json
        from datetime import datetime
        
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
        }
        
        # Log to structured logger
        self.logger.info(
            f"AUDIT: {event_type}",
            extra={"audit_data": json.dumps(audit_entry)},
        )
