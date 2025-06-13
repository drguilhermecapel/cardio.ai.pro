"""
Intelligent Alert System for ECG Analysis
Implements smart alerting with priority-based notifications and contextual information
Based on Phase 2 optimization specifications for CardioAI Pro
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = "critical"      # Life-threatening conditions
    HIGH = "high"             # Urgent medical attention needed
    MEDIUM = "medium"         # Important findings
    LOW = "low"              # Minor abnormalities
    INFO = "info"            # Informational alerts


class AlertCategory(Enum):
    """Categories of ECG alerts"""
    ARRHYTHMIA = "arrhythmia"
    ISCHEMIA = "ischemia"
    CONDUCTION = "conduction"
    MORPHOLOGY = "morphology"
    QUALITY = "quality"
    TECHNICAL = "technical"


@dataclass
class AlertRule:
    """Configuration for alert generation rules"""
    condition_name: str
    priority: AlertPriority
    category: AlertCategory
    threshold: float
    message_template: str
    clinical_context: str
    recommended_actions: List[str] = field(default_factory=list)
    suppress_duration_minutes: int = 5  # Suppress duplicate alerts
    requires_confirmation: bool = False


@dataclass
class ECGAlert:
    """Individual ECG alert with contextual information"""
    alert_id: str
    timestamp: datetime
    priority: AlertPriority
    category: AlertCategory
    condition_name: str
    confidence_score: float
    message: str
    clinical_context: str
    recommended_actions: List[str]
    patient_context: Optional[Dict[str, Any]] = None
    ecg_segment: Optional[npt.NDArray[np.float64]] = None
    lead_information: Optional[Dict[str, Any]] = None
    suppressed: bool = False
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class IntelligentAlertSystem:
    """
    Intelligent alert system for ECG analysis with contextual prioritization
    """
    
    def __init__(self):
        self.alert_rules = self._initialize_alert_rules()
        self.active_alerts: List[ECGAlert] = []
        self.alert_history: List[ECGAlert] = []
        self.suppressed_alerts: Dict[str, datetime] = {}
        
    def _initialize_alert_rules(self) -> Dict[str, AlertRule]:
        """Initialize comprehensive alert rules for ECG conditions"""
        rules = {
            "ventricular_fibrillation": AlertRule(
                condition_name="Ventricular Fibrillation",
                priority=AlertPriority.CRITICAL,
                category=AlertCategory.ARRHYTHMIA,
                threshold=0.85,
                message_template="CRITICAL: Ventricular fibrillation detected (confidence: {confidence:.1%})",
                clinical_context="Life-threatening arrhythmia requiring immediate defibrillation",
                recommended_actions=[
                    "Initiate CPR immediately",
                    "Prepare for defibrillation",
                    "Call emergency response team",
                    "Administer epinephrine as per protocol"
                ],
                suppress_duration_minutes=1,
                requires_confirmation=True
            ),
            
            "ventricular_tachycardia": AlertRule(
                condition_name="Ventricular Tachycardia",
                priority=AlertPriority.CRITICAL,
                category=AlertCategory.ARRHYTHMIA,
                threshold=0.80,
                message_template="CRITICAL: Ventricular tachycardia detected (confidence: {confidence:.1%})",
                clinical_context="Potentially life-threatening arrhythmia requiring immediate intervention",
                recommended_actions=[
                    "Assess patient consciousness and pulse",
                    "Prepare for cardioversion if unstable",
                    "Consider antiarrhythmic medications",
                    "Continuous cardiac monitoring"
                ],
                suppress_duration_minutes=2
            ),
            
            "stemi": AlertRule(
                condition_name="ST-Elevation Myocardial Infarction",
                priority=AlertPriority.CRITICAL,
                category=AlertCategory.ISCHEMIA,
                threshold=0.75,
                message_template="CRITICAL: STEMI detected in {leads} (confidence: {confidence:.1%})",
                clinical_context="Acute myocardial infarction with complete coronary occlusion",
                recommended_actions=[
                    "Activate cardiac catheterization lab",
                    "Administer dual antiplatelet therapy",
                    "Consider thrombolytic therapy if PCI unavailable",
                    "Serial cardiac enzymes and ECGs"
                ],
                suppress_duration_minutes=10
            ),
            
            "nstemi": AlertRule(
                condition_name="Non-ST-Elevation Myocardial Infarction",
                priority=AlertPriority.HIGH,
                category=AlertCategory.ISCHEMIA,
                threshold=0.70,
                message_template="HIGH: NSTEMI suspected (confidence: {confidence:.1%})",
                clinical_context="Acute coronary syndrome without complete vessel occlusion",
                recommended_actions=[
                    "Serial cardiac biomarkers",
                    "Risk stratification assessment",
                    "Consider early invasive strategy",
                    "Anticoagulation therapy"
                ],
                suppress_duration_minutes=15
            ),
            
            "complete_heart_block": AlertRule(
                condition_name="Complete Heart Block",
                priority=AlertPriority.HIGH,
                category=AlertCategory.CONDUCTION,
                threshold=0.80,
                message_template="HIGH: Complete AV block detected (confidence: {confidence:.1%})",
                clinical_context="Complete dissociation between atrial and ventricular activity",
                recommended_actions=[
                    "Assess hemodynamic stability",
                    "Prepare for temporary pacing",
                    "Consider permanent pacemaker evaluation",
                    "Monitor for escape rhythms"
                ],
                suppress_duration_minutes=5
            ),
            
            "poor_signal_quality": AlertRule(
                condition_name="Poor Signal Quality",
                priority=AlertPriority.MEDIUM,
                category=AlertCategory.QUALITY,
                threshold=0.3,  # Below 30% quality
                message_template="MEDIUM: Poor ECG signal quality detected (quality: {quality:.1%})",
                clinical_context="Signal quality may affect diagnostic accuracy",
                recommended_actions=[
                    "Check electrode placement and contact",
                    "Reduce patient movement artifacts",
                    "Verify equipment connections",
                    "Consider lead replacement if needed"
                ],
                suppress_duration_minutes=3
            ),
            
            "electrode_disconnection": AlertRule(
                condition_name="Electrode Disconnection",
                priority=AlertPriority.HIGH,
                category=AlertCategory.TECHNICAL,
                threshold=0.90,
                message_template="HIGH: Electrode disconnection detected in {leads}",
                clinical_context="Loss of ECG signal in one or more leads",
                recommended_actions=[
                    "Check electrode connections immediately",
                    "Verify lead placement",
                    "Replace electrodes if necessary",
                    "Resume monitoring once resolved"
                ],
                suppress_duration_minutes=1
            )
        }
        
        return rules
    
    def process_ecg_analysis(self, analysis_results: Dict[str, Any], 
                           patient_context: Optional[Dict[str, Any]] = None) -> List[ECGAlert]:
        """
        Process ECG analysis results and generate intelligent alerts
        
        Args:
            analysis_results: Results from ECG analysis including predictions and quality metrics
            patient_context: Optional patient information for contextual alerting
            
        Returns:
            List of generated alerts
        """
        generated_alerts = []
        current_time = datetime.now()
        
        try:
            ai_predictions = analysis_results.get('ai_results', {})
            pathology_results = analysis_results.get('pathology_results', {})
            quality_metrics = analysis_results.get('quality_metrics', {})
            signal_data = analysis_results.get('preprocessed_signal')
            
            if pathology_results:
                for condition, details in pathology_results.items():
                    if isinstance(details, dict) and 'confidence' in details:
                        confidence = details['confidence']
                        alert = self._evaluate_condition_alert(
                            condition, confidence, current_time, 
                            patient_context, signal_data, details
                        )
                        if alert:
                            generated_alerts.append(alert)
            
            if ai_predictions and 'predictions' in ai_predictions:
                predictions = ai_predictions['predictions']
                for condition, confidence in predictions.items():
                    if isinstance(confidence, (int, float)):
                        alert = self._evaluate_condition_alert(
                            condition, confidence, current_time,
                            patient_context, signal_data
                        )
                        if alert:
                            generated_alerts.append(alert)
            
            if quality_metrics:
                quality_score = quality_metrics.get('quality_score', 1.0)
                if quality_score < 0.5:  # Poor quality threshold
                    alert = self._create_quality_alert(
                        quality_score, current_time, quality_metrics
                    )
                    if alert:
                        generated_alerts.append(alert)
            
            filtered_alerts = self._apply_intelligent_filtering(generated_alerts)
            
            self.active_alerts.extend(filtered_alerts)
            self.alert_history.extend(filtered_alerts)
            
            if filtered_alerts:
                logger.info(f"Generated {len(filtered_alerts)} intelligent alerts")
                for alert in filtered_alerts:
                    logger.info(f"Alert: {alert.priority.value.upper()} - {alert.condition_name}")
            
            return filtered_alerts
            
        except Exception as e:
            logger.error(f"Error processing ECG analysis for alerts: {e}")
            return []
    
    def _evaluate_condition_alert(self, condition: str, confidence: float,
                                timestamp: datetime, patient_context: Optional[Dict[str, Any]],
                                signal_data: Optional[npt.NDArray[np.float64]],
                                details: Optional[Dict[str, Any]] = None) -> Optional[ECGAlert]:
        """
        Evaluate whether a condition warrants an alert
        
        Args:
            condition: Name of the detected condition
            confidence: Confidence score for the detection
            timestamp: Current timestamp
            patient_context: Patient information
            signal_data: ECG signal data
            details: Additional condition details
            
        Returns:
            ECGAlert if conditions are met, None otherwise
        """
        condition_key = condition.lower().replace(' ', '_').replace('-', '_')
        
        if condition_key not in self.alert_rules:
            if confidence > 0.8:
                return self._create_generic_alert(condition, confidence, timestamp)
            return None
        
        rule = self.alert_rules[condition_key]
        
        if confidence < rule.threshold:
            return None
        
        if self._is_alert_suppressed(condition_key, timestamp):
            return None
        
        alert_id = f"{condition_key}_{int(timestamp.timestamp())}"
        
        message_kwargs = {'confidence': confidence}
        if details and 'leads' in details:
            message_kwargs['leads'] = ', '.join(details['leads'])
        if 'quality' in (details or {}):
            message_kwargs['quality'] = details['quality']
        
        try:
            formatted_message = rule.message_template.format(**message_kwargs)
        except KeyError:
            formatted_message = rule.message_template.format(confidence=confidence)
        
        alert = ECGAlert(
            alert_id=alert_id,
            timestamp=timestamp,
            priority=rule.priority,
            category=rule.category,
            condition_name=rule.condition_name,
            confidence_score=confidence,
            message=formatted_message,
            clinical_context=rule.clinical_context,
            recommended_actions=rule.recommended_actions.copy(),
            patient_context=patient_context,
            ecg_segment=signal_data,
            lead_information=details
        )
        
        self.suppressed_alerts[condition_key] = timestamp + timedelta(
            minutes=rule.suppress_duration_minutes
        )
        
        return alert
    
    def _create_quality_alert(self, quality_score: float, timestamp: datetime,
                            quality_metrics: Dict[str, Any]) -> Optional[ECGAlert]:
        """Create alert for poor signal quality"""
        rule = self.alert_rules["poor_signal_quality"]
        
        if quality_score > rule.threshold:  # Quality is acceptable
            return None
        
        if self._is_alert_suppressed("poor_signal_quality", timestamp):
            return None
        
        alert_id = f"poor_quality_{int(timestamp.timestamp())}"
        message = rule.message_template.format(quality=quality_score)
        
        alert = ECGAlert(
            alert_id=alert_id,
            timestamp=timestamp,
            priority=rule.priority,
            category=rule.category,
            condition_name=rule.condition_name,
            confidence_score=1.0 - quality_score,  # Invert for confidence in poor quality
            message=message,
            clinical_context=rule.clinical_context,
            recommended_actions=rule.recommended_actions.copy(),
            lead_information=quality_metrics
        )
        
        self.suppressed_alerts["poor_signal_quality"] = timestamp + timedelta(
            minutes=rule.suppress_duration_minutes
        )
        
        return alert
    
    def _create_generic_alert(self, condition: str, confidence: float,
                            timestamp: datetime) -> ECGAlert:
        """Create a generic alert for unknown conditions"""
        alert_id = f"generic_{condition.lower()}_{int(timestamp.timestamp())}"
        
        return ECGAlert(
            alert_id=alert_id,
            timestamp=timestamp,
            priority=AlertPriority.MEDIUM,
            category=AlertCategory.MORPHOLOGY,
            condition_name=condition,
            confidence_score=confidence,
            message=f"MEDIUM: {condition} detected (confidence: {confidence:.1%})",
            clinical_context="Abnormal ECG finding detected by AI analysis",
            recommended_actions=[
                "Review ECG manually",
                "Consider clinical correlation",
                "Consult cardiology if indicated"
            ]
        )
    
    def _is_alert_suppressed(self, condition_key: str, current_time: datetime) -> bool:
        """Check if an alert type is currently suppressed"""
        if condition_key not in self.suppressed_alerts:
            return False
        
        suppression_end = self.suppressed_alerts[condition_key]
        return current_time < suppression_end
    
    def _apply_intelligent_filtering(self, alerts: List[ECGAlert]) -> List[ECGAlert]:
        """
        Apply intelligent filtering to reduce alert fatigue
        
        Args:
            alerts: List of generated alerts
            
        Returns:
            Filtered list of alerts
        """
        if not alerts:
            return alerts
        
        priority_order = {
            AlertPriority.CRITICAL: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3,
            AlertPriority.INFO: 4
        }
        
        sorted_alerts = sorted(alerts, key=lambda x: priority_order[x.priority])
        
        filtered_alerts = []
        
        for alert in sorted_alerts:
            if alert.priority == AlertPriority.CRITICAL:
                filtered_alerts.append(alert)
                continue
            
            if self._is_duplicate_alert(alert, filtered_alerts):
                continue
            
            if self._passes_confidence_filter(alert):
                filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def _is_duplicate_alert(self, alert: ECGAlert, existing_alerts: List[ECGAlert]) -> bool:
        """Check if alert is a duplicate of existing alerts"""
        for existing in existing_alerts:
            if (existing.condition_name == alert.condition_name and
                existing.category == alert.category):
                return True
        return False
    
    def _passes_confidence_filter(self, alert: ECGAlert) -> bool:
        """Apply confidence-based filtering"""
        confidence_thresholds = {
            AlertPriority.HIGH: 0.7,
            AlertPriority.MEDIUM: 0.75,
            AlertPriority.LOW: 0.8,
            AlertPriority.INFO: 0.85
        }
        
        threshold = confidence_thresholds.get(alert.priority, 0.5)
        return alert.confidence_score >= threshold
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert
        
        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: User who acknowledged the alert
            
        Returns:
            True if alert was found and acknowledged
        """
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        
        return False
    
    def get_active_alerts(self, priority_filter: Optional[AlertPriority] = None) -> List[ECGAlert]:
        """
        Get currently active alerts
        
        Args:
            priority_filter: Optional filter by priority level
            
        Returns:
            List of active alerts
        """
        active = [alert for alert in self.active_alerts if not alert.acknowledged]
        
        if priority_filter:
            active = [alert for alert in active if alert.priority == priority_filter]
        
        return active
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of alert system status
        
        Returns:
            Dictionary with alert statistics and status
        """
        active_alerts = self.get_active_alerts()
        
        priority_counts = {}
        for priority in AlertPriority:
            priority_counts[priority.value] = len([
                alert for alert in active_alerts if alert.priority == priority
            ])
        
        return {
            'total_active_alerts': len(active_alerts),
            'priority_breakdown': priority_counts,
            'total_alerts_generated': len(self.alert_history),
            'suppressed_conditions': len(self.suppressed_alerts),
            'last_alert_time': max([alert.timestamp for alert in self.alert_history]) if self.alert_history else None
        }
    
    def clear_old_alerts(self, hours_old: int = 24) -> int:
        """
        Clear old alerts from active list
        
        Args:
            hours_old: Age threshold in hours
            
        Returns:
            Number of alerts cleared
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_old)
        
        old_alerts = [alert for alert in self.active_alerts if alert.timestamp < cutoff_time]
        self.active_alerts = [alert for alert in self.active_alerts if alert.timestamp >= cutoff_time]
        
        logger.info(f"Cleared {len(old_alerts)} old alerts")
        return len(old_alerts)


def create_intelligent_alert_system() -> IntelligentAlertSystem:
    """
    Factory function to create intelligent alert system
    
    Returns:
        Configured intelligent alert system
    """
    return IntelligentAlertSystem()


if __name__ == "__main__":
    alert_system = create_intelligent_alert_system()
    
    test_analysis = {
        'ai_results': {
            'predictions': {
                'ventricular_fibrillation': 0.92,
                'normal_sinus_rhythm': 0.05
            }
        },
        'pathology_results': {
            'stemi': {
                'confidence': 0.85,
                'leads': ['V1', 'V2', 'V3']
            }
        },
        'quality_metrics': {
            'quality_score': 0.25,
            'meets_quality_threshold': False
        }
    }
    
    alerts = alert_system.process_ecg_analysis(test_analysis)
    
    print(f"Generated {len(alerts)} alerts:")
    for alert in alerts:
        print(f"- {alert.priority.value.upper()}: {alert.message}")
        print(f"  Clinical context: {alert.clinical_context}")
        print(f"  Recommended actions: {len(alert.recommended_actions)} actions")
        print()
    
    summary = alert_system.get_alert_summary()
    print("Alert System Summary:")
    print(f"- Total active alerts: {summary['total_active_alerts']}")
    print(f"- Priority breakdown: {summary['priority_breakdown']}")
