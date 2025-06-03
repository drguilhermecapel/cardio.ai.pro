"""
Regulatory Validation Service for ECG Analysis
Implements validation standards for FDA, ANVISA, NMSA (China), and European Union
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RegulatoryStandard(Enum):
    """Regulatory standards enumeration"""
    FDA = "FDA"
    ANVISA = "ANVISA"
    NMSA = "NMSA"
    EU_MDR = "EU_MDR"


class ValidationResult(BaseModel):
    """Validation result model"""
    standard: RegulatoryStandard
    compliant: bool
    confidence_score: float
    validation_errors: list[str]
    validation_warnings: list[str]
    test_results: dict[str, Any]
    timestamp: datetime


class RegulatoryValidationService:
    """
    Comprehensive regulatory validation service
    Implements standards from FDA, ANVISA, NMSA, and EU MDR
    """

    def __init__(self) -> None:
        self.validation_thresholds = {
            RegulatoryStandard.FDA: {
                'min_confidence': 0.85,
                'min_sensitivity': 0.90,
                'min_specificity': 0.85,
                'max_false_positive_rate': 0.15,
                'min_signal_quality': 0.7,
                'critical_condition_confidence': 0.95
            },
            RegulatoryStandard.ANVISA: {
                'min_confidence': 0.80,
                'min_sensitivity': 0.88,
                'min_specificity': 0.82,
                'max_false_positive_rate': 0.18,
                'min_signal_quality': 0.65,
                'critical_condition_confidence': 0.90
            },
            RegulatoryStandard.NMSA: {
                'min_confidence': 0.82,
                'min_sensitivity': 0.89,
                'min_specificity': 0.83,
                'max_false_positive_rate': 0.17,
                'min_signal_quality': 0.68,
                'critical_condition_confidence': 0.92
            },
            RegulatoryStandard.EU_MDR: {
                'min_confidence': 0.87,
                'min_sensitivity': 0.91,
                'min_specificity': 0.86,
                'max_false_positive_rate': 0.14,
                'min_signal_quality': 0.72,
                'critical_condition_confidence': 0.96
            }
        }

    async def validate_analysis_comprehensive(
        self,
        analysis_results: dict[str, Any],
        ground_truth: dict[str, Any] | None = None
    ) -> dict[RegulatoryStandard, ValidationResult]:
        """
        Comprehensive validation against all regulatory standards

        Args:
            analysis_results: ECG analysis results
            ground_truth: Optional ground truth for performance validation

        Returns:
            Dict mapping each standard to its validation result
        """
        validation_results = {}

        for standard in RegulatoryStandard:
            result = await self._validate_single_standard(
                standard, analysis_results, ground_truth
            )
            validation_results[standard] = result

        return validation_results

    async def _validate_single_standard(
        self,
        standard: RegulatoryStandard,
        analysis_results: dict[str, Any],
        ground_truth: dict[str, Any] | None = None
    ) -> ValidationResult:
        """Validate against a single regulatory standard"""

        thresholds = self.validation_thresholds[standard]
        errors = []
        warnings: list[str] = []
        test_results = {}

        confidence = analysis_results.get('ai_predictions', {}).get('confidence', 0.0)
        test_results['confidence'] = confidence

        if confidence < thresholds['min_confidence']:
            if confidence < (thresholds['min_confidence'] - 0.05):
                errors.append(f"AI confidence {confidence:.3f} below minimum {thresholds['min_confidence']}")

        signal_quality = analysis_results.get('signal_quality', {}).get('overall_score', 0.0)
        test_results['signal_quality'] = signal_quality

        if signal_quality < thresholds['min_signal_quality']:
            if signal_quality < (thresholds['min_signal_quality'] - 0.1):
                errors.append(f"Signal quality {signal_quality:.3f} below minimum {thresholds['min_signal_quality']}")

        clinical_assessment = analysis_results.get('clinical_assessment', {})
        if clinical_assessment.get('requires_immediate_attention', False):
            if confidence < thresholds['critical_condition_confidence']:
                errors.append(f"Critical condition requires confidence ≥ {thresholds['critical_condition_confidence']}")

        if ground_truth:
            performance_metrics = await self._calculate_performance_metrics(
                analysis_results, ground_truth
            )
            test_results.update(performance_metrics)

            if performance_metrics.get('sensitivity', 0) < thresholds['min_sensitivity']:
                errors.append(f"Sensitivity below minimum {thresholds['min_sensitivity']}")

            if performance_metrics.get('specificity', 0) < thresholds['min_specificity']:
                errors.append(f"Specificity below minimum {thresholds['min_specificity']}")

            if performance_metrics.get('false_positive_rate', 1) > thresholds['max_false_positive_rate']:
                errors.append(f"False positive rate above maximum {thresholds['max_false_positive_rate']}")

        if standard == RegulatoryStandard.FDA:
            errors.extend(await self._validate_fda_specific(analysis_results))
        elif standard == RegulatoryStandard.ANVISA:
            errors.extend(await self._validate_anvisa_specific(analysis_results))
        elif standard == RegulatoryStandard.NMSA:
            errors.extend(await self._validate_nmsa_specific(analysis_results))
        elif standard == RegulatoryStandard.EU_MDR:
            errors.extend(await self._validate_eu_mdr_specific(analysis_results))

        integrity_errors = await self._validate_data_integrity(analysis_results)
        errors.extend(integrity_errors)

        traceability_errors = await self._validate_traceability(analysis_results)
        errors.extend(traceability_errors)

        compliant = len(errors) == 0
        confidence_score = max(0.0, 1.0 - len(errors) * 0.1 - len(warnings) * 0.05)

        return ValidationResult(
            standard=standard,
            compliant=compliant,
            confidence_score=confidence_score,
            validation_errors=errors,
            validation_warnings=warnings,
            test_results=test_results,
            timestamp=datetime.utcnow()
        )

    async def _calculate_performance_metrics(
        self,
        analysis_results: dict[str, Any],
        ground_truth: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate performance metrics against ground truth"""

        predictions = analysis_results.get('ai_predictions', {}).get('predictions', {})
        true_labels = ground_truth.get('labels', {})

        metrics = {}

        for condition in predictions.keys():
            if condition in true_labels:
                pred_prob = predictions[condition]
                true_label = true_labels[condition]

                pred_binary = 1 if pred_prob > 0.5 else 0

                tp = 1 if pred_binary == 1 and true_label == 1 else 0
                tn = 1 if pred_binary == 0 and true_label == 0 else 0
                fp = 1 if pred_binary == 1 and true_label == 0 else 0
                fn = 1 if pred_binary == 0 and true_label == 1 else 0

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0  # Default to 1.0 for perfect specificity
                precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # Default to 1.0 for perfect precision

                metrics[f'{condition}_sensitivity'] = sensitivity
                metrics[f'{condition}_specificity'] = specificity
                metrics[f'{condition}_precision'] = precision

        if metrics:
            metrics['sensitivity'] = float(np.mean([v for k, v in metrics.items() if 'sensitivity' in k]))
            metrics['specificity'] = float(np.mean([v for k, v in metrics.items() if 'specificity' in k]))
            metrics['precision'] = float(np.mean([v for k, v in metrics.items() if 'precision' in k]))
            metrics['false_positive_rate'] = 1 - metrics['specificity']

        return metrics

    async def _validate_fda_specific(self, analysis_results: dict[str, Any]) -> list[str]:
        """FDA-specific validation requirements"""
        errors = []

        if 'model_version' not in analysis_results.get('ai_predictions', {}):
            errors.append("FDA: Missing algorithm version documentation")

        if 'confidence' not in analysis_results.get('ai_predictions', {}):
            errors.append("FDA: Missing uncertainty quantification")

        clinical_assessment = analysis_results.get('clinical_assessment', {})
        if not clinical_assessment.get('recommendations'):
            errors.append("FDA: Missing clinical recommendations")

        return errors

    async def _validate_anvisa_specific(self, analysis_results: dict[str, Any]) -> list[str]:
        """ANVISA-specific validation requirements"""
        errors = []

        metadata = analysis_results.get('metadata', {})
        if 'language_support' not in metadata:
            errors.append("ANVISA: Missing language localization documentation")

        if 'population_validation' not in metadata:
            errors.append("ANVISA: Missing Brazilian population validation data")

        return errors

    async def _validate_nmsa_specific(self, analysis_results: dict[str, Any]) -> list[str]:
        """NMSA (China) specific validation requirements"""
        errors = []

        metadata = analysis_results.get('metadata', {})
        if 'nmsa_certification' not in metadata:
            errors.append("NMSA: Missing Chinese regulatory certification")

        if 'data_residency' not in metadata:
            errors.append("NMSA: Missing data residency compliance")

        return errors

    async def _validate_eu_mdr_specific(self, analysis_results: dict[str, Any]) -> list[str]:
        """EU MDR specific validation requirements"""
        errors = []

        metadata = analysis_results.get('metadata', {})
        if 'gdpr_compliant' not in metadata:
            errors.append("EU MDR: Missing GDPR compliance documentation")

        if 'ce_marking' not in metadata:
            errors.append("EU MDR: Missing CE marking documentation")

        if 'surveillance_plan' not in metadata:
            errors.append("EU MDR: Missing post-market surveillance plan")

        return errors

    async def _validate_data_integrity(self, analysis_results: dict[str, Any]) -> list[str]:
        """Validate data integrity and completeness"""
        errors = []

        required_fields = [
            'analysis_id', 'patient_id', 'processing_time_seconds',
            'ai_predictions', 'clinical_assessment'
        ]

        for field in required_fields:
            if field not in analysis_results:
                errors.append(f"Missing required field: {field}")

        if 'processing_time_seconds' in analysis_results:
            processing_time = analysis_results['processing_time_seconds']
            if not isinstance(processing_time, int | float) or processing_time < 0:
                errors.append("Invalid processing time")

        return errors

    async def _validate_traceability(self, analysis_results: dict[str, Any]) -> list[str]:
        """Validate analysis traceability and audit trail"""
        errors = []

        if 'analysis_id' not in analysis_results:
            errors.append("Missing analysis identifier for traceability")

        metadata = analysis_results.get('metadata', {})
        if 'sampling_rate' not in metadata:
            errors.append("Missing signal metadata for traceability")

        ai_predictions = analysis_results.get('ai_predictions', {})
        if 'model_version' not in ai_predictions:
            errors.append("Missing algorithm version for traceability")

        return errors

    async def generate_validation_report(
        self,
        validation_results: dict[RegulatoryStandard, ValidationResult]
    ) -> dict[str, Any]:
        """Generate comprehensive validation report"""

        report = {
            'validation_timestamp': datetime.utcnow().isoformat(),
            'overall_compliance': all(result.compliant for result in validation_results.values()),
            'standards_summary': {},
            'recommendations': [],
            'next_steps': []
        }

        for standard, result in validation_results.items():
            report['standards_summary'][standard.value] = {  # type: ignore[index]
                'compliant': result.compliant,
                'confidence_score': result.confidence_score,
                'error_count': len(result.validation_errors),
                'warning_count': len(result.validation_warnings)
            }

            if not result.compliant:
                report['recommendations'].extend([  # type: ignore[attr-defined]
                    f"{standard.value}: {error}" for error in result.validation_errors
                ])

        if not report['overall_compliance']:
            report['next_steps'] = [
                "Address validation errors before clinical deployment",
                "Conduct additional testing with diverse patient populations",
                "Update algorithm documentation and version control",
                "Implement continuous monitoring and validation"
            ]
        else:
            report['next_steps'] = [
                "Proceed with regulatory submission",
                "Implement post-market surveillance",
                "Monitor performance in clinical practice"
            ]

        return report
