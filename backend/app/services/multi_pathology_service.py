"""
Multi-pathology Detection Service for 71 SCP-ECG conditions
Implements hierarchical detection: Normal→Category→Specific diagnosis
Based on scientific recommendations for CardioAI Pro
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.constants import ClinicalUrgency
from app.core.scp_ecg_conditions import (
    SCP_ECG_CONDITIONS,
    SCPCategory,
    get_conditions_by_urgency,
    get_critical_conditions,
)

logger = logging.getLogger(__name__)

@dataclass
class PathologyDetectionResult:
    """Result of pathology detection with confidence scores"""
    primary_diagnosis: str
    confidence: float
    clinical_urgency: str
    detected_conditions: dict[str, float]
    level_completed: int
    category_probabilities: dict[str, float]
    abnormal_indicators: list[tuple[str, float]]
    processing_time_ms: float

class MultiPathologyService:
    """Hierarchical multi-pathology detection for 71 SCP-ECG conditions"""

    def __init__(self) -> None:
        self.condition_models: dict[str, Any] = {}
        self.scp_conditions = SCP_ECG_CONDITIONS
        self.condition_categories = {code: condition.category for code, condition in SCP_ECG_CONDITIONS.items()}
        self.category_thresholds = self._initialize_category_thresholds()
        self.condition_thresholds = self._initialize_condition_thresholds()

    def _initialize_category_thresholds(self) -> dict[str, float]:
        """Initialize adaptive thresholds for each category"""
        return {
            SCPCategory.NORMAL: 0.99,  # High threshold for normal (NPV > 99%)
            SCPCategory.ARRHYTHMIA: 0.85,  # Sensitivity > 95% target
            SCPCategory.CONDUCTION_DISORDER: 0.90,  # Specificity > 95% target
            SCPCategory.ISCHEMIA: 0.80,  # High sensitivity for STEMI detection
            SCPCategory.HYPERTROPHY: 0.85,  # AUC > 0.90 target
            SCPCategory.AXIS_DEVIATION: 0.80,
            SCPCategory.REPOLARIZATION: 0.85,
            SCPCategory.OTHER: 0.75
        }

    def _initialize_condition_thresholds(self) -> dict[str, float]:
        """Initialize adaptive thresholds for specific conditions"""
        thresholds = {}

        for condition in get_critical_conditions():
            thresholds[condition.code] = 0.70  # Lower threshold = higher sensitivity

        for condition in get_conditions_by_urgency("high"):
            thresholds[condition.code] = 0.75

        for condition in get_conditions_by_urgency("medium"):
            thresholds[condition.code] = 0.80

        for condition in get_conditions_by_urgency("low"):
            thresholds[condition.code] = 0.85

        return thresholds

    async def detect_pathologies_hierarchical(
        self,
        signal: np.ndarray,
        features: dict[str, Any],
        preprocessing_quality: float
    ) -> PathologyDetectionResult:
        """
        Hierarchical pathology detection
        Level 1: Normal vs Abnormal (NPV > 99%)
        Level 2: Category classification
        Level 3: Specific diagnosis with confidence scores
        """
        import time
        start_time = time.time()

        try:
            normal_abnormal_result = await self._level1_normal_vs_abnormal(signal, features)

            if (normal_abnormal_result['is_normal'] and
                normal_abnormal_result['confidence'] > self.category_thresholds[SCPCategory.NORMAL]):

                processing_time = (time.time() - start_time) * 1000
                return PathologyDetectionResult(
                    primary_diagnosis='NORM',
                    confidence=normal_abnormal_result['confidence'],
                    clinical_urgency=ClinicalUrgency.LOW,
                    detected_conditions={'NORM': normal_abnormal_result['confidence']},
                    level_completed=1,
                    category_probabilities={'normal': normal_abnormal_result['normal_probability']},
                    abnormal_indicators=normal_abnormal_result['abnormal_indicators'],
                    processing_time_ms=processing_time
                )

            category_result = await self._level2_category_classification(signal, features)

            specific_result = await self._level3_specific_diagnosis(
                signal, features, category_result['predicted_category']
            )

            final_result = await self._compile_final_diagnosis(
                normal_abnormal_result, category_result, specific_result
            )

            processing_time = (time.time() - start_time) * 1000
            final_result.processing_time_ms = processing_time

            return final_result

        except Exception as e:
            logger.error(f"Error in hierarchical pathology detection: {e}")
            processing_time = (time.time() - start_time) * 1000

            return PathologyDetectionResult(
                primary_diagnosis='NONSPECIFIC',
                confidence=0.5,
                clinical_urgency=ClinicalUrgency.LOW,
                detected_conditions={'NONSPECIFIC': 0.5},
                level_completed=0,
                category_probabilities={},
                abnormal_indicators=[('processing_error', 1.0)],
                processing_time_ms=processing_time
            )

    async def _level1_normal_vs_abnormal(
        self, signal: np.ndarray, features: dict[str, Any]
    ) -> dict[str, Any]:
        """Level 1: High-sensitivity normal vs abnormal screening (NPV > 99%)"""

        abnormal_indicators = []

        hr = features.get('heart_rate', 70)
        if hr < 50:
            abnormal_indicators.append(('bradycardia', min(0.9, (50 - hr) / 20)))
        elif hr > 100:
            abnormal_indicators.append(('tachycardia', min(0.9, (hr - 100) / 50)))

        rr_std = features.get('rr_std', 0)
        rr_mean = features.get('rr_mean', 1000)
        if rr_mean > 0:
            rr_cv = rr_std / rr_mean
            if rr_cv > 0.15:  # High variability threshold
                abnormal_indicators.append(('rr_variability', min(0.8, rr_cv * 2)))

        qrs_duration = features.get('qrs_duration', 100)
        if qrs_duration > 120:
            abnormal_indicators.append(('wide_qrs', min(0.8, (qrs_duration - 120) / 50)))
        elif qrs_duration < 80:
            abnormal_indicators.append(('narrow_qrs', min(0.6, (80 - qrs_duration) / 20)))

        pr_interval = features.get('pr_interval', 160)
        if pr_interval > 200:
            abnormal_indicators.append(('long_pr', min(0.7, (pr_interval - 200) / 100)))
        elif pr_interval < 120:
            abnormal_indicators.append(('short_pr', min(0.7, (120 - pr_interval) / 40)))

        qtc = features.get('qtc', 420)
        if qtc > 450:  # Men threshold, more conservative
            abnormal_indicators.append(('long_qt', min(0.8, (qtc - 450) / 100)))
        elif qtc < 350:
            abnormal_indicators.append(('short_qt', min(0.7, (350 - qtc) / 50)))

        st_elevation = features.get('st_elevation_max', 0)
        st_depression = features.get('st_depression_max', 0)
        if st_elevation > 1.0:  # >1mm elevation
            abnormal_indicators.append(('st_elevation', min(0.9, st_elevation / 3)))
        if st_depression > 1.0:  # >1mm depression
            abnormal_indicators.append(('st_depression', min(0.8, st_depression / 3)))

        t_wave_inversion = features.get('t_wave_inversion_leads', 0)
        if t_wave_inversion > 2:  # More than 2 leads
            abnormal_indicators.append(('t_wave_inversion', min(0.7, t_wave_inversion / 6)))

        qrs_axis = features.get('qrs_axis', 60)
        if qrs_axis < -30:  # Left axis deviation
            abnormal_indicators.append(('left_axis', min(0.6, abs(qrs_axis + 30) / 60)))
        elif qrs_axis > 90:  # Right axis deviation
            abnormal_indicators.append(('right_axis', min(0.6, (qrs_axis - 90) / 90)))

        if abnormal_indicators:
            weighted_score = 0
            total_weight = 0.0

            for indicator, score in abnormal_indicators:
                weight = 1.0
                if indicator in ['st_elevation', 'wide_qrs', 'long_qt']:
                    weight = 2.0  # Critical indicators
                elif indicator in ['bradycardia', 'tachycardia', 'st_depression']:
                    weight = 1.5  # Important indicators

                weighted_score += score * weight
                total_weight += weight

            abnormal_prob = min(weighted_score / total_weight if total_weight > 0 else 0, 0.95)
            normal_prob = 1.0 - abnormal_prob
        else:
            normal_prob = 0.98  # High confidence normal
            abnormal_prob = 0.02

        return {
            'is_normal': normal_prob > 0.5,
            'confidence': max(normal_prob, abnormal_prob),
            'normal_probability': normal_prob,
            'abnormal_probability': abnormal_prob,
            'abnormal_indicators': abnormal_indicators
        }

    async def _level2_category_classification(
        self, signal: np.ndarray, features: dict[str, Any]
    ) -> dict[str, Any]:
        """Level 2: Category classification for abnormal ECGs"""

        category_scores = {}

        arrhythmia_score = 0.0
        hr = features.get('heart_rate', 70)
        rr_std = features.get('rr_std', 0)
        rr_mean = features.get('rr_mean', 1000)

        if hr < 50 or hr > 100:
            arrhythmia_score += 0.3
        if rr_mean > 0 and (rr_std / rr_mean) > 0.15:
            arrhythmia_score += 0.4
        if features.get('premature_beats', 0) > 0:
            arrhythmia_score += 0.5

        category_scores[SCPCategory.ARRHYTHMIA] = min(arrhythmia_score, 1.0)

        conduction_score = 0.0
        qrs_duration = features.get('qrs_duration', 100)
        pr_interval = features.get('pr_interval', 160)

        if qrs_duration > 120:
            conduction_score += 0.6
        if pr_interval > 200:
            conduction_score += 0.4
        if features.get('av_block_degree', 0) > 0:
            conduction_score += 0.7

        category_scores[SCPCategory.CONDUCTION_DISORDER] = min(conduction_score, 1.0)

        ischemia_score = 0.0
        st_elevation = features.get('st_elevation_max', 0)
        st_depression = features.get('st_depression_max', 0)
        t_wave_inversion = features.get('t_wave_inversion_leads', 0)

        if st_elevation > 1.0:
            ischemia_score += 0.8  # High weight for ST elevation
        if st_depression > 1.0:
            ischemia_score += 0.6
        if t_wave_inversion > 2:
            ischemia_score += 0.4
        if features.get('q_waves_pathological', False):
            ischemia_score += 0.5

        category_scores[SCPCategory.ISCHEMIA] = min(ischemia_score, 1.0)

        hypertrophy_score = 0.0
        r_wave_v5 = features.get('r_wave_v5', 0)
        s_wave_v1 = features.get('s_wave_v1', 0)

        if r_wave_v5 + s_wave_v1 > 35:  # Sokolow-Lyon criteria
            hypertrophy_score += 0.6
        if features.get('left_atrial_abnormality', False):
            hypertrophy_score += 0.3
        if features.get('right_atrial_abnormality', False):
            hypertrophy_score += 0.3

        category_scores[SCPCategory.HYPERTROPHY] = min(hypertrophy_score, 1.0)

        axis_score = 0.0
        qrs_axis = features.get('qrs_axis', 60)

        if qrs_axis < -30 or qrs_axis > 90:
            axis_score = 0.8
        elif qrs_axis < -90 or qrs_axis > 180:
            axis_score = 0.9  # Extreme axis deviation

        category_scores[SCPCategory.AXIS_DEVIATION] = axis_score

        repol_score = 0.0
        qtc = features.get('qtc', 420)

        if qtc > 450 or qtc < 350:
            repol_score += 0.6
        if features.get('early_repolarization', False):
            repol_score += 0.4
        if features.get('brugada_pattern', False):
            repol_score += 0.8

        category_scores[SCPCategory.REPOLARIZATION] = min(repol_score, 1.0)

        other_indicators = [
            features.get('low_voltage', False),
            features.get('paced_rhythm', False),
            features.get('artifact_present', False)
        ]
        other_score = sum(other_indicators) * 0.3
        category_scores[SCPCategory.OTHER] = min(other_score, 1.0)

        predicted_category = max(category_scores.items(), key=lambda x: x[1])[0]

        return {
            'predicted_category': predicted_category,
            'category_probabilities': category_scores,
            'confidence': category_scores[predicted_category]
        }

    async def _level3_specific_diagnosis(
        self, signal: np.ndarray, features: dict[str, Any], predicted_category: str
    ) -> dict[str, Any]:
        """Level 3: Specific diagnosis within predicted category"""

        condition_scores = {}

        if predicted_category == SCPCategory.ARRHYTHMIA:
            condition_scores = await self._diagnose_arrhythmias(features)
        elif predicted_category == SCPCategory.CONDUCTION_DISORDER:
            condition_scores = await self._diagnose_conduction_disorders(features)
        elif predicted_category == SCPCategory.ISCHEMIA:
            condition_scores = await self._diagnose_ischemia(features)
        elif predicted_category == SCPCategory.HYPERTROPHY:
            condition_scores = await self._diagnose_hypertrophy(features)
        elif predicted_category == SCPCategory.AXIS_DEVIATION:
            condition_scores = await self._diagnose_axis_deviation(features)
        elif predicted_category == SCPCategory.REPOLARIZATION:
            condition_scores = await self._diagnose_repolarization(features)
        else:  # OTHER category
            condition_scores = await self._diagnose_other_conditions(features)

        filtered_conditions = {}
        for condition_code, score in condition_scores.items():
            threshold = self.condition_thresholds.get(condition_code, 0.8)
            if score >= threshold:
                filtered_conditions[condition_code] = score

        if filtered_conditions:
            primary_diagnosis = max(filtered_conditions.items(), key=lambda x: x[1])[0]
            primary_confidence = filtered_conditions[primary_diagnosis]
        else:
            if condition_scores:
                primary_diagnosis = max(condition_scores.items(), key=lambda x: x[1])[0]
                primary_confidence = condition_scores[primary_diagnosis]
            else:
                primary_diagnosis = 'NONSPECIFIC'
                primary_confidence = 0.5

        return {
            'primary_diagnosis': primary_diagnosis,
            'confidence': primary_confidence,
            'all_conditions': condition_scores,
            'filtered_conditions': filtered_conditions
        }

    async def _diagnose_arrhythmias(self, features: dict[str, Any]) -> dict[str, float]:
        """Diagnose specific arrhythmias"""
        scores = {}

        hr = features.get('heart_rate', 70)
        rr_std = features.get('rr_std', 0)
        rr_mean = features.get('rr_mean', 1000)

        if hr < 60:
            scores['BRADY'] = min(0.9, (60 - hr) / 30)

        if hr > 100:
            scores['TACHY'] = min(0.9, (hr - 100) / 50)

        if rr_mean > 0:
            rr_cv = rr_std / rr_mean
            if rr_cv > 0.2 and hr > 90:
                scores['AFIB'] = min(0.9, rr_cv * 2)

        pvc_count = features.get('pvc_count', 0)
        pac_count = features.get('pac_count', 0)

        if pvc_count > 0:
            scores['PVC'] = min(0.9, pvc_count / 10)
        if pac_count > 0:
            scores['PAC'] = min(0.8, pac_count / 10)

        qrs_duration = features.get('qrs_duration', 100)
        if qrs_duration > 120 and hr > 150:
            scores['VTAC'] = 0.8

        return scores

    async def _diagnose_conduction_disorders(self, features: dict[str, Any]) -> dict[str, float]:
        """Diagnose specific conduction disorders"""
        scores = {}

        pr_interval = features.get('pr_interval', 160)
        qrs_duration = features.get('qrs_duration', 100)
        av_block_degree = features.get('av_block_degree', 0)

        if pr_interval > 200:
            scores['AVB1'] = min(0.9, (pr_interval - 200) / 100)

        if av_block_degree == 2:
            scores['AVB2M1'] = 0.8  # Default to Mobitz I

        if av_block_degree == 3:
            scores['AVB3'] = 0.9

        if qrs_duration > 120:
            if features.get('rbbb_pattern', False):
                scores['RBBB'] = 0.8
            elif features.get('lbbb_pattern', False):
                scores['LBBB'] = 0.8

        return scores

    async def _diagnose_ischemia(self, features: dict[str, Any]) -> dict[str, float]:
        """Diagnose specific ischemic conditions"""
        scores = {}

        st_elevation = features.get('st_elevation_max', 0)
        st_depression = features.get('st_depression_max', 0)

        if st_elevation > 2.0:
            scores['STEMI'] = min(0.95, st_elevation / 5)

        if st_depression > 1.0:
            scores['NSTEMI'] = min(0.8, st_depression / 3)

        if st_depression > 0.5 or features.get('t_wave_inversion_leads', 0) > 2:
            scores['ISCHEMIA'] = 0.7

        return scores

    async def _diagnose_hypertrophy(self, features: dict[str, Any]) -> dict[str, float]:
        """Diagnose specific hypertrophy conditions"""
        scores = {}

        r_wave_v5 = features.get('r_wave_v5', 0)
        s_wave_v1 = features.get('s_wave_v1', 0)

        if r_wave_v5 + s_wave_v1 > 35:
            scores['LVH'] = min(0.9, (r_wave_v5 + s_wave_v1 - 35) / 20)

        if features.get('r_wave_v1', 0) > 7:
            scores['RVH'] = 0.8

        return scores

    async def _diagnose_axis_deviation(self, features: dict[str, Any]) -> dict[str, float]:
        """Diagnose axis deviation conditions"""
        scores = {}

        qrs_axis = features.get('qrs_axis', 60)

        if qrs_axis < -30:
            scores['LAD'] = min(0.9, abs(qrs_axis + 30) / 60)
        elif qrs_axis > 90:
            scores['RAD'] = min(0.9, (qrs_axis - 90) / 90)

        return scores

    async def _diagnose_repolarization(self, features: dict[str, Any]) -> dict[str, float]:
        """Diagnose repolarization abnormalities"""
        scores = {}

        qtc = features.get('qtc', 420)

        if qtc > 450:
            scores['LQTS'] = min(0.9, (qtc - 450) / 100)
        elif qtc < 350:
            scores['SQTS'] = min(0.9, (350 - qtc) / 50)

        return scores

    async def _diagnose_other_conditions(self, features: dict[str, Any]) -> dict[str, float]:
        """Diagnose other conditions"""
        scores = {}

        if features.get('paced_rhythm', False):
            scores['PACE'] = 0.9

        if features.get('artifact_present', False):
            scores['ARTIFACT'] = 0.8

        return scores

    async def _compile_final_diagnosis(
        self,
        normal_abnormal_result: dict[str, Any],
        category_result: dict[str, Any],
        specific_result: dict[str, Any]
    ) -> PathologyDetectionResult:
        """Compile final diagnosis from all levels"""

        primary_diagnosis = specific_result['primary_diagnosis']
        confidence = specific_result['confidence']

        condition = SCP_ECG_CONDITIONS.get(primary_diagnosis)
        if condition:
            clinical_urgency = condition.clinical_urgency
        else:
            clinical_urgency = ClinicalUrgency.LOW

        return PathologyDetectionResult(
            primary_diagnosis=primary_diagnosis,
            confidence=confidence,
            clinical_urgency=clinical_urgency,
            detected_conditions=specific_result['filtered_conditions'],
            level_completed=3,
            category_probabilities=category_result['category_probabilities'],
            abnormal_indicators=normal_abnormal_result['abnormal_indicators'],
            processing_time_ms=0.0  # Will be set by caller
        )

    async def detect_multi_pathology(
        self,
        signal: np.ndarray,
        features: dict[str, Any],
        preprocessing_quality: float
    ) -> dict[str, Any]:
        """
        Wrapper method for detect_pathologies_hierarchical to match expected interface
        Returns dict format expected by hybrid_ecg_service.py
        """
        try:
            result = await self.detect_pathologies_hierarchical(signal, features, preprocessing_quality)

            # Convert PathologyDetectionResult to dict format expected by caller
            return {
                'primary_diagnosis': result.primary_diagnosis,
                'confidence': result.confidence,
                'clinical_urgency': result.clinical_urgency,
                'detected_conditions': result.detected_conditions,
                'level_completed': result.level_completed,
                'secondary_diagnoses': [],  # Extract from detected_conditions if needed
                'requires_immediate_attention': result.clinical_urgency in [ClinicalUrgency.CRITICAL, ClinicalUrgency.HIGH],
                'recommendations': self._generate_recommendations(result),
                'icd10_codes': self._get_icd10_codes(result.primary_diagnosis),
                'category_probabilities': result.category_probabilities,
                'abnormal_indicators': result.abnormal_indicators,
                'processing_time_ms': result.processing_time_ms
            }

        except Exception as e:
            logger.error(f"Error in detect_multi_pathology: {e}")
            return {
                'primary_diagnosis': 'NONSPECIFIC',
                'confidence': 0.5,
                'clinical_urgency': ClinicalUrgency.LOW,
                'detected_conditions': {'NONSPECIFIC': 0.5},
                'level_completed': 0,
                'secondary_diagnoses': [],
                'requires_immediate_attention': False,
                'recommendations': ['Unable to complete analysis due to processing error'],
                'icd10_codes': [],
                'category_probabilities': {},
                'abnormal_indicators': [('processing_error', 1.0)],
                'processing_time_ms': 0.0
            }

    def _generate_recommendations(self, result: PathologyDetectionResult) -> list[str]:
        """Generate clinical recommendations based on diagnosis"""
        recommendations = []

        if result.clinical_urgency == ClinicalUrgency.CRITICAL:
            recommendations.append('URGENT: Immediate cardiology consultation required')
            recommendations.append('Consider emergency intervention if clinically indicated')
        elif result.clinical_urgency == ClinicalUrgency.HIGH:
            recommendations.append('Cardiology consultation recommended within 24 hours')
            recommendations.append('Monitor patient closely for symptom progression')
        elif result.clinical_urgency == ClinicalUrgency.MEDIUM:
            recommendations.append('Follow-up with cardiology as clinically appropriate')
            recommendations.append('Continue routine cardiac monitoring')
        else:
            recommendations.append('Routine follow-up as clinically indicated')

        # Add specific recommendations based on diagnosis
        if result.primary_diagnosis == 'AFIB':
            recommendations.append('Assess stroke risk and consider anticoagulation')
        elif result.primary_diagnosis in ['STEMI', 'NSTEMI']:
            recommendations.append('Activate cardiac catheterization protocol if indicated')
        elif result.primary_diagnosis in ['BRADY', 'AVB3']:
            recommendations.append('Evaluate for pacemaker indication')

        return recommendations

    def _get_icd10_codes(self, diagnosis: str) -> list[str]:
        """Get ICD-10 codes for diagnosis"""
        icd10_mapping = {
            'AFIB': ['I48.0', 'I48.1', 'I48.2'],
            'STEMI': ['I21.0', 'I21.1', 'I21.2', 'I21.3'],
            'NSTEMI': ['I21.4'],
            'BRADY': ['R00.1'],
            'TACHY': ['R00.0'],
            'AVB1': ['I44.0'],
            'AVB2M1': ['I44.1'],
            'AVB2M2': ['I44.1'],
            'AVB3': ['I44.2'],
            'VTAC': ['I47.2'],
            'VFIB': ['I49.01'],
            'PVC': ['I49.3'],
            'PAC': ['I49.1']
        }

        return icd10_mapping.get(diagnosis, [])
