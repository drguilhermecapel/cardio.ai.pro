"""
Continuous Learning and Feedback Loop System for ECG Analysis
Implements continuous improvement mechanisms as specified in Phase 6 optimization
Based on CardioAI Pro optimization guide requirements
"""

import logging
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """Individual feedback entry from expert diagnosis"""
    ecg_id: str
    ai_prediction: Dict[str, float]
    expert_diagnosis: Dict[str, Any]
    timestamp: datetime
    discrepancy: bool
    confidence_score: float
    processing_time: float
    signal_quality: float
    expert_confidence: Optional[float] = None
    clinical_context: Optional[str] = None
    follow_up_outcome: Optional[str] = None


@dataclass
class ErrorPattern:
    """Identified error pattern in AI predictions"""
    pattern_type: str
    frequency: int
    conditions: List[str]
    common_features: Dict[str, Any]
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    recommendations: List[str]
    first_observed: datetime
    last_observed: datetime


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    overall_accuracy: float
    per_class_sensitivity: Dict[str, float]
    per_class_specificity: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: npt.NDArray[np.int32]
    roc_auc_scores: Dict[str, float]
    processing_time_stats: Dict[str, float]
    confidence_calibration_error: float
    false_positive_rate: float
    false_negative_rate: float
    critical_miss_rate: float  # For life-threatening conditions


@dataclass
class RetrainingTrigger:
    """Conditions that trigger model retraining"""
    accuracy_threshold: float = 0.85
    critical_miss_threshold: int = 5
    feedback_buffer_size: int = 100
    time_window_days: int = 30
    confidence_degradation_threshold: float = 0.1
    new_error_pattern_threshold: int = 10


class PerformanceTracker:
    """
    Tracks model performance over time and identifies degradation
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.performance_history: deque = deque(maxlen=window_size)
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.current_metrics: Optional[PerformanceMetrics] = None
        
    def update_performance(self, predictions: List[Dict[str, float]], 
                          ground_truth: List[Dict[str, Any]],
                          processing_times: List[float],
                          confidence_scores: List[float]) -> PerformanceMetrics:
        """
        Update performance metrics with new predictions
        
        Args:
            predictions: List of AI predictions
            ground_truth: List of expert diagnoses
            processing_times: Processing time for each prediction
            confidence_scores: Confidence scores for predictions
            
        Returns:
            Updated performance metrics
        """
        metrics = self._calculate_comprehensive_metrics(
            predictions, ground_truth, processing_times, confidence_scores
        )
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'sample_count': len(predictions)
        })
        
        self.current_metrics = metrics
        
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
            logger.info("Baseline performance metrics established")
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, predictions: List[Dict[str, float]], 
                                       ground_truth: List[Dict[str, Any]],
                                       processing_times: List[float],
                                       confidence_scores: List[float]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        all_conditions = set()
        for pred in predictions:
            all_conditions.update(pred.keys())
        for gt in ground_truth:
            if 'conditions' in gt:
                all_conditions.update(gt['conditions'].keys())
        
        all_conditions = list(all_conditions)
        
        per_class_sensitivity = {}
        per_class_specificity = {}
        per_class_precision = {}
        per_class_f1 = {}
        roc_auc_scores = {}
        
        for condition in all_conditions:
            y_true = []
            y_pred = []
            y_scores = []
            
            for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
                if 'conditions' in gt and condition in gt['conditions']:
                    true_label = 1 if gt['conditions'][condition].get('detected', False) else 0
                else:
                    true_label = 0
                
                pred_score = pred.get(condition, 0.0)
                pred_label = 1 if pred_score > 0.5 else 0
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                y_scores.append(pred_score)
            
            if sum(y_true) > 0 and sum(y_pred) > 0:
                try:
                    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
                    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
                    per_class_sensitivity[condition] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    
                    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
                    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
                    per_class_specificity[condition] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    
                    per_class_precision[condition] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    
                    precision = per_class_precision[condition]
                    sensitivity = per_class_sensitivity[condition]
                    per_class_f1[condition] = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
                    
                    if len(set(y_true)) > 1:  # Need both classes for AUC
                        roc_auc_scores[condition] = roc_auc_score(y_true, y_scores)
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate metrics for {condition}: {e}")
        
        total_correct = 0
        total_predictions = 0
        
        for pred, gt in zip(predictions, ground_truth):
            for condition in all_conditions:
                true_label = 0
                if 'conditions' in gt and condition in gt['conditions']:
                    true_label = 1 if gt['conditions'][condition].get('detected', False) else 0
                
                pred_label = 1 if pred.get(condition, 0.0) > 0.5 else 0
                
                if true_label == pred_label:
                    total_correct += 1
                total_predictions += 1
        
        overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        processing_time_stats = {
            'mean': float(np.mean(processing_times)),
            'std': float(np.std(processing_times)),
            'min': float(np.min(processing_times)),
            'max': float(np.max(processing_times)),
            'p95': float(np.percentile(processing_times, 95))
        }
        
        confidence_calibration_error = self._calculate_calibration_error(
            confidence_scores, [1 if any(pred.values()) > 0.5 else 0 for pred in predictions]
        )
        
        critical_conditions = ['ventricular_fibrillation', 'ventricular_tachycardia', 'complete_heart_block']
        critical_misses = 0
        critical_cases = 0
        
        for pred, gt in zip(predictions, ground_truth):
            for condition in critical_conditions:
                if 'conditions' in gt and condition in gt['conditions']:
                    if gt['conditions'][condition].get('detected', False):
                        critical_cases += 1
                        if pred.get(condition, 0.0) < 0.5:  # Missed critical condition
                            critical_misses += 1
        
        critical_miss_rate = critical_misses / critical_cases if critical_cases > 0 else 0.0
        
        return PerformanceMetrics(
            overall_accuracy=overall_accuracy,
            per_class_sensitivity=per_class_sensitivity,
            per_class_specificity=per_class_specificity,
            per_class_precision=per_class_precision,
            per_class_f1=per_class_f1,
            confusion_matrix=np.array([[0]]),  # Simplified for now
            roc_auc_scores=roc_auc_scores,
            processing_time_stats=processing_time_stats,
            confidence_calibration_error=confidence_calibration_error,
            false_positive_rate=0.0,  # Calculate if needed
            false_negative_rate=0.0,  # Calculate if needed
            critical_miss_rate=critical_miss_rate
        )
    
    def _calculate_calibration_error(self, confidence_scores: List[float], 
                                   correct_predictions: List[int]) -> float:
        """Calculate expected calibration error"""
        if len(confidence_scores) == 0:
            return 0.0
        
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [(conf > bin_lower) and (conf <= bin_upper) 
                     for conf in confidence_scores]
            prop_in_bin = sum(in_bin) / len(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = sum(correct_predictions[i] for i, in_b in enumerate(in_bin) if in_b) / sum(in_bin)
                avg_confidence_in_bin = sum(confidence_scores[i] for i, in_b in enumerate(in_bin) if in_b) / sum(in_bin)
                ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def detect_performance_degradation(self) -> Dict[str, Any]:
        """Detect if model performance has degraded"""
        if not self.baseline_metrics or not self.current_metrics:
            return {'degradation_detected': False, 'reason': 'Insufficient data'}
        
        degradation_issues = []
        
        accuracy_drop = self.baseline_metrics.overall_accuracy - self.current_metrics.overall_accuracy
        if accuracy_drop > 0.05:  # 5% drop threshold
            degradation_issues.append(f"Overall accuracy dropped by {accuracy_drop:.3f}")
        
        if self.current_metrics.critical_miss_rate > 0.02:  # 2% threshold for critical conditions
            degradation_issues.append(f"Critical miss rate too high: {self.current_metrics.critical_miss_rate:.3f}")
        
        calibration_degradation = (self.current_metrics.confidence_calibration_error - 
                                 self.baseline_metrics.confidence_calibration_error)
        if calibration_degradation > 0.1:
            degradation_issues.append(f"Confidence calibration degraded by {calibration_degradation:.3f}")
        
        return {
            'degradation_detected': len(degradation_issues) > 0,
            'issues': degradation_issues,
            'severity': 'HIGH' if any('critical' in issue.lower() for issue in degradation_issues) else 'MEDIUM'
        }


class ErrorPatternAnalyzer:
    """
    Analyzes feedback to identify common error patterns
    """
    
    def __init__(self):
        self.identified_patterns: List[ErrorPattern] = []
        self.pattern_history: List[Dict[str, Any]] = []
        
    def analyze_error_patterns(self, feedback_entries: List[FeedbackEntry]) -> List[ErrorPattern]:
        """
        Analyze feedback entries to identify error patterns
        
        Args:
            feedback_entries: List of feedback entries with discrepancies
            
        Returns:
            List of identified error patterns
        """
        discrepancies = [entry for entry in feedback_entries if entry.discrepancy]
        
        if len(discrepancies) < 5:  # Need minimum samples
            return []
        
        patterns = []
        
        low_quality_errors = [entry for entry in discrepancies if entry.signal_quality < 0.7]
        if len(low_quality_errors) >= 3:
            patterns.append(ErrorPattern(
                pattern_type="LOW_SIGNAL_QUALITY",
                frequency=len(low_quality_errors),
                conditions=list(set([list(entry.ai_prediction.keys())[0] for entry in low_quality_errors])),
                common_features={'signal_quality_threshold': 0.7},
                severity="MEDIUM",
                recommendations=[
                    "Improve signal quality assessment",
                    "Add quality-based confidence adjustment",
                    "Implement better preprocessing for low-quality signals"
                ],
                first_observed=min(entry.timestamp for entry in low_quality_errors),
                last_observed=max(entry.timestamp for entry in low_quality_errors)
            ))
        
        overconfident_errors = [entry for entry in discrepancies if entry.confidence_score > 0.8]
        if len(overconfident_errors) >= 3:
            patterns.append(ErrorPattern(
                pattern_type="OVERCONFIDENT_PREDICTIONS",
                frequency=len(overconfident_errors),
                conditions=list(set([list(entry.ai_prediction.keys())[0] for entry in overconfident_errors])),
                common_features={'confidence_threshold': 0.8},
                severity="HIGH",
                recommendations=[
                    "Improve confidence calibration",
                    "Add uncertainty quantification",
                    "Implement ensemble methods for better confidence estimation"
                ],
                first_observed=min(entry.timestamp for entry in overconfident_errors),
                last_observed=max(entry.timestamp for entry in overconfident_errors)
            ))
        
        condition_errors = defaultdict(list)
        for entry in discrepancies:
            for condition in entry.ai_prediction.keys():
                condition_errors[condition].append(entry)
        
        for condition, errors in condition_errors.items():
            if len(errors) >= 5:  # Frequent errors for specific condition
                patterns.append(ErrorPattern(
                    pattern_type="CONDITION_SPECIFIC_ERROR",
                    frequency=len(errors),
                    conditions=[condition],
                    common_features={'condition': condition},
                    severity="HIGH" if condition in ['ventricular_fibrillation', 'ventricular_tachycardia'] else "MEDIUM",
                    recommendations=[
                        f"Retrain model specifically for {condition}",
                        f"Collect more training data for {condition}",
                        f"Review feature extraction for {condition}"
                    ],
                    first_observed=min(entry.timestamp for entry in errors),
                    last_observed=max(entry.timestamp for entry in errors)
                ))
        
        self.identified_patterns.extend(patterns)
        
        for pattern in patterns:
            logger.warning(f"Identified error pattern: {pattern.pattern_type} "
                         f"(frequency: {pattern.frequency}, severity: {pattern.severity})")
        
        return patterns
    
    def generate_recommendations(self, patterns: List[ErrorPattern]) -> List[str]:
        """Generate actionable recommendations based on error patterns"""
        recommendations = []
        
        critical_patterns = [p for p in patterns if p.severity == "CRITICAL"]
        high_patterns = [p for p in patterns if p.severity == "HIGH"]
        
        if critical_patterns:
            recommendations.append("URGENT: Critical error patterns detected - immediate model review required")
            for pattern in critical_patterns:
                recommendations.extend(pattern.recommendations)
        
        if high_patterns:
            recommendations.append("HIGH PRIORITY: Significant error patterns require attention")
            for pattern in high_patterns:
                recommendations.extend(pattern.recommendations[:2])  # Top 2 recommendations
        
        if len(patterns) > 5:
            recommendations.append("Consider comprehensive model retraining due to multiple error patterns")
        
        return list(set(recommendations))  # Remove duplicates


class ContinuousLearningSystem:
    """
    Main continuous learning system that orchestrates feedback collection,
    performance monitoring, and retraining decisions
    """
    
    def __init__(self, model: Any, config: Optional[RetrainingTrigger] = None):
        self.model = model
        self.config = config or RetrainingTrigger()
        self.feedback_buffer: List[FeedbackEntry] = []
        self.performance_tracker = PerformanceTracker()
        self.error_analyzer = ErrorPatternAnalyzer()
        
        self.last_retraining = datetime.now()
        self.retraining_in_progress = False
        self.retraining_history: List[Dict[str, Any]] = []
        
    def collect_feedback(self, ecg_id: str, ai_prediction: Dict[str, float], 
                        expert_diagnosis: Dict[str, Any], 
                        confidence_score: float = 0.0,
                        processing_time: float = 0.0,
                        signal_quality: float = 1.0,
                        expert_confidence: Optional[float] = None,
                        clinical_context: Optional[str] = None) -> None:
        """
        Collect feedback from expert diagnosis
        
        Args:
            ecg_id: Unique identifier for the ECG
            ai_prediction: AI model predictions
            expert_diagnosis: Expert diagnosis with conditions and confidence
            confidence_score: AI confidence score
            processing_time: Time taken for AI analysis
            signal_quality: Quality score of the ECG signal
            expert_confidence: Expert's confidence in their diagnosis
            clinical_context: Additional clinical context
        """
        discrepancy = self._check_discrepancy(ai_prediction, expert_diagnosis)
        
        feedback_entry = FeedbackEntry(
            ecg_id=ecg_id,
            ai_prediction=ai_prediction,
            expert_diagnosis=expert_diagnosis,
            timestamp=datetime.now(),
            discrepancy=discrepancy,
            confidence_score=confidence_score,
            processing_time=processing_time,
            signal_quality=signal_quality,
            expert_confidence=expert_confidence,
            clinical_context=clinical_context
        )
        
        self.feedback_buffer.append(feedback_entry)
        
        logger.info(f"Collected feedback for ECG {ecg_id}: discrepancy={discrepancy}")
        
        if len(self.feedback_buffer) >= self.config.feedback_buffer_size:
            self._analyze_performance()
    
    def _check_discrepancy(self, ai_prediction: Dict[str, float], 
                          expert_diagnosis: Dict[str, Any]) -> bool:
        """Check if there's a significant discrepancy between AI and expert"""
        if 'conditions' not in expert_diagnosis:
            return False
        
        for condition, ai_score in ai_prediction.items():
            ai_detected = ai_score > 0.5
            
            if condition in expert_diagnosis['conditions']:
                expert_detected = expert_diagnosis['conditions'][condition].get('detected', False)
                
                if ai_detected != expert_detected:
                    return True
        
        return False
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze current performance and determine if retraining is needed
        
        Returns:
            Analysis report with recommendations
        """
        logger.info(f"Analyzing performance with {len(self.feedback_buffer)} feedback entries")
        
        predictions = [entry.ai_prediction for entry in self.feedback_buffer]
        ground_truth = [entry.expert_diagnosis for entry in self.feedback_buffer]
        processing_times = [entry.processing_time for entry in self.feedback_buffer]
        confidence_scores = [entry.confidence_score for entry in self.feedback_buffer]
        
        current_metrics = self.performance_tracker.update_performance(
            predictions, ground_truth, processing_times, confidence_scores
        )
        
        degradation_analysis = self.performance_tracker.detect_performance_degradation()
        
        error_patterns = self.error_analyzer.analyze_error_patterns(self.feedback_buffer)
        
        recommendations = self.error_analyzer.generate_recommendations(error_patterns)
        
        retraining_needed = self._should_trigger_retraining(
            current_metrics, degradation_analysis, error_patterns
        )
        
        analysis_report = {
            'timestamp': datetime.now(),
            'sample_count': len(self.feedback_buffer),
            'current_metrics': current_metrics,
            'degradation_analysis': degradation_analysis,
            'error_patterns': error_patterns,
            'recommendations': recommendations,
            'retraining_needed': retraining_needed,
            'retraining_reasons': self._get_retraining_reasons(
                current_metrics, degradation_analysis, error_patterns
            )
        }
        
        logger.info(f"Performance analysis completed: "
                   f"accuracy={current_metrics.overall_accuracy:.3f}, "
                   f"critical_miss_rate={current_metrics.critical_miss_rate:.3f}, "
                   f"retraining_needed={retraining_needed}")
        
        if retraining_needed and not self.retraining_in_progress:
            self._trigger_retraining(analysis_report)
        
        self.feedback_buffer = []
        
        return analysis_report
    
    def _should_trigger_retraining(self, metrics: PerformanceMetrics, 
                                 degradation: Dict[str, Any],
                                 patterns: List[ErrorPattern]) -> bool:
        """Determine if retraining should be triggered"""
        if metrics.overall_accuracy < self.config.accuracy_threshold:
            return True
        
        if metrics.critical_miss_rate > (self.config.critical_miss_threshold / 100):
            return True
        
        if degradation['degradation_detected'] and degradation.get('severity') == 'HIGH':
            return True
        
        critical_patterns = [p for p in patterns if p.severity == "CRITICAL"]
        if len(critical_patterns) > 0:
            return True
        
        high_severity_patterns = [p for p in patterns if p.severity == "HIGH"]
        if len(high_severity_patterns) >= 3:
            return True
        
        days_since_retraining = (datetime.now() - self.last_retraining).days
        if days_since_retraining > self.config.time_window_days:
            return True
        
        return False
    
    def _get_retraining_reasons(self, metrics: PerformanceMetrics, 
                              degradation: Dict[str, Any],
                              patterns: List[ErrorPattern]) -> List[str]:
        """Get specific reasons for retraining recommendation"""
        reasons = []
        
        if metrics.overall_accuracy < self.config.accuracy_threshold:
            reasons.append(f"Overall accuracy below threshold: {metrics.overall_accuracy:.3f}")
        
        if metrics.critical_miss_rate > (self.config.critical_miss_threshold / 100):
            reasons.append(f"Critical miss rate too high: {metrics.critical_miss_rate:.3f}")
        
        if degradation['degradation_detected']:
            reasons.extend(degradation['issues'])
        
        critical_patterns = [p for p in patterns if p.severity in ["CRITICAL", "HIGH"]]
        if critical_patterns:
            reasons.append(f"Identified {len(critical_patterns)} high-severity error patterns")
        
        return reasons
    
    def _trigger_retraining(self, analysis_report: Dict[str, Any]) -> None:
        """Trigger model retraining process"""
        logger.warning("Triggering model retraining based on performance analysis")
        
        self.retraining_in_progress = True
        
        retraining_record = {
            'timestamp': datetime.now(),
            'trigger_reasons': analysis_report['retraining_reasons'],
            'performance_before': analysis_report['current_metrics'],
            'error_patterns': analysis_report['error_patterns'],
            'status': 'INITIATED'
        }
        
        self.retraining_history.append(retraining_record)
        
        logger.info("Retraining pipeline would be triggered here")
        
        self.last_retraining = datetime.now()
        self.retraining_in_progress = False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        if not self.performance_tracker.current_metrics:
            return {'status': 'No performance data available'}
        
        metrics = self.performance_tracker.current_metrics
        
        return {
            'overall_accuracy': metrics.overall_accuracy,
            'critical_miss_rate': metrics.critical_miss_rate,
            'confidence_calibration_error': metrics.confidence_calibration_error,
            'processing_time_mean': metrics.processing_time_stats['mean'],
            'feedback_entries_collected': len(self.feedback_buffer),
            'last_analysis': self.performance_tracker.performance_history[-1]['timestamp'] if self.performance_tracker.performance_history else None,
            'retraining_status': 'IN_PROGRESS' if self.retraining_in_progress else 'IDLE',
            'days_since_last_retraining': (datetime.now() - self.last_retraining).days
        }
    
    def export_performance_data(self, filepath: str) -> None:
        """Export performance data for external analysis"""
        export_data = {
            'feedback_buffer': [
                {
                    'ecg_id': entry.ecg_id,
                    'ai_prediction': entry.ai_prediction,
                    'expert_diagnosis': entry.expert_diagnosis,
                    'timestamp': entry.timestamp.isoformat(),
                    'discrepancy': entry.discrepancy,
                    'confidence_score': entry.confidence_score,
                    'signal_quality': entry.signal_quality
                }
                for entry in self.feedback_buffer
            ],
            'performance_history': [
                {
                    'timestamp': record['timestamp'].isoformat(),
                    'metrics': {
                        'overall_accuracy': record['metrics'].overall_accuracy,
                        'critical_miss_rate': record['metrics'].critical_miss_rate,
                        'confidence_calibration_error': record['metrics'].confidence_calibration_error
                    },
                    'sample_count': record['sample_count']
                }
                for record in self.performance_tracker.performance_history
            ],
            'error_patterns': [
                {
                    'pattern_type': pattern.pattern_type,
                    'frequency': pattern.frequency,
                    'conditions': pattern.conditions,
                    'severity': pattern.severity,
                    'recommendations': pattern.recommendations,
                    'first_observed': pattern.first_observed.isoformat(),
                    'last_observed': pattern.last_observed.isoformat()
                }
                for pattern in self.error_analyzer.identified_patterns
            ],
            'retraining_history': [
                {
                    'timestamp': record['timestamp'].isoformat(),
                    'trigger_reasons': record['trigger_reasons'],
                    'status': record['status']
                }
                for record in self.retraining_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Performance data exported to {filepath}")


def create_continuous_learning_system(model: Any, **kwargs) -> ContinuousLearningSystem:
    """
    Factory function to create continuous learning system
    
    Args:
        model: The ML model to monitor and improve
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured continuous learning system
    """
    config = RetrainingTrigger(**kwargs)
    return ContinuousLearningSystem(model, config)


if __name__ == "__main__":
    class MockModel:
        def predict_proba(self, X):
            return np.random.random((len(X), 2))
    
    mock_model = MockModel()
    learning_system = create_continuous_learning_system(
        model=mock_model,
        accuracy_threshold=0.85,
        critical_miss_threshold=5
    )
    
    for i in range(50):
        ai_prediction = {
            'atrial_fibrillation': np.random.random(),
            'normal_sinus_rhythm': np.random.random()
        }
        
        expert_diagnosis = {
            'conditions': {
                'atrial_fibrillation': {'detected': np.random.choice([True, False])},
                'normal_sinus_rhythm': {'detected': np.random.choice([True, False])}
            }
        }
        
        learning_system.collect_feedback(
            ecg_id=f"ecg_{i:03d}",
            ai_prediction=ai_prediction,
            expert_diagnosis=expert_diagnosis,
            confidence_score=np.random.random(),
            processing_time=np.random.uniform(1.0, 5.0),
            signal_quality=np.random.uniform(0.5, 1.0)
        )
    
    summary = learning_system.get_performance_summary()
    print("Performance Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    learning_system.export_performance_data("/tmp/performance_data.json")
    print("Performance data exported to /tmp/performance_data.json")
