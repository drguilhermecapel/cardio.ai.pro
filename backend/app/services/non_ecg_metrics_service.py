"""
Non-ECG Metrics Service for tracking response effectiveness and user learning patterns.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from prometheus_client import Counter as PrometheusCounter
from prometheus_client import Gauge, Histogram
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


non_ecg_detections_total = PrometheusCounter(
    'non_ecg_detections_total',
    'Total number of non-ECG images detected',
    ['category', 'confidence_range']
)

user_success_rate = Gauge(
    'user_success_rate_after_guidance',
    'Rate of successful ECG uploads after non-ECG guidance',
    ['time_window']
)

response_effectiveness = Histogram(
    'response_effectiveness_score',
    'Effectiveness score of contextual responses',
    ['category', 'response_type']
)

learning_progression = Gauge(
    'user_learning_progression',
    'User learning progression score',
    ['user_cohort']
)

category_accuracy = Gauge(
    'category_classification_accuracy',
    'Accuracy of non-ECG image classification',
    ['category']
)


@dataclass
class UserLearningMetrics:
    """Metrics for tracking individual user learning patterns."""
    user_id: str
    total_attempts: int
    successful_attempts: int
    non_ecg_attempts: int
    categories_encountered: list[str]
    improvement_rate: float
    last_success_time: datetime | None
    learning_stage: str  # 'beginner', 'intermediate', 'advanced'
    response_helpfulness_scores: list[float]
    time_to_success: float | None  # seconds from first attempt to first success


@dataclass
class SystemMetrics:
    """System-wide metrics for non-ECG detection and response system."""
    total_non_ecg_detections: int
    category_distribution: dict[str, int]
    average_confidence_by_category: dict[str, float]
    response_effectiveness_by_category: dict[str, float]
    user_success_rate_24h: float
    user_success_rate_7d: float
    user_success_rate_30d: float
    average_time_to_success: float
    most_common_categories: list[tuple[str, int]]
    learning_progression_trends: dict[str, float]


class NonECGMetricsService:
    """Service for tracking and analyzing non-ECG image detection metrics."""

    def __init__(self, db_session: AsyncSession | None = None):
        self.db_session = db_session
        self.user_metrics_cache: dict[str, UserLearningMetrics] = {}
        self.system_metrics_cache: SystemMetrics | None = None
        self.cache_expiry = datetime.now()
        self.cache_duration = timedelta(minutes=15)

    async def track_non_ecg_detection(
        self,
        user_id: str,
        category: str,
        confidence: float,
        contextual_response: dict[str, Any],
        session_id: str | None = None
    ) -> None:
        """Track a non-ECG image detection event."""
        try:
            confidence_range = self._get_confidence_range(confidence)
            non_ecg_detections_total.labels(
                category=category,
                confidence_range=confidence_range
            ).inc()

            logger.info(
                "Non-ECG detection tracked",
                extra={
                    "user_id": user_id,
                    "category": category,
                    "confidence": confidence,
                    "session_id": session_id,
                    "response_type": contextual_response.get("response_type", "standard")
                }
            )

            await self._update_user_learning_metrics(
                user_id, category, success=False, confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error tracking non-ECG detection: {str(e)}")

    async def track_successful_ecg_upload(
        self,
        user_id: str,
        confidence: float,
        time_since_last_attempt: float | None = None,
        session_id: str | None = None
    ) -> None:
        """Track a successful ECG upload after previous non-ECG attempts."""
        try:
            await self._update_user_learning_metrics(
                user_id, "ecg_success", success=True, confidence=confidence
            )

            await self._update_success_rate_metrics()

            logger.info(
                "Successful ECG upload tracked",
                extra={
                    "user_id": user_id,
                    "confidence": confidence,
                    "time_since_last_attempt": time_since_last_attempt,
                    "session_id": session_id
                }
            )

        except Exception as e:
            logger.error(f"Error tracking successful ECG upload: {str(e)}")

    async def track_response_feedback(
        self,
        user_id: str,
        category: str,
        helpfulness_score: float,
        feedback_type: str,
        additional_feedback: str | None = None
    ) -> None:
        """Track user feedback on contextual responses."""
        try:
            response_effectiveness.labels(
                category=category,
                response_type=feedback_type
            ).observe(helpfulness_score)

            user_metrics = await self._get_user_metrics(user_id)
            user_metrics.response_helpfulness_scores.append(helpfulness_score)

            if len(user_metrics.response_helpfulness_scores) > 50:
                user_metrics.response_helpfulness_scores = user_metrics.response_helpfulness_scores[-50:]

            logger.info(
                "Response feedback tracked",
                extra={
                    "user_id": user_id,
                    "category": category,
                    "helpfulness_score": helpfulness_score,
                    "feedback_type": feedback_type,
                    "additional_feedback": additional_feedback
                }
            )

        except Exception as e:
            logger.error(f"Error tracking response feedback: {str(e)}")

    async def get_user_learning_metrics(self, user_id: str) -> UserLearningMetrics:
        """Get learning metrics for a specific user."""
        return await self._get_user_metrics(user_id)

    async def get_system_metrics(self, force_refresh: bool = False) -> SystemMetrics:
        """Get system-wide metrics."""
        if (
            force_refresh or
            self.system_metrics_cache is None or
            datetime.now() > self.cache_expiry
        ):
            await self._refresh_system_metrics()

        return self.system_metrics_cache or SystemMetrics(
            total_non_ecg_detections=0,
            category_distribution={},
            average_confidence_by_category={},
            response_effectiveness_by_category={},
            user_success_rate_24h=0.0,
            user_success_rate_7d=0.0,
            user_success_rate_30d=0.0,
            average_time_to_success=0.0,
            most_common_categories=[],
            learning_progression_trends={}
        )

    async def get_category_performance(self) -> dict[str, dict[str, float]]:
        """Get performance metrics by category."""
        try:
            category_stats: dict[str, dict[str, float]] = defaultdict(lambda: {
                'detection_count': 0,
                'average_confidence': 0.0,
                'success_rate_after_guidance': 0.0,
                'average_helpfulness_score': 0.0
            })

            for user_metrics in self.user_metrics_cache.values():
                category_counter: Counter[str] = Counter(user_metrics.categories_encountered)

                for category, count in category_counter.items():
                    category_stats[category]['detection_count'] += count

                if user_metrics.response_helpfulness_scores:
                    avg_helpfulness = sum(user_metrics.response_helpfulness_scores) / len(user_metrics.response_helpfulness_scores)
                    category_stats['overall']['average_helpfulness_score'] = avg_helpfulness

            return dict(category_stats)

        except Exception as e:
            logger.error(f"Error getting category performance: {str(e)}")
            return {}

    async def get_learning_progression_analysis(self) -> dict[str, Any]:
        """Analyze user learning progression patterns."""
        try:
            progression_data = {
                'beginner_users': 0,
                'intermediate_users': 0,
                'advanced_users': 0,
                'average_improvement_rate': 0.0,
                'users_with_positive_progression': 0,
                'average_time_to_first_success': 0.0
            }

            total_improvement = 0.0
            positive_progression_count = 0
            time_to_success_values = []

            for user_metrics in self.user_metrics_cache.values():
                progression_data[f"{user_metrics.learning_stage}_users"] += 1

                total_improvement += user_metrics.improvement_rate
                if user_metrics.improvement_rate > 0:
                    positive_progression_count += 1

                if user_metrics.time_to_success:
                    time_to_success_values.append(user_metrics.time_to_success)

            total_users = len(self.user_metrics_cache)
            if total_users > 0:
                progression_data['average_improvement_rate'] = total_improvement / total_users
                progression_data['users_with_positive_progression'] = positive_progression_count

            if time_to_success_values:
                progression_data['average_time_to_first_success'] = sum(time_to_success_values) / len(time_to_success_values)

            return progression_data

        except Exception as e:
            logger.error(f"Error analyzing learning progression: {str(e)}")
            return {}

    async def generate_insights_report(self) -> dict[str, Any]:
        """Generate comprehensive insights report."""
        try:
            system_metrics = await self.get_system_metrics()
            category_performance = await self.get_category_performance()
            learning_analysis = await self.get_learning_progression_analysis()

            insights = {
                'summary': {
                    'total_non_ecg_detections': system_metrics.total_non_ecg_detections,
                    'overall_success_rate_24h': system_metrics.user_success_rate_24h,
                    'most_common_category': system_metrics.most_common_categories[0] if system_metrics.most_common_categories else None,
                    'average_time_to_success_hours': system_metrics.average_time_to_success / 3600 if system_metrics.average_time_to_success else 0
                },
                'category_insights': self._generate_category_insights(category_performance),
                'learning_insights': self._generate_learning_insights(learning_analysis),
                'recommendations': self._generate_recommendations(system_metrics, category_performance, learning_analysis),
                'generated_at': datetime.now().isoformat()
            }

            return insights

        except Exception as e:
            logger.error(f"Error generating insights report: {str(e)}")
            return {'error': str(e)}


    def _get_confidence_range(self, confidence: float) -> str:
        """Categorize confidence into ranges."""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"

    async def _get_user_metrics(self, user_id: str) -> UserLearningMetrics:
        """Get or create user learning metrics."""
        if user_id not in self.user_metrics_cache:
            self.user_metrics_cache[user_id] = UserLearningMetrics(
                user_id=user_id,
                total_attempts=0,
                successful_attempts=0,
                non_ecg_attempts=0,
                categories_encountered=[],
                improvement_rate=0.0,
                last_success_time=None,
                learning_stage='beginner',
                response_helpfulness_scores=[],
                time_to_success=None
            )

        return self.user_metrics_cache[user_id]

    async def _update_user_learning_metrics(
        self,
        user_id: str,
        category: str,
        success: bool,
        confidence: float
    ) -> None:
        """Update user learning metrics."""
        user_metrics = await self._get_user_metrics(user_id)

        user_metrics.total_attempts += 1

        if success:
            user_metrics.successful_attempts += 1
            user_metrics.last_success_time = datetime.now()

            if user_metrics.time_to_success is None and user_metrics.total_attempts > 1:
                user_metrics.time_to_success = user_metrics.total_attempts * 300  # 5 minutes per attempt estimate
        else:
            user_metrics.non_ecg_attempts += 1
            user_metrics.categories_encountered.append(category)

        if user_metrics.total_attempts > 1:
            user_metrics.improvement_rate = user_metrics.successful_attempts / user_metrics.total_attempts

        user_metrics.learning_stage = self._determine_learning_stage(user_metrics)

        learning_progression.labels(user_cohort=user_metrics.learning_stage).set(user_metrics.improvement_rate)

    def _determine_learning_stage(self, user_metrics: UserLearningMetrics) -> str:
        """Determine user learning stage based on metrics."""
        if user_metrics.total_attempts < 3:
            return 'beginner'
        elif user_metrics.improvement_rate > 0.7:
            return 'advanced'
        elif user_metrics.improvement_rate > 0.3:
            return 'intermediate'
        else:
            return 'beginner'

    async def _update_success_rate_metrics(self) -> None:
        """Update success rate Prometheus metrics."""
        try:
            now = datetime.now()

            success_24h = await self._calculate_success_rate(now - timedelta(hours=24))
            user_success_rate.labels(time_window='24h').set(success_24h)

            success_7d = await self._calculate_success_rate(now - timedelta(days=7))
            user_success_rate.labels(time_window='7d').set(success_7d)

            success_30d = await self._calculate_success_rate(now - timedelta(days=30))
            user_success_rate.labels(time_window='30d').set(success_30d)

        except Exception as e:
            logger.error(f"Error updating success rate metrics: {str(e)}")

    async def _calculate_success_rate(self, since: datetime) -> float:
        """Calculate success rate since a given time."""
        try:
            total_users_with_attempts = 0
            users_with_success = 0

            for user_metrics in self.user_metrics_cache.values():
                if user_metrics.last_success_time and user_metrics.last_success_time >= since:
                    total_users_with_attempts += 1
                    if user_metrics.successful_attempts > 0:
                        users_with_success += 1

            if total_users_with_attempts == 0:
                return 0.0

            return users_with_success / total_users_with_attempts

        except Exception as e:
            logger.error(f"Error calculating success rate: {str(e)}")
            return 0.0

    async def _refresh_system_metrics(self) -> None:
        """Refresh system-wide metrics cache."""
        try:
            total_detections = sum(
                user_metrics.non_ecg_attempts
                for user_metrics in self.user_metrics_cache.values()
            )

            category_counter: Counter[str] = Counter()
            for user_metrics in self.user_metrics_cache.values():
                category_counter.update(user_metrics.categories_encountered)

            now = datetime.now()
            success_24h = await self._calculate_success_rate(now - timedelta(hours=24))
            success_7d = await self._calculate_success_rate(now - timedelta(days=7))
            success_30d = await self._calculate_success_rate(now - timedelta(days=30))

            time_to_success_values = [
                user_metrics.time_to_success
                for user_metrics in self.user_metrics_cache.values()
                if user_metrics.time_to_success is not None
            ]
            avg_time_to_success = sum(time_to_success_values) / len(time_to_success_values) if time_to_success_values else 0.0

            self.system_metrics_cache = SystemMetrics(
                total_non_ecg_detections=total_detections,
                category_distribution=dict(category_counter),
                average_confidence_by_category={},  # Would need more detailed tracking
                response_effectiveness_by_category={},  # Would need more detailed tracking
                user_success_rate_24h=success_24h,
                user_success_rate_7d=success_7d,
                user_success_rate_30d=success_30d,
                average_time_to_success=avg_time_to_success,
                most_common_categories=category_counter.most_common(5),
                learning_progression_trends={}  # Would need historical data
            )

            self.cache_expiry = datetime.now() + self.cache_duration

        except Exception as e:
            logger.error(f"Error refreshing system metrics: {str(e)}")

    def _generate_category_insights(self, category_performance: dict[str, dict[str, float]]) -> list[str]:
        """Generate insights about category performance."""
        insights = []

        try:
            sorted_categories = sorted(
                category_performance.items(),
                key=lambda x: x[1].get('success_rate_after_guidance', 0)
            )

            if sorted_categories:
                worst_category = sorted_categories[0]
                insights.append(f"Category '{worst_category[0]}' has the lowest success rate after guidance")

            for category, stats in category_performance.items():
                if stats.get('detection_count', 0) > 10 and stats.get('success_rate_after_guidance', 0) < 0.5:
                    insights.append(f"Category '{category}' needs improved guidance - high detection but low success rate")

        except Exception as e:
            logger.error(f"Error generating category insights: {str(e)}")

        return insights

    def _generate_learning_insights(self, learning_analysis: dict[str, Any]) -> list[str]:
        """Generate insights about user learning patterns."""
        insights = []

        try:
            total_users = (
                learning_analysis.get('beginner_users', 0) +
                learning_analysis.get('intermediate_users', 0) +
                learning_analysis.get('advanced_users', 0)
            )

            if total_users > 0:
                beginner_ratio = learning_analysis.get('beginner_users', 0) / total_users
                if beginner_ratio > 0.7:
                    insights.append("High proportion of beginner users - consider improving onboarding")

                avg_improvement = learning_analysis.get('average_improvement_rate', 0)
                if avg_improvement < 0.3:
                    insights.append("Low average improvement rate - guidance effectiveness needs improvement")

                avg_time_to_success = learning_analysis.get('average_time_to_first_success', 0)
                if avg_time_to_success > 1800:  # 30 minutes
                    insights.append("Users take too long to achieve first success - streamline guidance process")

        except Exception as e:
            logger.error(f"Error generating learning insights: {str(e)}")

        return insights

    def _generate_recommendations(
        self,
        system_metrics: SystemMetrics,
        category_performance: dict[str, dict[str, float]],
        learning_analysis: dict[str, Any]
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        try:
            if system_metrics.user_success_rate_24h < 0.6:
                recommendations.append("Improve contextual responses - 24h success rate is below 60%")

            for category, stats in category_performance.items():
                if stats.get('detection_count', 0) > 5 and stats.get('average_helpfulness_score', 0) < 3.0:
                    recommendations.append(f"Enhance responses for '{category}' category - low helpfulness scores")

            beginner_ratio = learning_analysis.get('beginner_users', 0) / max(1, sum([
                learning_analysis.get('beginner_users', 0),
                learning_analysis.get('intermediate_users', 0),
                learning_analysis.get('advanced_users', 0)
            ]))

            if beginner_ratio > 0.8:
                recommendations.append("Create more comprehensive onboarding tutorials")

            if learning_analysis.get('average_time_to_first_success', 0) > 1200:  # 20 minutes
                recommendations.append("Simplify initial guidance to reduce time to first success")

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")

        return recommendations


non_ecg_metrics_service = NonECGMetricsService()
