"""
Adaptive Feedback Service for Non-ECG Image System
Tracks user attempts and provides personalized responses based on behavior patterns.
"""

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.db.session import get_session_factory
from app.models.user import User, UserSession

logger = logging.getLogger(__name__)


class UserFeedbackMetrics:
    """Data class for user feedback metrics."""

    def __init__(self, user_id: str = "", total_attempts: int = 0, successful_attempts: int = 0,
                 learning_stage: str = "beginner", feedback_scores: list[float] | None = None,
                 last_attempt_timestamp: datetime | None = None,
                 preferred_response_style: str = "educational",
                 category_history: list[str] | None = None,
                 learning_progress: float = 0.0,
                 last_attempt_time: datetime | None = None) -> None:
        self.user_id = user_id
        self.total_attempts: int = total_attempts
        self.successful_attempts: int = successful_attempts
        self.category_history: list[str] = category_history or []
        self.last_attempt_time: datetime | None = last_attempt_timestamp or last_attempt_time
        self.learning_progress: float = learning_progress
        self.response_effectiveness: dict[str, float] = {}
        self.preferred_response_style: str = preferred_response_style
        self.needs_extra_guidance: bool = False
        self.learning_stage: str = learning_stage
        self.feedback_scores: list[float] = feedback_scores or []
        self.feedback_history: list[dict[str, Any]] = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate based on attempts."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts


class AdaptiveFeedbackService:
    """Service for tracking user attempts and providing personalized responses."""

    def __init__(self) -> None:
        """Initialize the adaptive feedback service."""
        self.session_factory = get_session_factory()
        self.user_metrics: dict[str, UserFeedbackMetrics] = {}
        self.category_patterns = {
            'medical_confusion': ['medical_other', 'xray', 'prescription', 'lab_results'],
            'photo_uploads': ['photo_person', 'nature', 'object', 'food'],
            'document_uploads': ['document', 'text_document', 'handwritten'],
            'screen_captures': ['screenshot', 'monitor_screen']
        }

    async def track_user_attempt(
        self,
        user_session: Any,
        category: str,
        confidence: float,
        success: bool = False,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Track user attempt and update feedback metrics."""
        try:
            user_id = user_id or user_session.id
            if user_id not in self.user_metrics:
                self.user_metrics[user_id] = UserFeedbackMetrics()

            metrics = self.user_metrics[user_id]

            metrics.total_attempts += 1
            if success:
                metrics.successful_attempts += 1
            metrics.category_history.append(category)

            self.user_metrics[user_id] = metrics
            metrics.last_attempt_time = datetime.utcnow()

            if len(metrics.category_history) > 50:
                metrics.category_history = metrics.category_history[-50:]

            await self._update_learning_progress(metrics)
            await self._analyze_user_patterns(metrics)

            logger.info(f"Tracked attempt for user {user_id}: category={category}, confidence={confidence}")

        except Exception as e:
            logger.error(f"Error tracking user attempt: {str(e)}")

    def get_personalized_response(
        self,
        user_session: Any,
        category: str,
        base_message: str
    ) -> dict[str, Any]:
        """Generate personalized response based on user history."""
        try:
            user_id = user_session.id if hasattr(user_session, 'id') else str(user_session)

            if user_id not in self.user_metrics:
                self.user_metrics[user_id] = UserFeedbackMetrics(
                    user_id=user_id,
                    total_attempts=0,
                    successful_attempts=0,
                    category_history=[],
                    learning_progress=0.0,
                    last_attempt_time=datetime.now()
                )

            metrics = self.user_metrics[user_id]
            learning_stage = self._determine_learning_stage(metrics)

            suggestions = ["Dica: ECGs sempre têm uma grade milimetrada de fundo e ondas que representam batimentos cardíacos."]

            learning_tips = [
                "ECGs são documentos médicos, não fotos pessoais. Procure por papel com grade milimetrada.",
                "ECGs têm características visuais únicas: grade milimetrada e ondas cardíacas.",
                "Procure por documentos que mostrem múltiplas 'derivações' (I, II, III, etc.)."
            ]

            personalized_response = {
                'message': base_message,
                'adaptive_suggestions': suggestions,
                'learning_tips': learning_tips,
                'encouragement': "Continue tentando! Estamos aqui para ajudar.",
                'learning_stage': learning_stage
            }

            return personalized_response

        except Exception as e:
            logger.error(f"Error generating personalized response: {str(e)}")
            return {
                'message': base_message,
                'adaptive_suggestions': [],
                'learning_tips': [],
                'encouragement': "Continue tentando!",
                'learning_stage': 'beginner'
            }

    async def collect_feedback(
        self,
        user_session: Any,
        category: str,
        helpfulness_score: float,
        feedback_text: str | None = None,
        user_id: str | None = None,
        feedback_type: str = "general",
        rating: int | None = None,
        response_helpful: bool = True,
        comments: str | None = None
    ) -> None:
        """Collect user feedback to improve future responses."""
        try:
            user_id = user_id or user_session.id
            if user_id not in self.user_metrics:
                self.user_metrics[user_id] = UserFeedbackMetrics()

            metrics = self.user_metrics[user_id]

            metrics.feedback_scores.append(helpfulness_score)

            if not hasattr(metrics, 'feedback_history'):
                metrics.feedback_history = []

            feedback_data = {
                'helpfulness_score': helpfulness_score,
                'feedback_text': feedback_text,
                'category': category,
                'helpful': response_helpful,
                'comments': comments,
                'timestamp': datetime.utcnow().isoformat()
            }
            metrics.feedback_history.append(feedback_data)

            if len(metrics.feedback_history) > 20:
                metrics.feedback_history = metrics.feedback_history[-20:]

            logger.info(f"Collected feedback for user {user_id}: helpfulness_score={helpfulness_score}")

        except Exception as e:
            logger.error(f"Error collecting feedback: {str(e)}")

    async def get_user_learning_stats(self, user_session: UserSession) -> dict[str, Any]:
        """Get user learning statistics and progress."""
        try:
            user_id = str(user_session.user_id)

            if user_id not in self.user_metrics:
                self.user_metrics[user_id] = await self._load_user_metrics(user_id)

            metrics = self.user_metrics[user_id]

            success_rate = (
                metrics.successful_attempts / metrics.total_attempts
                if metrics.total_attempts > 0 else 0.0
            )

            category_distribution = await self._analyze_category_distribution(metrics.category_history)

            learning_stage = self._determine_learning_stage(metrics)

            return {
                'total_attempts': metrics.total_attempts,
                'successful_attempts': metrics.successful_attempts,
                'success_rate': success_rate,
                'learning_progress': metrics.learning_progress,
                'learning_stage': learning_stage,
                'category_distribution': category_distribution,
                'needs_extra_guidance': metrics.needs_extra_guidance,
                'preferred_style': metrics.preferred_response_style,
                'last_attempt': metrics.last_attempt_time.isoformat() if metrics.last_attempt_time else None
            }

        except Exception as e:
            logger.error(f"Error getting user learning stats: {str(e)}")
            return {}

    async def _load_user_metrics(self, user_id: str) -> UserFeedbackMetrics:
        """Load user metrics from database or create new ones."""
        try:
            async with self.session_factory() as session:
                stmt = select(User).where(User.id == int(user_id))
                result = await session.execute(stmt)
                user = result.scalar_one_or_none()

                if user and user.ui_preferences:
                    try:
                        preferences = json.loads(user.ui_preferences)
                        feedback_data = preferences.get('feedback_metrics', {})

                        metrics = UserFeedbackMetrics()
                        metrics.total_attempts = feedback_data.get('total_attempts', 0)
                        metrics.successful_attempts = feedback_data.get('successful_attempts', 0)
                        metrics.category_history = feedback_data.get('category_history', [])
                        metrics.learning_progress = feedback_data.get('learning_progress', 0.0)
                        metrics.response_effectiveness = feedback_data.get('response_effectiveness', {})
                        metrics.preferred_response_style = feedback_data.get('preferred_style', 'professional')
                        metrics.needs_extra_guidance = feedback_data.get('needs_extra_guidance', False)

                        if feedback_data.get('last_attempt_time'):
                            metrics.last_attempt_time = datetime.fromisoformat(feedback_data['last_attempt_time'])

                        return metrics
                    except (json.JSONDecodeError, KeyError):
                        pass

                return UserFeedbackMetrics()

        except Exception as e:
            logger.error(f"Error loading user metrics: {str(e)}")
            return UserFeedbackMetrics()

    async def _save_user_metrics(self, user_id: str, metrics: UserFeedbackMetrics) -> None:
        """Save user metrics to database."""
        try:
            async with self.session_factory() as session:
                stmt = select(User).where(User.id == int(user_id))
                result = await session.execute(stmt)
                user = result.scalar_one_or_none()

                if user:
                    feedback_data = {
                        'total_attempts': metrics.total_attempts,
                        'successful_attempts': metrics.successful_attempts,
                        'category_history': metrics.category_history,
                        'learning_progress': metrics.learning_progress,
                        'response_effectiveness': metrics.response_effectiveness,
                        'preferred_style': metrics.preferred_response_style,
                        'needs_extra_guidance': metrics.needs_extra_guidance,
                        'last_attempt_time': metrics.last_attempt_time.isoformat() if metrics.last_attempt_time else None
                    }

                    current_preferences = {}
                    if user.ui_preferences:
                        try:
                            current_preferences = json.loads(user.ui_preferences)
                        except json.JSONDecodeError:
                            pass

                    current_preferences['feedback_metrics'] = feedback_data
                    user.ui_preferences = json.dumps(current_preferences)

                    await session.commit()

        except Exception as e:
            logger.error(f"Error saving user metrics: {str(e)}")

    async def _update_learning_progress(self, metrics: UserFeedbackMetrics) -> None:
        """Update user learning progress based on attempts and success."""
        if metrics.total_attempts == 0:
            metrics.learning_progress = 0.0
            return

        success_rate = metrics.successful_attempts / metrics.total_attempts
        attempt_factor = min(metrics.total_attempts / 10, 1.0)  # Max factor at 10 attempts

        recent_history = metrics.category_history[-10:] if len(metrics.category_history) >= 10 else metrics.category_history
        recent_success_indicators = sum(1 for cat in recent_history if cat == 'ecg_success')
        recent_success_rate = recent_success_indicators / len(recent_history) if recent_history else 0

        metrics.learning_progress = (
            0.4 * success_rate +
            0.3 * attempt_factor +
            0.3 * recent_success_rate
        )

        metrics.learning_progress = min(metrics.learning_progress, 1.0)

    async def _analyze_user_patterns(self, metrics: UserFeedbackMetrics) -> None:
        """Analyze user patterns to determine if extra guidance is needed."""
        if len(metrics.category_history) < 3:
            return

        recent_attempts = metrics.category_history[-5:]  # Last 5 attempts

        pattern_indicators = {
            'medical_confusion': sum(1 for cat in recent_attempts if cat in self.category_patterns['medical_confusion']),
            'photo_uploads': sum(1 for cat in recent_attempts if cat in self.category_patterns['photo_uploads']),
            'document_uploads': sum(1 for cat in recent_attempts if cat in self.category_patterns['document_uploads']),
            'screen_captures': sum(1 for cat in recent_attempts if cat in self.category_patterns['screen_captures'])
        }

        max_pattern_count = max(pattern_indicators.values())
        metrics.needs_extra_guidance = max_pattern_count >= 3

        if pattern_indicators['medical_confusion'] >= 2:
            metrics.preferred_response_style = "technical"
        elif pattern_indicators['photo_uploads'] >= 2:
            metrics.preferred_response_style = "friendly"
        else:
            metrics.preferred_response_style = "professional"

    async def _generate_personalization(self, metrics: UserFeedbackMetrics, category: str) -> dict[str, Any]:
        """Generate personalized elements for response."""
        personalization = {}

        if metrics.total_attempts == 1:
            personalization['first_time_message'] = "Bem-vindo! Vamos ajudá-lo a entender como enviar ECGs corretamente."
        elif metrics.total_attempts > 5 and metrics.successful_attempts == 0:
            personalization['struggling_message'] = "Percebemos que você está tentando há um tempo. Vamos dar dicas mais específicas!"

        category_count = metrics.category_history.count(category)
        if category_count > 2:
            personalization['repeated_category'] = f"Notamos que você já enviou {category_count} imagens deste tipo. Vamos focar em como identificar ECGs."

        if metrics.learning_progress > 0.7:
            personalization['progress_message'] = "Você está progredindo bem! Continue assim."
        elif metrics.learning_progress < 0.3 and metrics.total_attempts > 3:
            personalization['encouragement_message'] = "Não desista! Aprender a identificar ECGs leva tempo."

        return personalization

    async def _get_adaptive_suggestions(self, metrics: UserFeedbackMetrics, category: str) -> list[str]:
        """Get adaptive suggestions based on user history."""
        suggestions = []

        recent_categories = metrics.category_history[-5:]

        if 'photo_person' in recent_categories and 'nature' in recent_categories:
            suggestions.append("Vemos que você está enviando fotos pessoais. ECGs são documentos médicos específicos com ondas cardíacas.")

        if recent_categories.count('medical_other') > 1:
            suggestions.append("Você tem documentos médicos! Procure especificamente por um que tenha ondas cardíacas em papel milimetrado.")

        if 'screenshot' in recent_categories:
            suggestions.append("Para melhor análise, use o arquivo original do ECG ou uma foto direta do documento.")

        if metrics.learning_progress < 0.3:
            suggestions.append("Dica: ECGs sempre têm uma grade milimetrada de fundo e ondas que representam batimentos cardíacos.")
        elif metrics.learning_progress < 0.7:
            suggestions.append("Você está aprendendo! Lembre-se: ECGs mostram 12 derivações diferentes do coração.")

        if metrics.total_attempts > 10 and metrics.successful_attempts == 0:
            suggestions.append("Que tal consultar um profissional de saúde para obter um ECG real para análise?")

        return suggestions

    async def _generate_encouragement(self, metrics: UserFeedbackMetrics) -> str:
        """Generate encouragement message based on user progress."""
        if metrics.total_attempts == 1:
            return "Primeira tentativa! Vamos aprender juntos sobre ECGs."

        if metrics.successful_attempts > 0:
            return f"Ótimo! Você já teve {metrics.successful_attempts} análises bem-sucedidas."

        if metrics.learning_progress > 0.5:
            return "Você está no caminho certo! Continue praticando."

        if metrics.total_attempts > 5:
            return "Persistência é fundamental! Cada tentativa nos ajuda a aprender."

        return "Continue tentando! Estamos aqui para ajudar."

    async def _get_learning_tips(self, metrics: UserFeedbackMetrics, category: str) -> list[str]:
        """Get learning tips based on user's current understanding."""
        tips = []

        if category in self.category_patterns['medical_confusion']:
            tips.append("ECGs são diferentes de outros exames médicos - eles mostram especificamente a atividade elétrica do coração.")

        if category in self.category_patterns['photo_uploads']:
            tips.append("ECGs são documentos médicos, não fotos pessoais. Procure por papel com grade milimetrada.")

        if metrics.learning_progress < 0.3:
            tips.extend([
                "ECGs têm características visuais únicas: grade milimetrada e ondas cardíacas.",
                "Procure por documentos que mostrem múltiplas 'derivações' (I, II, III, etc.)."
            ])
        elif metrics.learning_progress < 0.7:
            tips.extend([
                "ECGs padrão mostram 12 derivações diferentes do coração.",
                "A qualidade da imagem é importante para análise precisa."
            ])

        return tips

    async def _analyze_feedback(self, user_response: str) -> float:
        """Analyze user feedback and return effectiveness score."""
        positive_keywords = ['útil', 'ajudou', 'entendi', 'obrigado', 'bom', 'claro', 'sim']
        negative_keywords = ['confuso', 'não entendi', 'difícil', 'não ajudou', 'ruim', 'não']

        user_response_lower = user_response.lower()

        positive_count = sum(1 for word in positive_keywords if word in user_response_lower)
        negative_count = sum(1 for word in negative_keywords if word in user_response_lower)

        if positive_count > negative_count:
            return 0.8
        elif negative_count > positive_count:
            return 0.2
        else:
            return 0.5  # Neutral

    async def _adjust_response_style(self, metrics: UserFeedbackMetrics, user_response: str, feedback_score: float) -> None:
        """Adjust response style based on user feedback."""
        if feedback_score < 0.4:  # Negative feedback
            if metrics.preferred_response_style == "professional":
                metrics.preferred_response_style = "friendly"
            elif metrics.preferred_response_style == "friendly":
                metrics.preferred_response_style = "technical"
            else:
                metrics.preferred_response_style = "professional"

    async def _analyze_category_distribution(self, category_history: list[str]) -> dict[str, int]:
        """Analyze distribution of categories in user history."""
        distribution: dict[str, int] = {}
        for category in category_history:
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

    def _determine_learning_stage(self, metrics: UserFeedbackMetrics) -> str:
        """Determine user's learning stage."""
        if metrics.total_attempts == 0:
            return "new_user"

        # success_rate is a property, no need to set it manually

        if metrics.learning_progress < 0.3:
            return "beginner"
        elif metrics.learning_progress < 0.7:
            return "intermediate"
        else:
            return "advanced"

    async def get_system_metrics(self) -> dict[str, Any]:
        """Get system-wide metrics for monitoring."""
        try:
            total_users = len(self.user_metrics)

            if total_users == 0:
                return {
                    'total_users': 0,
                    'average_success_rate': 0.0,
                    'average_learning_progress': 0.0,
                    'users_needing_guidance': 0
                }

            total_attempts = sum(m.total_attempts for m in self.user_metrics.values())
            total_successes = sum(m.successful_attempts for m in self.user_metrics.values())
            average_success_rate = total_successes / total_attempts if total_attempts > 0 else 0.0

            average_learning_progress = sum(m.learning_progress for m in self.user_metrics.values()) / total_users
            users_needing_guidance = sum(1 for m in self.user_metrics.values() if m.needs_extra_guidance)

            all_categories = []
            for metrics in self.user_metrics.values():
                all_categories.extend(metrics.category_history)

            category_distribution = await self._analyze_category_distribution(all_categories)

            return {
                'total_users': total_users,
                'total_attempts': total_attempts,
                'total_successes': total_successes,
                'average_success_rate': average_success_rate,
                'average_learning_progress': average_learning_progress,
                'users_needing_guidance': users_needing_guidance,
                'category_distribution': category_distribution
            }

        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}


adaptive_feedback_service = AdaptiveFeedbackService()
