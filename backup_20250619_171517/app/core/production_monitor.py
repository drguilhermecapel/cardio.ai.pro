import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ProductionMonitor:
    """Monitor de performance em produção com alertas"""

    def __init__(
        self,
        alert_email: str = "alerts@cardioai.pro",
        performance_threshold: float = 0.85,
    ):

        self.alert_email = alert_email
        self.threshold = performance_threshold

        self.prediction_buffer: deque[dict[str, Any]] = deque(maxlen=10000)
        self.quality_buffer: deque[float] = deque(maxlen=1000)
        self.error_buffer: deque[dict[str, Any]] = deque(maxlen=100)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - CardioAI Monitor - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        self.performance_metrics: dict[str, Any] = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
            "quality_score_average": 0.0,
            "last_alert_time": None,
            "alert_cooldown_minutes": 30,
        }

        self.alert_thresholds = {
            "min_confidence": 0.7,
            "max_processing_time": 5.0,  # segundos
            "min_quality_score": 0.6,
            "max_error_rate": 0.1,  # 10%
            "min_success_rate": 0.9,  # 90%
        }

        self.logger.info("ProductionMonitor inicializado com sucesso")

    def log_prediction(
        self,
        prediction_result: dict[str, Any],
        processing_time: float,
        signal_quality: float,
        patient_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """
        Registrar resultado de predição para monitoramento

        Args:
            prediction_result: Resultado da predição do modelo
            processing_time: Tempo de processamento em segundos
            signal_quality: Score de qualidade do sinal (0-1)
            patient_id: ID do paciente (opcional)
            error: Mensagem de erro se houver
        """
        try:
            timestamp = datetime.now()

            prediction_log = {
                "timestamp": timestamp.isoformat(),
                "patient_id": patient_id,
                "processing_time": processing_time,
                "signal_quality": signal_quality,
                "success": error is None,
                "error": error,
            }

            if error is None:
                confidence = prediction_result.get("confidence", 0.0)
                primary_diagnosis = prediction_result.get(
                    "primary_diagnosis", "Unknown"
                )

                prediction_log.update(
                    {
                        "confidence": confidence,
                        "primary_diagnosis": primary_diagnosis,
                        "predictions": prediction_result.get("predictions", {}),
                        "clinical_urgency": str(
                            prediction_result.get("clinical_urgency", "LOW")
                        ),
                    }
                )

                self.prediction_buffer.append(prediction_log)
                self.performance_metrics["successful_predictions"] += 1

                self.quality_buffer.append(signal_quality)

            else:
                self.error_buffer.append(prediction_log)
                self.performance_metrics["failed_predictions"] += 1

                self.logger.error(
                    f"Erro na predição para paciente {patient_id}: {error}"
                )

            self.performance_metrics["total_predictions"] += 1

            if self.performance_metrics["total_predictions"] % 100 == 0:
                self._check_performance()

            self.logger.info(
                f"Predição registrada - Paciente: {patient_id}, "
                f"Sucesso: {error is None}, "
                f"Tempo: {processing_time:.2f}s, "
                f"Qualidade: {signal_quality:.3f}"
            )

        except Exception as e:
            self.logger.error(f"Erro ao registrar predição: {e}")

    def _check_performance(self) -> None:
        """Verificar métricas de performance e enviar alertas se necessário"""
        try:
            if self.performance_metrics["total_predictions"] == 0:
                return

            total_preds = self.performance_metrics["total_predictions"]
            success_rate = (
                self.performance_metrics["successful_predictions"] / total_preds
            )

            error_rate = self.performance_metrics["failed_predictions"] / total_preds

            recent_predictions = (
                list(self.prediction_buffer)[-100:] if self.prediction_buffer else []
            )
            recent_qualities = (
                list(self.quality_buffer)[-100:] if self.quality_buffer else []
            )
            recent_errors = list(self.error_buffer)[-10:] if self.error_buffer else []

            if recent_predictions:
                avg_confidence = np.mean(
                    [
                        p.get("confidence", 0)
                        for p in recent_predictions
                        if p.get("confidence")
                    ]
                )
                avg_processing_time = np.mean(
                    [p.get("processing_time", 0) for p in recent_predictions]
                )
            else:
                avg_confidence = 0.0
                avg_processing_time = 0.0

            if recent_qualities:
                avg_quality = float(np.mean(recent_qualities))
            else:
                avg_quality = 0.0

            self.performance_metrics.update(
                {
                    "average_confidence": avg_confidence,
                    "average_processing_time": avg_processing_time,
                    "quality_score_average": avg_quality,
                }
            )

            alerts = []

            if success_rate < self.alert_thresholds["min_success_rate"]:
                alerts.append(f"Taxa de sucesso baixa: {success_rate:.2%}")

            if error_rate > self.alert_thresholds["max_error_rate"]:
                alerts.append(f"Taxa de erro alta: {error_rate:.2%}")

            if avg_confidence < self.alert_thresholds["min_confidence"]:
                alerts.append(f"Confiança média baixa: {avg_confidence:.3f}")

            if avg_processing_time > self.alert_thresholds["max_processing_time"]:
                alerts.append(
                    f"Tempo de processamento alto: {avg_processing_time:.2f}s"
                )

            if avg_quality < self.alert_thresholds["min_quality_score"]:
                alerts.append(f"Qualidade de sinal baixa: {avg_quality:.3f}")

            if len(recent_errors) >= 5:
                alerts.append(
                    f"Múltiplos erros recentes: {len(recent_errors)} nos últimos registros"
                )

            if alerts:
                self._send_alerts(
                    alerts,
                    {
                        "success_rate": success_rate,
                        "error_rate": error_rate,
                        "avg_confidence": avg_confidence,
                        "avg_processing_time": avg_processing_time,
                        "avg_quality": float(avg_quality),
                        "total_predictions": self.performance_metrics[
                            "total_predictions"
                        ],
                    },
                )

            self.logger.info(
                f"Verificação de performance - "
                f"Taxa sucesso: {success_rate:.2%}, "
                f"Confiança média: {avg_confidence:.3f}, "
                f"Tempo médio: {avg_processing_time:.2f}s, "
                f"Qualidade média: {avg_quality:.3f}"
            )

        except Exception as e:
            self.logger.error(f"Erro na verificação de performance: {e}")

    def _send_alerts(self, alerts: list[str], metrics: dict[str, float]) -> None:
        """Enviar alertas por email e log"""
        try:
            now = datetime.now()
            last_alert = self.performance_metrics["last_alert_time"]
            cooldown_mins = self.performance_metrics["alert_cooldown_minutes"]
            if (
                last_alert
                and isinstance(last_alert, datetime)
                and now - last_alert < timedelta(minutes=cooldown_mins)
            ):
                self.logger.info("Alerta suprimido devido ao cooldown")
                return

            alert_message = (
                "🚨 ALERTA CardioAI Pro - Problemas de Performance Detectados\n\n"
            )
            alert_message += "Problemas identificados:\n"
            for alert in alerts:
                alert_message += f"• {alert}\n"

            alert_message += "\nMétricas atuais:\n"
            alert_message += f"• Total de predições: {metrics['total_predictions']}\n"
            alert_message += f"• Taxa de sucesso: {metrics['success_rate']:.2%}\n"
            alert_message += f"• Taxa de erro: {metrics['error_rate']:.2%}\n"
            alert_message += f"• Confiança média: {metrics['avg_confidence']:.3f}\n"
            alert_message += f"• Tempo médio: {metrics['avg_processing_time']:.2f}s\n"
            alert_message += f"• Qualidade média: {metrics['avg_quality']:.3f}\n"

            alert_message += f"\nTimestamp: {now.isoformat()}\n"
            alert_message += "Sistema: CardioAI Pro Production Monitor\n"

            self.logger.critical(f"ALERTA DE PERFORMANCE: {', '.join(alerts)}")

            try:
                self._send_email_alert(alert_message)
            except Exception as email_error:
                self.logger.error(f"Falha ao enviar email de alerta: {email_error}")

            self.performance_metrics["last_alert_time"] = now

            self.logger.info(
                f"Alerta enviado com {len(alerts)} problemas identificados"
            )

        except Exception as e:
            self.logger.error(f"Erro ao enviar alertas: {e}")

    def _send_email_alert(self, message: str) -> None:
        """Enviar alerta por email (implementação básica)"""
        self.logger.info(f"EMAIL ALERT TO {self.alert_email}:")
        self.logger.info(message)

        #

    def get_performance_report(self) -> dict[str, Any]:
        """Obter relatório completo de performance"""
        try:
            if self.performance_metrics["total_predictions"] == 0:
                return {
                    "status": "No predictions recorded yet",
                    "metrics": self.performance_metrics,
                }

            total_preds = self.performance_metrics["total_predictions"]
            success_rate = (
                self.performance_metrics["successful_predictions"] / total_preds
            )

            error_rate = self.performance_metrics["failed_predictions"] / total_preds

            recent_predictions = (
                list(self.prediction_buffer)[-1000:] if self.prediction_buffer else []
            )
            recent_errors = list(self.error_buffer)[-50:] if self.error_buffer else []

            overall_status = "HEALTHY"
            if (
                success_rate < self.alert_thresholds["min_success_rate"]
                or error_rate > self.alert_thresholds["max_error_rate"]
                or self.performance_metrics["average_confidence"]
                < self.alert_thresholds["min_confidence"]
            ):
                overall_status = "WARNING"

            if success_rate < 0.8 or error_rate > 0.2:
                overall_status = "CRITICAL"

            report = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": overall_status,
                "performance_metrics": {
                    "total_predictions": self.performance_metrics["total_predictions"],
                    "success_rate": success_rate,
                    "error_rate": error_rate,
                    "average_confidence": self.performance_metrics[
                        "average_confidence"
                    ],
                    "average_processing_time": self.performance_metrics[
                        "average_processing_time"
                    ],
                    "average_quality_score": self.performance_metrics[
                        "quality_score_average"
                    ],
                },
                "recent_activity": {
                    "predictions_last_batch": len(recent_predictions),
                    "errors_last_batch": len(recent_errors),
                    "buffer_utilization": {
                        "predictions": f"{len(self.prediction_buffer)}/10000",
                        "quality": f"{len(self.quality_buffer)}/1000",
                        "errors": f"{len(self.error_buffer)}/100",
                    },
                },
                "alert_configuration": self.alert_thresholds,
                "last_alert": (
                    self.performance_metrics["last_alert_time"].isoformat()
                    if isinstance(self.performance_metrics["last_alert_time"], datetime)
                    else None
                ),
            }

            return report

        except Exception as e:
            self.logger.error(f"Erro ao gerar relatório de performance: {e}")
            return {"error": str(e)}

    def reset_metrics(self) -> None:
        """Resetar todas as métricas (usar com cuidado)"""
        self.prediction_buffer.clear()
        self.quality_buffer.clear()
        self.error_buffer.clear()

        self.performance_metrics.update(
            {
                "total_predictions": 0,
                "successful_predictions": 0,
                "failed_predictions": 0,
                "average_confidence": 0.0,
                "average_processing_time": 0.0,
                "quality_score_average": 0.0,
                "last_alert_time": None,
            }
        )

        self.logger.info("Métricas de performance resetadas")

    def update_thresholds(self, new_thresholds: dict[str, float]) -> None:
        """Atualizar thresholds de alerta"""
        try:
            for key, value in new_thresholds.items():
                if key in self.alert_thresholds:
                    old_value = self.alert_thresholds[key]
                    self.alert_thresholds[key] = value
                    self.logger.info(
                        f"Threshold {key} atualizado: {old_value} -> {value}"
                    )
                else:
                    self.logger.warning(f"Threshold desconhecido ignorado: {key}")

            self.logger.info("Thresholds de alerta atualizados")

        except Exception as e:
            self.logger.error(f"Erro ao atualizar thresholds: {e}")
