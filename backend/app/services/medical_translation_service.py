import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    INFORMATIONAL = "informational"

class MedicalTerminologyValidator:
    """Validates medical terminology translations for accuracy and consistency."""

    def __init__(self) -> None:
        self.critical_terms: dict[str, dict[str, str]] = {}
        self.validated_terms: dict[str, dict[str, dict[str, Any]]] = {}
        self.severity_mapping: dict[str, ValidationSeverity] = {}
        self.load_critical_terms()
        self.load_validated_terms()
        self._initialize_critical_medical_terms()

    def load_critical_terms(self) -> None:
        """Load critical medical terms that require professional validation."""
        terms_path = Path(__file__).parent.parent / "translations" / "medical_terms.json"
        if terms_path.exists():
            try:
                with open(terms_path, encoding='utf-8') as f:
                    self.critical_terms = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load critical terms: {e}")
                self.critical_terms = {}

    def load_validated_terms(self) -> None:
        """Load previously validated terms."""
        validated_path = Path(__file__).parent.parent / "translations" / "validated_terms.json"
        if validated_path.exists():
            try:
                with open(validated_path, encoding='utf-8') as f:
                    self.validated_terms = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load validated terms: {e}")
                self.validated_terms = {}

    def _initialize_critical_medical_terms(self) -> None:
        """Initialize critical cardiac/ECG terminology that requires validation."""
        critical_cardiac_terms = {
            "atrial_fibrillation": {
                "severity": ValidationSeverity.CRITICAL,
                "en": "Atrial Fibrillation",
                "pt": "Fibrilação Atrial",
                "es": "Fibrilación Auricular",
                "fr": "Fibrillation Auriculaire",
                "de": "Vorhofflimmern",
                "zh": "心房颤动",
                "ar": "الرجفان الأذيني"
            },
            "ventricular_tachycardia": {
                "severity": ValidationSeverity.CRITICAL,
                "en": "Ventricular Tachycardia",
                "pt": "Taquicardia Ventricular",
                "es": "Taquicardia Ventricular",
                "fr": "Tachycardie Ventriculaire",
                "de": "Ventrikuläre Tachykardie",
                "zh": "室性心动过速",
                "ar": "تسرع القلب البطيني"
            },
            "ventricular_fibrillation": {
                "severity": ValidationSeverity.CRITICAL,
                "en": "Ventricular Fibrillation",
                "pt": "Fibrilação Ventricular",
                "es": "Fibrilación Ventricular",
                "fr": "Fibrillation Ventriculaire",
                "de": "Kammerflimmern",
                "zh": "心室颤动",
                "ar": "الرجفان البطيني"
            },
            "asystole": {
                "severity": ValidationSeverity.CRITICAL,
                "en": "Asystole",
                "pt": "Assistolia",
                "es": "Asistolia",
                "fr": "Asystolie",
                "de": "Asystolie",
                "zh": "心脏停搏",
                "ar": "انقطاع النظم"
            },

            "p_wave": {
                "severity": ValidationSeverity.IMPORTANT,
                "en": "P Wave",
                "pt": "Onda P",
                "es": "Onda P",
                "fr": "Onde P",
                "de": "P-Welle",
                "zh": "P波",
                "ar": "موجة P"
            },
            "qrs_complex": {
                "severity": ValidationSeverity.IMPORTANT,
                "en": "QRS Complex",
                "pt": "Complexo QRS",
                "es": "Complejo QRS",
                "fr": "Complexe QRS",
                "de": "QRS-Komplex",
                "zh": "QRS波群",
                "ar": "مركب QRS"
            },
            "t_wave": {
                "severity": ValidationSeverity.IMPORTANT,
                "en": "T Wave",
                "pt": "Onda T",
                "es": "Onda T",
                "fr": "Onde T",
                "de": "T-Welle",
                "zh": "T波",
                "ar": "موجة T"
            },
            "st_segment": {
                "severity": ValidationSeverity.IMPORTANT,
                "en": "ST Segment",
                "pt": "Segmento ST",
                "es": "Segmento ST",
                "fr": "Segment ST",
                "de": "ST-Strecke",
                "zh": "ST段",
                "ar": "قطعة ST"
            },

            "myocardial_infarction": {
                "severity": ValidationSeverity.CRITICAL,
                "en": "Myocardial Infarction",
                "pt": "Infarto do Miocárdio",
                "es": "Infarto de Miocardio",
                "fr": "Infarctus du Myocarde",
                "de": "Myokardinfarkt",
                "zh": "心肌梗死",
                "ar": "احتشاء عضلة القلب"
            },
            "cardiac_arrest": {
                "severity": ValidationSeverity.CRITICAL,
                "en": "Cardiac Arrest",
                "pt": "Parada Cardíaca",
                "es": "Paro Cardíaco",
                "fr": "Arrêt Cardiaque",
                "de": "Herzstillstand",
                "zh": "心脏骤停",
                "ar": "السكتة القلبية"
            },

            "heart_rate": {
                "severity": ValidationSeverity.IMPORTANT,
                "en": "Heart Rate",
                "pt": "Frequência Cardíaca",
                "es": "Frecuencia Cardíaca",
                "fr": "Fréquence Cardiaque",
                "de": "Herzfrequenz",
                "zh": "心率",
                "ar": "معدل ضربات القلب"
            },
            "pr_interval": {
                "severity": ValidationSeverity.IMPORTANT,
                "en": "PR Interval",
                "pt": "Intervalo PR",
                "es": "Intervalo PR",
                "fr": "Intervalle PR",
                "de": "PR-Intervall",
                "zh": "PR间期",
                "ar": "فترة PR"
            },
            "qt_interval": {
                "severity": ValidationSeverity.IMPORTANT,
                "en": "QT Interval",
                "pt": "Intervalo QT",
                "es": "Intervalo QT",
                "fr": "Intervalle QT",
                "de": "QT-Intervall",
                "zh": "QT间期",
                "ar": "فترة QT"
            }
        }

        for term_key, term_data in critical_cardiac_terms.items():
            severity_obj = term_data.pop("severity")
            if isinstance(severity_obj, ValidationSeverity):
                severity = severity_obj
            else:
                severity = ValidationSeverity.INFORMATIONAL
            self.severity_mapping[term_key] = severity
            term_translations: dict[str, str] = {}
            for k, v in term_data.items():
                if k != "severity" and isinstance(v, str):
                    term_translations[k] = v
            self.critical_terms[term_key] = term_translations

    def validate_term(self, term: str, translation: str, language: str) -> tuple[bool, str, ValidationSeverity]:
        """
        Validate a medical term translation.

        Args:
            term: The original medical term key
            translation: The translated term
            language: The target language code

        Returns:
            Tuple of (is_valid, message, severity)
        """
        severity = self.severity_mapping.get(term, ValidationSeverity.INFORMATIONAL)

        if term in self.critical_terms:
            if language in self.critical_terms[term]:
                expected = self.critical_terms[term][language]
                if translation != expected:
                    return False, f"Critical medical term '{term}' has incorrect translation. Expected: '{expected}'", severity
                return True, "Validated critical term", severity

            return False, f"Critical medical term '{term}' requires professional validation for {language}", severity

        if len(translation) < 2:
            return False, "Translation too short", severity

        if translation.isdigit():
            return False, "Translation cannot be only numbers", severity

        dangerous_patterns = ["error", "null", "undefined", "test"]
        if any(pattern in translation.lower() for pattern in dangerous_patterns):
            return False, "Translation contains potentially dangerous pattern", severity

        return True, "Non-critical term, requires review", severity

    def register_professional_validation(self, term: str, translation: str, language: str,
                                        validator_id: str, validator_credentials: str) -> None:
        """Register that a medical professional has validated this translation."""
        if term not in self.critical_terms:
            self.critical_terms[term] = {}

        self.critical_terms[term][language] = translation

        if term not in self.validated_terms:
            self.validated_terms[term] = {}

        self.validated_terms[term][language] = {
            "validated": True,
            "validator_id": validator_id,
            "validator_credentials": validator_credentials,
            "timestamp": datetime.now().isoformat(),
            "translation": translation
        }

        self._save_critical_terms()
        self._save_validated_terms()

        logger.info(f"Professional validation registered for term '{term}' in {language} by {validator_id}")

    def get_validation_status(self, term: str, language: str) -> dict[str, Any]:
        """Get the validation status for a specific term and language."""
        status = {
            "term": term,
            "language": language,
            "is_critical": term in self.critical_terms,
            "has_validated_translation": False,
            "severity": self.severity_mapping.get(term, ValidationSeverity.INFORMATIONAL).value,
            "validated_translation": None,
            "validation_info": None
        }

        if term in self.critical_terms and language in self.critical_terms[term]:
            status["has_validated_translation"] = True
            status["validated_translation"] = self.critical_terms[term][language]

            if term in self.validated_terms and language in self.validated_terms[term]:
                status["validation_info"] = self.validated_terms[term][language]

        return status

    def get_pending_validations(self, language: str) -> list[dict[str, Any]]:
        """Get list of critical terms that need professional validation for a language."""
        pending = []

        for term in self.critical_terms:
            if language not in self.critical_terms[term]:
                pending.append({
                    "term": term,
                    "language": language,
                    "severity": self.severity_mapping.get(term, ValidationSeverity.INFORMATIONAL).value,
                    "english_term": self.critical_terms[term].get("en", term)
                })

        return pending

    def validate_translation_batch(self, translations: dict[str, str], language: str) -> dict[str, dict[str, Any]]:
        """Validate a batch of translations."""
        results = {}

        for term, translation in translations.items():
            is_valid, message, severity = self.validate_term(term, translation, language)
            results[term] = {
                "is_valid": is_valid,
                "message": message,
                "severity": severity.value,
                "translation": translation
            }

        return results

    def _save_critical_terms(self) -> None:
        """Save the updated critical terms to disk."""
        terms_path = Path(__file__).parent.parent / "translations" / "medical_terms.json"
        terms_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(terms_path, 'w', encoding='utf-8') as f:
                json.dump(self.critical_terms, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"Failed to save critical terms: {e}")

    def _save_validated_terms(self) -> None:
        """Save the validated terms to disk."""
        validated_path = Path(__file__).parent.parent / "translations" / "validated_terms.json"
        validated_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(validated_path, 'w', encoding='utf-8') as f:
                json.dump(self.validated_terms, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"Failed to save validated terms: {e}")

    def export_validation_report(self, language: str) -> dict[str, Any]:
        """Export a comprehensive validation report for a language."""
        report = {
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_critical_terms": len(self.critical_terms),
                "validated_terms": 0,
                "pending_validations": 0,
                "critical_pending": 0,
                "important_pending": 0
            },
            "validated_terms": [],
            "pending_validations": [],
            "validation_coverage": 0.0
        }

        for term in self.critical_terms:
            status = self.get_validation_status(term, language)

            if status["has_validated_translation"]:
                if isinstance(report["summary"], dict):
                    report["summary"]["validated_terms"] += 1
                if isinstance(report["validated_terms"], list):
                    report["validated_terms"].append(status)
            else:
                if isinstance(report["summary"], dict):
                    report["summary"]["pending_validations"] += 1
                severity = self.severity_mapping.get(term, ValidationSeverity.INFORMATIONAL)

                if severity == ValidationSeverity.CRITICAL:
                    if isinstance(report["summary"], dict):
                        report["summary"]["critical_pending"] += 1
                elif severity == ValidationSeverity.IMPORTANT:
                    if isinstance(report["summary"], dict):
                        report["summary"]["important_pending"] += 1

                if isinstance(report["pending_validations"], list):
                    report["pending_validations"].append(status)

        if len(self.critical_terms) > 0 and isinstance(report["summary"], dict):
            validated_count = report["summary"]["validated_terms"]
            if isinstance(validated_count, int):
                report["validation_coverage"] = (validated_count / len(self.critical_terms)) * 100

        return report


medical_validator = MedicalTerminologyValidator()


def get_medical_validator() -> MedicalTerminologyValidator:
    """Get the global medical terminology validator instance."""
    return medical_validator
