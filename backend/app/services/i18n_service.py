from typing import Dict, Any, Optional, Tuple
import json
import os
from pathlib import Path
import structlog
from .medical_translation_service import get_medical_validator, ValidationSeverity

logger = structlog.get_logger(__name__)


class I18nService:
    """
    Internationalization service for backend API responses and error messages.
    Provides translation functionality with fallback support.
    """
    
    def __init__(self) -> None:
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.default_language = "en"
        self.medical_validator = get_medical_validator()
        self.load_translations()
    
    def load_translations(self) -> None:
        """Load translation files from the translations directory."""
        try:
            translations_dir = Path(__file__).parent.parent / "translations"
            if not translations_dir.exists():
                logger.warning(f"Translations directory not found: {translations_dir}")
                return
            
            for lang_file in translations_dir.glob("*.json"):
                lang_code = lang_file.stem
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[lang_code] = json.load(f)
                    logger.info(f"Loaded translations for language: {lang_code}")
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Failed to load translations for {lang_code}: {e}")
        except Exception as e:
            logger.error(f"Error loading translations: {e}")
    
    def translate(self, key: str, lang: str = "en", validate_medical: bool = True, **kwargs: Any) -> str:
        """
        Translate a key to the specified language with parameter substitution and optional medical validation.
        
        Args:
            key: Translation key (e.g., "errors.validation_error")
            lang: Language code (e.g., "en", "pt", "es")
            validate_medical: Whether to validate medical terminology
            **kwargs: Parameters for string formatting
            
        Returns:
            Translated string with parameters substituted, or the key if translation not found
        """
        try:
            translation = self._get_nested_value(self.translations.get(lang, {}), key)
            
            if translation is None and lang != self.default_language:
                translation = self._get_nested_value(
                    self.translations.get(self.default_language, {}), key
                )
            
            if translation is None:
                logger.warning(f"Translation not found for key: {key}, language: {lang}")
                return key
            
            if validate_medical and self._is_medical_term(key):
                is_valid, message, severity = self.medical_validator.validate_term(key, translation, lang)
                if not is_valid and severity == ValidationSeverity.CRITICAL:
                    logger.error(f"Critical medical term validation failed: {message}")
                    if lang != self.default_language:
                        fallback_translation = self._get_nested_value(
                            self.translations.get(self.default_language, {}), key
                        )
                        if fallback_translation:
                            logger.warning(f"Using English fallback for critical medical term: {key}")
                            translation = fallback_translation
                elif not is_valid:
                    logger.warning(f"Medical term validation warning: {message}")
            
            if kwargs:
                try:
                    return translation.format(**kwargs)
                except (KeyError, ValueError) as e:
                    logger.error(f"Error formatting translation for key {key}: {e}")
                    return translation
            
            return translation
            
        except Exception as e:
            logger.error(f"Error translating key {key}: {e}")
            return key
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Optional[str]:
        """
        Get nested value from dictionary using dot notation.
        
        Args:
            data: Dictionary to search in
            key: Dot-separated key (e.g., "errors.validation_error")
            
        Returns:
            Value if found, None otherwise
        """
        try:
            keys = key.split('.')
            current = data
            
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return None
            
            return current if isinstance(current, str) else None
            
        except Exception:
            return None
    
    def get_available_languages(self) -> list[str]:
        """Get list of available language codes."""
        return list(self.translations.keys())
    
    def is_language_supported(self, lang: str) -> bool:
        """Check if a language is supported."""
        return lang in self.translations
    
    def reload_translations(self) -> None:
        """Reload all translation files."""
        self.translations.clear()
        self.load_translations()
    
    def _is_medical_term(self, key: str) -> bool:
        """Check if a translation key represents a medical term."""
        medical_prefixes = [
            "medical.", "ecg.", "arrhythmia.", "cardiac.", "diagnosis.", 
            "condition.", "symptom.", "treatment.", "medication.", "procedure."
        ]
        return any(key.startswith(prefix) for prefix in medical_prefixes)
    
    def validate_medical_translation(self, key: str, translation: str, lang: str) -> Tuple[bool, str, str]:
        """
        Validate a medical translation.
        
        Args:
            key: Translation key
            translation: The translated text
            lang: Language code
            
        Returns:
            Tuple of (is_valid, message, severity)
        """
        is_valid, message, severity = self.medical_validator.validate_term(key, translation, lang)
        return is_valid, message, severity.value
    
    def get_medical_validation_status(self, key: str, lang: str) -> Dict[str, Any]:
        """Get medical validation status for a specific term and language."""
        return self.medical_validator.get_validation_status(key, lang)
    
    def register_medical_validation(self, key: str, translation: str, lang: str, 
                                   validator_id: str, validator_credentials: str) -> None:
        """Register professional validation for a medical term."""
        self.medical_validator.register_professional_validation(
            key, translation, lang, validator_id, validator_credentials
        )
    
    def get_pending_medical_validations(self, lang: str) -> list[Dict[str, Any]]:
        """Get list of medical terms pending validation for a language."""
        return self.medical_validator.get_pending_validations(lang)
    
    def export_medical_validation_report(self, lang: str) -> Dict[str, Any]:
        """Export medical validation report for a language."""
        return self.medical_validator.export_validation_report(lang)


i18n_service = I18nService()
