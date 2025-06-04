#!/usr/bin/env python3
"""
Test script to verify backend i18n service functionality without database dependencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.i18n_service import i18n_service

def test_i18n_service():
    """Test the i18n service functionality."""
    print("Testing CardioAI Pro Backend i18n Service")
    print("=" * 50)
    
    print("\n1. Available Languages:")
    try:
        languages = i18n_service.get_available_languages()
        print(f"   Supported languages: {languages}")
    except Exception as e:
        print(f"   Error getting languages: {e}")
    
    print("\n2. Translation Tests:")
    test_key = "validation.error.required"
    
    for lang in ['en', 'pt', 'es', 'fr', 'de']:
        try:
            translation = i18n_service.get_translation(test_key, lang)
            print(f"   {lang}: {translation}")
        except Exception as e:
            print(f"   {lang}: Error - {e}")
    
    print("\n3. Medical Terminology Tests:")
    try:
        is_valid, message, severity = i18n_service.validate_medical_translation(
            "atrial_fibrillation", "Fibrilação Atrial", "pt"
        )
        print(f"   Medical validation (PT): valid={is_valid}, severity={severity}")
        print(f"   Message: {message}")
        
        is_valid, message, severity = i18n_service.validate_medical_translation(
            "ventricular_tachycardia", "Taquicardia Ventricular", "es"
        )
        print(f"   Medical validation (ES): valid={is_valid}, severity={severity}")
        print(f"   Message: {message}")
        
    except Exception as e:
        print(f"   Medical validation error: {e}")
    
    print("\n4. Error Message Localization:")
    error_keys = [
        "validation.error.invalid_format",
        "validation.error.missing_field",
        "api.error.unauthorized"
    ]
    
    for key in error_keys:
        for lang in ['en', 'pt', 'es']:
            try:
                translation = i18n_service.get_translation(key, lang)
                print(f"   {key} ({lang}): {translation}")
            except Exception as e:
                print(f"   {key} ({lang}): Error - {e}")
    
    print("\n5. Fallback Behavior Test:")
    try:
        translation = i18n_service.get_translation("validation.error.required", "ru")
        print(f"   Unsupported language (ru) fallback: {translation}")
    except Exception as e:
        print(f"   Fallback test error: {e}")
    
    print("\n" + "=" * 50)
    print("Backend i18n Service Test Complete")

if __name__ == "__main__":
    test_i18n_service()
