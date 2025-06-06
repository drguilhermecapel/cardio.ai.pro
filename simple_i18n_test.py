#!/usr/bin/env python3
"""
Simple test script to verify backend i18n service translate method.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.i18n_service import i18n_service

def test_translate_method():
    """Test the translate method functionality."""
    print("Testing CardioAI Pro Backend i18n Translate Method")
    print("=" * 55)
    
    print("\n1. Basic Translation Tests:")
    test_keys = [
        "errors.validation_error",
        "errors.unauthorized", 
        "messages.success",
        "api.patient_not_found"
    ]
    
    for key in test_keys:
        for lang in ['en', 'pt', 'es']:
            try:
                translation = i18n_service.translate(key, lang)
                print(f"   {key} ({lang}): {translation}")
            except Exception as e:
                print(f"   {key} ({lang}): Error - {e}")
    
    print("\n2. Fallback Behavior Test:")
    try:
        translation = i18n_service.translate("errors.validation_error", "ru")
        print(f"   Unsupported language (ru) fallback: {translation}")
    except Exception as e:
        print(f"   Fallback test error: {e}")
    
    print("\n3. Parameter Substitution Test:")
    try:
        translation = i18n_service.translate("errors.field_required", "en", field="username")
        print(f"   Parameter substitution: {translation}")
    except Exception as e:
        print(f"   Parameter substitution error: {e}")
    
    print("\n4. Available Languages:")
    try:
        languages = i18n_service.get_available_languages()
        print(f"   Supported languages: {languages}")
    except Exception as e:
        print(f"   Error getting languages: {e}")
    
    print("\n" + "=" * 55)
    print("Simple i18n Translate Test Complete")

if __name__ == "__main__":
    test_translate_method()
