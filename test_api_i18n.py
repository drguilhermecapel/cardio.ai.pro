#!/usr/bin/env python3
"""
Test script to verify API responses with different Accept-Language headers.
Uses FastAPI test client to bypass database dependencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from fastapi.testclient import TestClient
from app.main import app

def test_api_i18n_responses():
    """Test API responses with different Accept-Language headers."""
    print("Testing CardioAI Pro API i18n Responses")
    print("=" * 50)
    
    client = TestClient(app)
    
    test_endpoints = [
        "/",
        "/health",
        "/api/v1/docs"
    ]
    
    languages = ['en', 'pt', 'es', 'fr', 'de']
    
    print("\n1. Testing API Root Endpoints:")
    for endpoint in test_endpoints:
        print(f"\n   Testing {endpoint}:")
        for lang in languages:
            try:
                response = client.get(endpoint, headers={"Accept-Language": lang})
                print(f"     {lang}: Status {response.status_code}")
                if response.status_code == 200:
                    content = response.text if hasattr(response, 'text') else str(response.content)
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"         Content preview: {content}")
            except Exception as e:
                print(f"     {lang}: Error - {e}")
    
    print("\n2. Testing Error Response Localization:")
    error_endpoints = [
        "/api/v1/patients/999999",  # Should return 404 or similar
        "/api/v1/nonexistent"       # Should return 404
    ]
    
    for endpoint in error_endpoints:
        print(f"\n   Testing {endpoint}:")
        for lang in ['en', 'pt', 'es']:
            try:
                response = client.get(endpoint, headers={"Accept-Language": lang})
                print(f"     {lang}: Status {response.status_code}")
                if response.status_code >= 400:
                    try:
                        error_data = response.json()
                        print(f"         Error detail: {error_data.get('detail', 'No detail')}")
                    except:
                        print(f"         Raw response: {response.text[:100]}...")
            except Exception as e:
                print(f"     {lang}: Error - {e}")
    
    print("\n3. Testing Fallback Behavior:")
    try:
        response = client.get("/", headers={"Accept-Language": "ru"})  # Unsupported language
        print(f"   Unsupported language (ru): Status {response.status_code}")
        if response.status_code == 200:
            content = response.text[:100] if hasattr(response, 'text') else str(response.content)[:100]
            print(f"   Content preview: {content}...")
    except Exception as e:
        print(f"   Fallback test error: {e}")
    
    print("\n" + "=" * 50)
    print("API i18n Response Test Complete")

if __name__ == "__main__":
    test_api_i18n_responses()
