#!/usr/bin/env python3
"""
Simple Azure Translator API Test
"""

import os
import requests
import uuid
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_translator_api():
    """Test Azure Translator API with sample text"""
    
    # Check if credentials are available
    api_key = os.environ.get('AZURE_TRANSLATOR_KEY')
    location = os.environ.get('AZURE_TRANSLATOR_LOCATION', 'global')
    
    print("🔤 Azure Translator API Test")
    print("=" * 40)
    
    if not api_key:
        print("❌ AZURE_TRANSLATOR_KEY not found in environment variables")
        print("\n📝 To set up Azure Translator:")
        print("1. Get your Azure Translator API key from Azure portal")
        print("2. Add to your .env file:")
        print("   AZURE_TRANSLATOR_KEY=your_translator_key_here")
        print("   AZURE_TRANSLATOR_LOCATION=global")
        print("3. Run this test again")
        return False
    
    print(f"✅ API Key: {'*' * 10}...")
    print(f"📍 Location: {location}")
    
    # Test translation
    try:
        endpoint = "https://api.cognitive.microsofttranslator.com"
        constructed_url = endpoint + '/translate'
        
        params = {
            'api-version': '3.0',
            'to': 'en',
            'from': 'ar'
        }
        
        headers = {
            'Ocp-Apim-Subscription-Key': api_key,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        
        # Test with Arabic text
        test_text = "مرحبا بك في نظام استخراج السياسات"  # "Welcome to the policy extraction system"
        body = [{'text': test_text}]
        
        print(f"\n🔄 Testing translation...")
        print(f"Original (Arabic): {test_text}")
        
        response = requests.post(constructed_url, params=params, headers=headers, json=body)
        response.raise_for_status()
        
        result = response.json()
        if result and len(result) > 0:
            translated = result[0]['translations'][0]['text']
            detected_lang = result[0].get('detectedLanguage', {}).get('language', 'unknown')
            confidence = result[0].get('detectedLanguage', {}).get('score', 0)
            
            print(f"✅ Translation successful!")
            print(f"Translated (English): {translated}")
            print(f"🔍 Detected language: {detected_lang} (confidence: {confidence:.2f})")
            print(f"⏱️  Response time: {response.elapsed.total_seconds():.2f} seconds")
            
            return True
        else:
            print("❌ Invalid response from translator API")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Response text: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_translator_api()
    
    if success:
        print("\n🎉 Azure Translator API is working correctly!")
        print("✅ Your translation service is ready for use.")
    else:
        print("\n⚠️  Azure Translator API is not working.")
        print("📋 Please check your credentials and try again.")

