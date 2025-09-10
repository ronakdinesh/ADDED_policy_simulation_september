#!/usr/bin/env python3
"""
API Health Check Script
This script tests the Azure OpenAI API connectivity and configuration.
"""

import os
import asyncio
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import httpx
from openai import AsyncAzureOpenAI
from datetime import datetime
import json
import requests
import uuid

# Load environment variables
load_dotenv()

class APIHealthChecker:
    """Health checker for Azure OpenAI API"""
    
    def __init__(self):
        self.results = {}
        self.setup_config()
    
    def setup_config(self):
        """Setup configuration from environment variables"""
        self.config = {
            'AZURE_OPENAI_KEY': os.environ.get("AZURE_OPENAI_API_KEY", ""),
            'AZURE_OPENAI_ENDPOINT': os.environ.get("AZURE_OPENAI_ENDPOINT", "https://teams-ai-agent.openai.azure.com/"),
            'AZURE_OPENAI_API_VERSION': os.environ.get("OPENAI_API_VERSION", "2024-12-01-preview"),
            'AZURE_OPENAI_DEPLOYMENT': os.environ.get("OPENAI_MODEL_NAME", "gpt-5"),
            'LLM_CONCURRENCY': int(os.environ.get("LLM_CONCURRENCY", "12")),
            'AZURE_TRANSLATOR_KEY': os.environ.get("AZURE_TRANSLATOR_KEY", ""),
            'AZURE_TRANSLATOR_LOCATION': os.environ.get("AZURE_TRANSLATOR_LOCATION", "global"),
            'TRANSLATE_CONCURRENCY': int(os.environ.get("TRANSLATE_CONCURRENCY", "4"))
        }
    
    def check_environment_variables(self) -> Dict[str, Any]:
        """Check if all required environment variables are set"""
        print("🔍 Checking Environment Variables...")
        
        required_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT', 
            'OPENAI_API_VERSION',
            'OPENAI_MODEL_NAME'
        ]
        
        optional_vars = [
            'AZURE_TRANSLATOR_KEY',
            'AZURE_TRANSLATOR_LOCATION'
        ]
        
        missing_vars = []
        present_vars = []
        
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                present_vars.append(var)
                print(f"✅ {var}: {'*' * min(len(value), 10)}...")
            else:
                missing_vars.append(var)
                print(f"❌ {var}: NOT SET")
        
        # Check optional variables for OpenAI
        openai_optional_vars = ['LLM_CONCURRENCY', 'TRANSLATE_CONCURRENCY']
        for var in openai_optional_vars:
            value = os.environ.get(var)
            if value:
                print(f"ℹ️  {var}: {value}")
            else:
                print(f"ℹ️  {var}: Using default")
        
        # Check optional translator variables
        translator_missing = []
        translator_present = []
        for var in optional_vars:
            value = os.environ.get(var)
            if value:
                translator_present.append(var)
                print(f"🔤 {var}: {'*' * min(len(value), 10)}...")
            else:
                translator_missing.append(var)
                print(f"🔤 {var}: NOT SET (optional for translation)")
        
        translator_configured = len(translator_missing) == 0
        
        result = {
            'status': 'pass' if not missing_vars else 'fail',
            'missing_variables': missing_vars,
            'present_variables': present_vars,
            'total_required': len(required_vars),
            'total_present': len(present_vars),
            'translator_configured': translator_configured,
            'translator_missing': translator_missing,
            'translator_present': translator_present
        }
        
        if missing_vars:
            print(f"\n⚠️  Missing {len(missing_vars)} required environment variables")
            print("📝 Create a .env file with the following variables:")
            print("=" * 50)
            for var in missing_vars:
                print(f"{var}=your_value_here")
            print("=" * 50)
        else:
            print("✅ All required environment variables are present")
        
        return result
    
    async def check_api_connectivity(self) -> Dict[str, Any]:
        """Test basic API connectivity"""
        print("\n🌐 Testing API Connectivity...")
        
        if not self.config['AZURE_OPENAI_KEY']:
            return {
                'status': 'skip',
                'message': 'API key not available - skipping connectivity test'
            }
        
        try:
            # Create HTTP client
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=25,
                    max_keepalive_connections=15,
                    keepalive_expiry=30
                ),
                timeout=httpx.Timeout(60.0)
            )
            
            # Create Azure OpenAI client
            client = AsyncAzureOpenAI(
                api_key=self.config['AZURE_OPENAI_KEY'],
                api_version=self.config['AZURE_OPENAI_API_VERSION'],
                azure_endpoint=self.config['AZURE_OPENAI_ENDPOINT'],
                max_retries=3,
                http_client=http_client
            )
            
            print(f"🔗 Testing connection to: {self.config['AZURE_OPENAI_ENDPOINT']}")
            print(f"📦 Using deployment: {self.config['AZURE_OPENAI_DEPLOYMENT']}")
            print(f"🔢 API Version: {self.config['AZURE_OPENAI_API_VERSION']}")
            
            # Test with a simple completion
            start_time = datetime.now()
            response = await client.chat.completions.create(
                model=self.config['AZURE_OPENAI_DEPLOYMENT'],
                messages=[
                    {"role": "user", "content": "Say 'API test successful' if you can read this."}
                ],
                max_completion_tokens=50
            )
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            
            await client.close()
            await http_client.aclose()
            
            print(f"✅ API Response received successfully")
            print(f"⏱️  Response time: {response_time:.2f} seconds")
            print(f"💬 Response: {response.choices[0].message.content.strip()}")
            
            return {
                'status': 'pass',
                'response_time': response_time,
                'response_content': response.choices[0].message.content.strip(),
                'model': self.config['AZURE_OPENAI_DEPLOYMENT'],
                'tokens_used': response.usage.total_tokens if response.usage else 'unknown'
            }
            
        except Exception as e:
            print(f"❌ API connectivity test failed: {str(e)}")
            return {
                'status': 'fail',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    async def check_model_availability(self) -> Dict[str, Any]:
        """Check if the specified model/deployment is available"""
        print("\n🤖 Testing Model Availability...")
        
        if not self.config['AZURE_OPENAI_KEY']:
            return {
                'status': 'skip',
                'message': 'API key not available - skipping model test'
            }
        
        try:
            http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
            client = AsyncAzureOpenAI(
                api_key=self.config['AZURE_OPENAI_KEY'],
                api_version=self.config['AZURE_OPENAI_API_VERSION'],
                azure_endpoint=self.config['AZURE_OPENAI_ENDPOINT'],
                http_client=http_client
            )
            
            # Test a more complex request to ensure the model works properly
            response = await client.chat.completions.create(
                model=self.config['AZURE_OPENAI_DEPLOYMENT'],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds in JSON format."},
                    {"role": "user", "content": 'Respond with JSON: {"status": "working", "model_test": "passed"}'}
                ],
                max_completion_tokens=100
            )
            
            await client.close()
            await http_client.aclose()
            
            content = response.choices[0].message.content.strip()
            print(f"✅ Model {self.config['AZURE_OPENAI_DEPLOYMENT']} is working correctly")
            print(f"📊 Tokens used: {response.usage.total_tokens if response.usage else 'unknown'}")
            
            # Try to parse JSON response
            try:
                json_response = json.loads(content)
                print(f"✅ JSON parsing successful: {json_response}")
            except json.JSONDecodeError:
                print(f"⚠️  Response is not valid JSON: {content}")
            
            return {
                'status': 'pass',
                'model': self.config['AZURE_OPENAI_DEPLOYMENT'],
                'response': content,
                'tokens_used': response.usage.total_tokens if response.usage else None
            }
            
        except Exception as e:
            print(f"❌ Model availability test failed: {str(e)}")
            return {
                'status': 'fail',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    async def test_concurrent_requests(self) -> Dict[str, Any]:
        """Test concurrent API requests"""
        print("\n🚀 Testing Concurrent Requests...")
        
        if not self.config['AZURE_OPENAI_KEY']:
            return {
                'status': 'skip',
                'message': 'API key not available - skipping concurrency test'
            }
        
        try:
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=25, max_keepalive_connections=15),
                timeout=httpx.Timeout(30.0)
            )
            client = AsyncAzureOpenAI(
                api_key=self.config['AZURE_OPENAI_KEY'],
                api_version=self.config['AZURE_OPENAI_API_VERSION'],
                azure_endpoint=self.config['AZURE_OPENAI_ENDPOINT'],
                http_client=http_client
            )
            
            # Test with 3 concurrent requests
            num_requests = 3
            print(f"🔄 Sending {num_requests} concurrent requests...")
            
            async def make_request(request_id: int):
                try:
                    response = await client.chat.completions.create(
                        model=self.config['AZURE_OPENAI_DEPLOYMENT'],
                        messages=[
                            {"role": "user", "content": f"This is test request #{request_id}. Respond with: 'Request {request_id} processed successfully'"}
                        ],
                        max_completion_tokens=30
                    )
                    return {
                        'id': request_id, 
                        'status': 'success',
                        'content': response.choices[0].message.content.strip(),
                        'tokens': response.usage.total_tokens if response.usage else None
                    }
                except Exception as e:
                    return {'id': request_id, 'status': 'error', 'error': str(e)}
            
            start_time = datetime.now()
            tasks = [make_request(i) for i in range(1, num_requests + 1)]
            results = await asyncio.gather(*tasks)
            end_time = datetime.now()
            
            await client.close()
            await http_client.aclose()
            
            total_time = (end_time - start_time).total_seconds()
            successful_requests = sum(1 for r in results if r['status'] == 'success')
            
            print(f"✅ Completed {successful_requests}/{num_requests} requests in {total_time:.2f} seconds")
            
            for result in results:
                if result['status'] == 'success':
                    print(f"  ✅ Request {result['id']}: {result['content']}")
                else:
                    print(f"  ❌ Request {result['id']}: {result['error']}")
            
            return {
                'status': 'pass' if successful_requests == num_requests else 'partial',
                'total_requests': num_requests,
                'successful_requests': successful_requests,
                'total_time': total_time,
                'requests_per_second': num_requests / total_time,
                'results': results
            }
            
        except Exception as e:
            print(f"❌ Concurrent requests test failed: {str(e)}")
            return {
                'status': 'fail',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def check_translator_api(self) -> Dict[str, Any]:
        """Test Azure Translator API connectivity"""
        print("\n🔤 Testing Azure Translator API...")
        
        if not self.config['AZURE_TRANSLATOR_KEY']:
            return {
                'status': 'skip',
                'message': 'Azure Translator API key not available - skipping translation test'
            }
        
        try:
            # Set up the request
            endpoint = "https://api.cognitive.microsofttranslator.com"
            location = self.config['AZURE_TRANSLATOR_LOCATION']
            key = self.config['AZURE_TRANSLATOR_KEY']
            
            constructed_url = endpoint + '/translate'
            params = {'api-version': '3.0', 'to': 'en', 'from': 'ar'}
            headers = {
                'Ocp-Apim-Subscription-Key': key,
                'Ocp-Apim-Subscription-Region': location,
                'Content-type': 'application/json',
                'X-ClientTraceId': str(uuid.uuid4())
            }
            
            # Test with a simple Arabic text
            test_text = "مرحبا بك"  # "Welcome" in Arabic
            body = [{'text': test_text}]
            
            print(f"🔗 Testing connection to: {endpoint}")
            print(f"📍 Using region: {location}")
            print(f"🔤 Testing translation: '{test_text}' (Arabic to English)")
            
            start_time = datetime.now()
            response = requests.post(constructed_url, params=params, headers=headers, json=body)
            response.raise_for_status()
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            translation_result = response.json()
            
            if translation_result and len(translation_result) > 0:
                translated_text = translation_result[0]['translations'][0]['text']
                detected_lang = translation_result[0].get('detectedLanguage', {}).get('language', 'unknown')
                
                print(f"✅ Translation successful")
                print(f"⏱️  Response time: {response_time:.2f} seconds")
                print(f"🔍 Detected language: {detected_lang}")
                print(f"💬 Translation: '{test_text}' → '{translated_text}'")
                
                return {
                    'status': 'pass',
                    'response_time': response_time,
                    'original_text': test_text,
                    'translated_text': translated_text,
                    'detected_language': detected_lang,
                    'endpoint': endpoint,
                    'region': location
                }
            else:
                print("❌ Invalid response from translator API")
                return {
                    'status': 'fail',
                    'error': 'Invalid response format',
                    'response': translation_result
                }
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Azure Translator API test failed: {str(e)}")
            return {
                'status': 'fail',
                'error': str(e),
                'error_type': type(e).__name__
            }
        except Exception as e:
            print(f"❌ Unexpected error during translation test: {str(e)}")
            return {
                'status': 'fail',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def print_summary(self):
        """Print a summary of all test results"""
        print("\n" + "="*60)
        print("📋 API HEALTH CHECK SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get('status') == 'pass')
        failed_tests = sum(1 for r in self.results.values() if r.get('status') == 'fail')
        skipped_tests = sum(1 for r in self.results.values() if r.get('status') == 'skip')
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏭️  Skipped: {skipped_tests}")
        
        if failed_tests == 0 and passed_tests > 0:
            print("\n🎉 All API tests passed! Your Azure OpenAI setup is working correctly.")
        elif failed_tests > 0:
            print(f"\n⚠️  {failed_tests} test(s) failed. Check the errors above for details.")
        else:
            print("\n⚠️  Most tests were skipped due to missing configuration.")
        
        # Print next steps
        if self.results.get('environment', {}).get('status') == 'fail':
            print("\n📝 NEXT STEPS:")
            print("1. Create a .env file in your project directory")
            print("2. Add your Azure OpenAI credentials")
            print("3. Run this script again to verify the setup")
    
    async def run_all_checks(self):
        """Run all health checks"""
        print("🔧 Starting API Health Check...")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all checks
        self.results['environment'] = self.check_environment_variables()
        self.results['connectivity'] = await self.check_api_connectivity()
        self.results['model'] = await self.check_model_availability()
        self.results['concurrency'] = await self.test_concurrent_requests()
        self.results['translator'] = self.check_translator_api()
        
        # Print summary
        self.print_summary()
        
        return self.results

async def main():
    """Main function"""
    checker = APIHealthChecker()
    results = await checker.run_all_checks()
    
    # Save results to file
    with open('api_health_check_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: api_health_check_results.json")

if __name__ == "__main__":
    asyncio.run(main())
