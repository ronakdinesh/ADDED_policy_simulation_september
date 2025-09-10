# API Setup and Health Check Guide

## Overview

This project uses **Azure OpenAI** and **Azure Translator** services. This guide will help you set up the required APIs and verify they're working correctly.

## 🚨 Current Status

**❌ APIs are NOT configured**

Your APIs are currently not set up. All API tests are failing or being skipped due to missing credentials.

## 📋 Required Services

### 1. Azure OpenAI (REQUIRED)
- **Purpose**: Policy extraction and AI processing
- **Status**: ❌ Not configured
- **Required for**: Core functionality

### 2. Azure Translator (OPTIONAL)
- **Purpose**: Translating Arabic documents to English
- **Status**: ❌ Not configured  
- **Required for**: Document translation only

## 🔧 Setup Instructions

### Step 1: Create Environment File

1. Copy the template content from `env_template.txt`
2. Create a `.env` file in your project root:
   ```bash
   touch .env
   ```
3. Paste the template content into your `.env` file

### Step 2: Get Azure OpenAI Credentials

You need to obtain these credentials from your Azure OpenAI resource:

1. **AZURE_OPENAI_API_KEY**: Your API key from Azure portal
2. **AZURE_OPENAI_ENDPOINT**: Your endpoint URL (format: `https://your-resource-name.openai.azure.com/`)
3. **OPENAI_API_VERSION**: Use `2024-12-01-preview` for latest features
4. **OPENAI_MODEL_NAME**: Your deployed model name (e.g., `gpt-4`, `gpt-5`)

### Step 3: Get Azure Translator Credentials (Optional)

If you need translation capabilities:

1. **AZURE_TRANSLATOR_KEY**: Your Translator service API key
2. **AZURE_TRANSLATOR_LOCATION**: Your service region (usually `global`)

### Step 4: Configure Your .env File

Example `.env` file:
```bash
# Azure OpenAI Configuration (REQUIRED)
AZURE_OPENAI_API_KEY=your_actual_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
OPENAI_API_VERSION=2024-12-01-preview
OPENAI_MODEL_NAME=gpt-4

# Azure Translator Configuration (OPTIONAL)
AZURE_TRANSLATOR_KEY=your_translator_key_here
AZURE_TRANSLATOR_LOCATION=global

# Performance Settings (OPTIONAL)
LLM_CONCURRENCY=12
TRANSLATE_CONCURRENCY=4
```

## ✅ Verify Setup

After configuring your `.env` file, run the health check:

```bash
python api_health_check.py
```

### Expected Results (Successful Setup)

```
🔧 Starting API Health Check...
🔍 Checking Environment Variables...
✅ AZURE_OPENAI_API_KEY: **********...
✅ AZURE_OPENAI_ENDPOINT: **********...
✅ OPENAI_API_VERSION: **********...
✅ OPENAI_MODEL_NAME: **********...

🌐 Testing API Connectivity...
✅ API Response received successfully
⏱️  Response time: 1.23 seconds

🤖 Testing Model Availability...
✅ Model gpt-4 is working correctly

🚀 Testing Concurrent Requests...
✅ Completed 3/3 requests in 2.45 seconds

🔤 Testing Azure Translator API...
✅ Translation successful
💬 Translation: 'مرحبا بك' → 'Welcome'

📋 API HEALTH CHECK SUMMARY
✅ Passed: 4/5 tests
🎉 All API tests passed! Your Azure OpenAI setup is working correctly.
```

## 🔍 Troubleshooting

### Common Issues

1. **Environment Variables Not Found**
   - Ensure `.env` file is in the project root directory
   - Check that variable names match exactly (case-sensitive)
   - Restart your terminal/IDE after creating `.env`

2. **API Key Invalid**
   - Verify your API key is correct and active
   - Check that your Azure subscription is active
   - Ensure proper permissions for the API key

3. **Endpoint URL Issues**
   - Ensure endpoint format is correct: `https://your-resource-name.openai.azure.com/`
   - Don't include trailing paths like `/openai/deployments/`

4. **Model Not Found**
   - Verify your model deployment name matches `OPENAI_MODEL_NAME`
   - Check that the model is deployed and active in Azure portal

5. **Rate Limit Errors**
   - Reduce `LLM_CONCURRENCY` value (try 3-5)
   - Check your quota limits in Azure portal

## 📊 API Usage in This Project

### Files Using Azure OpenAI:
- `c_policy_agent.py` - Main policy extraction agent
- `utils.py` - Utility functions with OpenAI client
- Archive files - Legacy implementations

### Files Using Azure Translator:
- `utils.py` - Translation functions for Arabic documents

## 💡 Performance Tuning

### Recommended Settings:

For **Standard** Azure OpenAI tier:
```bash
LLM_CONCURRENCY=6
TRANSLATE_CONCURRENCY=4
```

For **Premium** Azure OpenAI tier:
```bash
LLM_CONCURRENCY=12
TRANSLATE_CONCURRENCY=8
```

## 🔐 Security Notes

- Never commit your `.env` file to version control
- Store API keys securely
- Use Azure Key Vault for production deployments
- Rotate API keys regularly

## 📞 Need Help?

1. Run the health check: `python api_health_check.py`
2. Check the detailed results in `api_health_check_results.json`
3. Review Azure portal for service status
4. Check Azure OpenAI service logs for detailed error information

---

**Next Steps**: Once all API tests pass, you can run the main policy extraction pipeline with confidence that all services are properly configured.


