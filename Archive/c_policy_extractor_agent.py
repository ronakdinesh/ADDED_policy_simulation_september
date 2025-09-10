#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Extractor

A unified tool for extracting structured policy information from documents
using smart chunking and Pydantic AI agents.

This script:
1. Intelligently chunks documents based on policy boundaries
2. Uses Pydantic AI to extract structured information from each policy
3. Combines results into a coherent CSV output matching ParametersV2.csv format

Usage:
    python policy_extractor.py --input <input_file.pdf> --output <output_dir>
"""

import os
import sys
import json
import csv
import time
import pandas as pd
import traceback
from dotenv import load_dotenv
import logging
import argparse
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Load environment variables
load_dotenv()

# Import our custom modules
from b_smart_chunking_agent import create_smart_chunks, save_chunks_to_disk
from a_preprocessing import count_tokens  # Reuse token counting function

# OpenAI dependencies
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel


# Configure logging - updated to use Output directory
OUTPUT_DIR = "Output"  # Standardized output directory name
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure directory exists

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'policy_extraction.log'))
    ]
)
logger = logging.getLogger(__name__)


# Document source information
class DocumentSource(BaseModel):
    """Source information for a document"""
    filename: str
    folder_path: str
    page_numbers: List[int] = Field(default_factory=list)
    language: Optional[str] = None


# Policy model based on ParametersV2.csv
class Policy(BaseModel):
    """Pydantic model for policy information extraction"""
    # Main identification fields
    policy_name: str = Field(description="The official name of the policy as it appears in the document. Look for standard naming formats such as 'Law No. (XX) of Year YYYY concerning...', 'Chairman of the Executive Council Resolution No. (XX) of Year YYYY...', 'Executive Council Resolution No. (XX)...', 'Amiri Decree No. (XX)...', or 'Circular No. (XX)...'")
    relevance: Optional[str] = Field(None, description="Whether the policy is an 'economic policy' (label as 'relevant') or not (label as 'not relevant'). An economic policy must have a direct and intended impact on GDP components (government spending/revenues, investment, consumption, or net exports).")
    issuing_year: Optional[str] = Field(None, description="The year the policy was officially issued. Extract this information either from the policy name (e.g., 'Law No. (XX) of Year 2020') or from the publication date found within the document. Format as a 4-digit year (e.g., '2020').")
    has_tables: Optional[str] = Field(None, description="Yes/No if policy has tables or numeric values")
    
    # Content fields
    policy_summary: Optional[str] = Field(None, description="Create a comprehensive summary of this policy document using ONLY terminology and concepts explicitly mentioned in the text. Do not introduce new terms or infer information not directly stated. Include: 1) The exact name and number of the policy, 2) Direct quotes of key purposes and objectives, 3) Specific bodies/committees established (with exact names), 4) Explicitly stated implementation mechanisms. Limit to 3-4 sentences while maintaining substantive content.")
    goals: Optional[str] = Field(None, description="High-level qualitative aims the policy wants to achieve")
    objectives: Optional[str] = Field(None, description="Specific, actionable, quantitative targets")
    situation_analysis: Optional[str] = Field(None, description="Background context for the policy")
    
    # Classification fields
    legislative_instrument: Optional[str] = Field(None, description="The legal tool used to enforce the policy. Must be classified as one of the following options only: 'UAE Constitution', 'Federal Law', 'Decree by Law', 'Federal Regulation and Decision', 'Local Law of the Emirate of Abu Dhabi', 'Amiri Decree', 'Executive Council Decisions for the Emirate of Abu Dhabi', or 'Executive Regulation and Local Regulatory Decision'.")
    federal_emirate: Optional[str] = Field(None, description="Whether the policy is developed on the federal level or just the Emirate of Abu Dhabi's level. Must be classified as either 'Federal' or 'Emirate'.")
    region: Optional[str] = Field(None, description="Geographical coverage area. Use 'All' if policy refers to entire Emirate of Abu Dhabi, simply mentions 'Abu Dhabi', or doesn't specify a region. Otherwise use specific regions: 'Abu Dhabi City', 'Al Ain', 'Al Dhafra', 'Abu Dhabi City & Al Ain', 'Abu Dhabi City & Al Dhafra', or 'Al Ain & Al Dhafra' if explicitly stated in the policy.")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction scope")
    sector: Optional[str] = Field(None, description="Economic sector(s) affected")
    
    # Performance and ownership fields
    metrics: Optional[str] = Field(None, description="KPIs explicitly mentioned in the document")
    policy_owner: Optional[str] = Field(None, description="Government entity owning the policy")
    stakeholders_involved: Optional[str] = Field(None, description="Other entities with roles")
    thematic_categorization: Optional[str] = Field(None, description="General policy theme")
    
    # Source tracking (added by us, not extracted by the agent)
    source: Optional[DocumentSource] = None
    confidence_score: Optional[float] = None
    extraction_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    # Helper methods
    def get_valid_fields(self) -> Dict[str, Any]:
        """Return only fields that have values (not None)"""
        return {k: v for k, v in self.model_dump().items() 
                if k != 'source' and v is not None and v != ""}
    
    model_config = ConfigDict(
        extra="ignore",
        protected_namespaces=()
    )


# Agent response models
class ExtractionResult(BaseModel):
    """Model for extraction results"""
    policies: List[Policy] = Field(default_factory=list, description="List of policies found in the document chunk")
    error: Optional[str] = None
    has_policy: bool = Field(description="Whether a policy was found")
    policy_count: int = Field(default=0, description="Number of policies found")
    policy_page_ranges: Dict[str, List[int]] = Field(default_factory=dict, description="Policy name to page ranges mapping")


class AgentResponse(BaseModel):
    """Main agent response model"""
    message: Optional[str] = Field(None, description="Human-readable message")
    error: Optional[str] = Field(None, description="Error message if any")
    extraction_result: Optional[ExtractionResult] = Field(None, description="Extraction results")
    suggested_next_action: Optional[str] = Field(None, description="Suggested next action")


class Config:
    """Configuration for the Policy Extraction Agent"""
    # Azure OpenAI settings - use values from .env
    AZURE_OPENAI_KEY = os.environ.get("Azure_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT = os.environ.get("Azure_OPENAI_ENDPOINT", "https://teams-ai-agent.openai.azure.com") 
    AZURE_OPENAI_API_VERSION = "2024-10-21"  # Updated to match policy_extraction_agent.py
    AZURE_OPENAI_DEPLOYMENT = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")

    # Processing settings
    MAX_TOKENS = 128000  # Maximum context window for GPT-4o
    SYSTEM_OVERHEAD = 15000  # Estimate for system prompt overhead
    OUTPUT_DIR = "Output"  # Updated to standardized output directory name
    CHUNKS_DIR = "chunks"  # Directory to store chunks


class PolicyExtractor:
    """Policy extraction orchestrator combining smart chunking with AI agent extraction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_openai_client()
        self.load_parameter_definitions()
        self.setup_agent()
        self.api_timeout = 180  # 3 minutes timeout for API calls
    
    def setup_openai_client(self):
        """Set up the Azure OpenAI client"""
        logger.info(f"Connecting to Azure OpenAI at: {self.config.AZURE_OPENAI_ENDPOINT}")
        logger.info(f"Using deployment: {self.config.AZURE_OPENAI_DEPLOYMENT}")
        logger.info(f"API Version: {self.config.AZURE_OPENAI_API_VERSION}")
        
        # Use the working approach from policy_extraction_agent.py
        self.client = AsyncAzureOpenAI(
            api_key=self.config.AZURE_OPENAI_KEY,
            api_version=self.config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT
        )
        
        # Set up OpenAI model for Pydantic AI exactly as in policy_extraction_agent.py
        self.model = OpenAIModel(
            self.config.AZURE_OPENAI_DEPLOYMENT,
            openai_client=self.client
        )
        
        # Model settings to use during run
        self.model_settings = {'temperature': 0.0, 'seed': 42}
    
    def load_parameter_definitions(self) -> str:
        """Load parameter definitions from CSV file"""
        try:
            # Try to load ParametersV2.csv
            v2_path = os.path.join(os.path.dirname(__file__), "ParametersV2.csv")
            if os.path.exists(v2_path):
                logger.info(f"Loading parameter definitions from {v2_path}")
                with open(v2_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header row
                    definitions = []
                    for row in reader:
                        if len(row) >= 3:
                            param_name, param_type, description = row[0], row[1], row[2]
                            if param_name and description:
                                definitions.append(f"{param_name} ({param_type}): {description}")
                    return "\n".join(definitions)
            
            logger.warning("ParametersV2.csv not found")
            return "Parameter definitions could not be loaded."
        except Exception as e:
            logger.error(f"Error loading parameter definitions: {e}")
            return "Parameter definitions could not be loaded."
    
    def setup_agent(self) -> None:
        """Set up the AI agent with system prompt"""
        # Load parameter definitions
        parameter_definitions = self.load_parameter_definitions()
        
        # System prompt with detailed instructions
        self.system_prompt = f"""
You are an expert policy analyst specializing in economic policy extraction and analysis. Your task is to extract structured information about economic policies from documents.

## ECONOMIC POLICY IDENTIFICATION GUIDELINES:
An economic policy must have a direct and intended economic impact on GDP components (government spending, investment, consumption, net exports). Examples include:
- Establishing governmental revenue-generating entities
- Creating investment zones or economic clusters
- Digitizing government services to improve efficiency
- Tax incentives or subsidies for businesses
- Developing infrastructure to support economic activities
- Regulating business activities or market operations
- Implementing labor market reforms
- Establishing economic development funds
- Creating special economic zones
- Implementing trade policies

NOT considered economic policies:
- Purely governance-related policies without direct economic impact
- Social policies focused solely on welfare without economic objectives
- Strictly environmental policies without economic components

## POLICY NAME IDENTIFICATION:
Look for policy names in formats such as:
- "Law No. (XX) of Year YYYY concerning..."
- "Decree No. (XX) of Year YYYY regarding..."
- "Resolution No. (XX) of Year YYYY on..."
- "Policy on [subject matter]"
- "Strategy for [economic objective]"
- "Program for [economic development area]"

## LEGISLATIVE INSTRUMENT CLASSIFICATION GUIDELINES:
When determining the legislative instrument, classify it as one of the following options only:

1. UAE Constitution: The highest source of all legislation in the country, establishing the general foundations of the legal and legislative system (note: examples of this are rare).
2. Federal Law: Issued by the Federal National Council and ratified by the President of the State. Example format: "Federal Law No. (X) of Year YYYY".
3. Decree by Law: Issued directly by the President of the State in urgent cases that do not allow for delays (such as during periods when the Federal National Council is not in session), and they carry the force of law. Example format: "Federal Decree by Law No. (X) of Year YYYY".
4. Federal Regulation and Decision: Issued by the Cabinet or relevant federal regulatory authorities to implement and interpret federal laws. Example format: "Federal Cabinet Resolution No. (X) of Year YYYY" or "Ministerial Decision No. (X) of Year YYYY".
5. Local Law of the Emirate of Abu Dhabi: Issued by the Ruler of the Emirate to regulate local matters, provided they do not conflict with federal laws. Example format: "Law No. (X) of Year YYYY" issued by the Ruler of Abu Dhabi.
6. Amiri Decree: Issued by the Ruler of Abu Dhabi to address specific matters within the Emirate. Example format: "Amiri Decree No. (X) of Year YYYY".
7. Executive Council Decisions for the Emirate of Abu Dhabi: Issued by the Executive Council, addressing public policies and executive procedures within the Emirate. Example format: "Executive Council Resolution No. (X) of Year YYYY".
8. Executive Regulation and Local Regulatory Decision: Issued by relevant government entities, such as the Department of Economic Development, to regulate economic and administrative activities in accordance with applicable laws. Example format: "Chairman Resolution No. (X) of Year YYYY" or "Administrative Decision No. (X) of Year YYYY".

## PARAMETER DEFINITIONS:
{parameter_definitions}

## OUTPUT FORMAT:
Provide your analysis in a structured JSON format that matches the Policy model definition. Include all required fields and as many optional fields as you can extract from the document.

Example output format:
```json
[
  {{
    "policy_name": "Law No. (XX) of Year YYYY concerning Economic Development",
    "relevance": "relevant",
    "issuing_year": "YYYY",
    "has_tables": "Yes",
    "policy_summary": "This law establishes a framework for economic development...",
    "goals": "1. Increase economic diversification\\n2. Attract foreign investment",
    "objectives": "1. Increase non-oil GDP by 5% by 2025\\n2. Create 10,000 new jobs",
    "situation_analysis": "The policy addresses challenges in economic diversification...",
    "legislative_instrument": "Federal Law",
    "federal_emirate": "Emirate",
    "region": "All",
    "jurisdiction": "Mainland",
    "sector": "Multi-sector",
    "metrics": "1. GDP growth rate\\n2. Employment rate in targeted sectors",
    "policy_owner": "Department of Economic Development",
    "stakeholders_involved": "Ministry of Finance, Abu Dhabi Investment Authority",
    "thematic_categorization": "Non oil GDP growth",
    "page_numbers": [1, 2, 3, 4]
  }}
]
```

Remember to be thorough, accurate, and provide as much detail as possible for each policy identified.
"""
        
        self.agent = Agent(
            self.model,
            result_type=AgentResponse,
            system_prompt=self.system_prompt
        )
    
    async def process_chunk(self, chunk: Dict[str, Any], source_info: Dict[str, Any]) -> AgentResponse:
        """Process a single document chunk with the agent"""
        try:
            # Create context header
            context_header = f"""
Document Information:
- Filename: {source_info['filename']}
- Location: {source_info['folder_path']}
- Pages: {', '.join(map(str, source_info['page_numbers']))}
- Chunk ID: {chunk['chunk_id']}

Policy Context:
- Policy Name: {chunk['policy_name']}
- Complete Policy: {'Yes' if chunk.get('complete_policy', False) else 'No'}
- Chunk Part: {chunk.get('chunk_part', 1)} of {chunk.get('total_chunks', 1)}

Document Content:
==================
"""
            # Combine header with content
            text_with_context = context_header + chunk["content"]
            
            # Count tokens to ensure we're within limits
            total_tokens = count_tokens(text_with_context, self.config.AZURE_OPENAI_DEPLOYMENT)
            available_tokens = self.config.MAX_TOKENS - self.config.SYSTEM_OVERHEAD
            
            if total_tokens > available_tokens:
                logger.warning(f"Chunk exceeds available context window ({total_tokens} tokens > {available_tokens} tokens)")
                logger.info(f"Truncating chunk to fit within context window...")
                
                # Truncate to fit within context window
                # This is a simple approach - could be improved with smarter truncation
                import tiktoken
                encoding = tiktoken.encoding_for_model(self.config.AZURE_OPENAI_DEPLOYMENT)
                tokens = encoding.encode(text_with_context)
                
                # Calculate how much to truncate
                excess_tokens = total_tokens - available_tokens + 1000  # Add buffer
                truncated_tokens = tokens[:-excess_tokens]
                text_with_context = encoding.decode(truncated_tokens)
                
                # Add truncation note
                text_with_context += "\n\n[NOTE: This document was truncated to fit within the context window.]"
                
                # Recount tokens
                new_token_count = count_tokens(text_with_context, self.config.AZURE_OPENAI_DEPLOYMENT)
                logger.info(f"Truncated chunk from {total_tokens} to {new_token_count} tokens")

            # Run the agent and handle the coroutine (properly awaiting the async call)
            logger.info(f"Starting AI API call for chunk...")
            start_time = time.time()
            
            # Run the agent with await directly, no need for complex asyncio handling now
            try:
                result = await self.agent.run(
                    text_with_context,
                    model_settings=self.model_settings
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"AI API call completed in {elapsed_time:.2f} seconds")
                
                return result.data
                
            except Exception as api_error:
                elapsed_time = time.time() - start_time
                logger.error(f"API call failed after {elapsed_time:.2f} seconds: {str(api_error)}")
                raise api_error
            
        except Exception as e:
            # Handle errors gracefully
            logger.error(f"Error processing chunk: {str(e)}")
            traceback.print_exc()
            
            # Return an error response
            error_message = str(e)
            if isinstance(e, TimeoutError) or "timeout" in error_message.lower():
                error_message = f"API call timed out after {self.api_timeout} seconds. Please try again or check API service."
            
            return AgentResponse(
                error=error_message,
                extraction_result=ExtractionResult(
                    error=error_message,
                    has_policy=False,
                    policy_count=0
                )
            )
    
    async def process_document(self, file_path: str) -> List[Policy]:
        """Process a complete document using smart chunking and extract policies"""
        start_time = time.time()
        logger.info(f"\n===== Processing document: {file_path} =====")
        
        # Extract document information
        filename = os.path.basename(file_path)
        folder_path = os.path.dirname(file_path)
        
        # Create output directories
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        chunks_dir = os.path.join(self.config.OUTPUT_DIR, self.config.CHUNKS_DIR)
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Step 1: Create smart chunks based on policy boundaries
        logger.info("Creating smart chunks from document...")
        chunk_start_time = time.time()
        chunks = create_smart_chunks(
            file_path, 
            max_tokens=self.config.MAX_TOKENS - self.config.SYSTEM_OVERHEAD - 5000  # Extra safety margin
        )
        
        if not chunks:
            logger.warning("No policy chunks created. Document may not contain policies or smart chunking failed.")
            # Handle case with no chunking
            # TODO: Implement fallback strategy
            return []
        
        # Save chunks for debugging/review
        save_chunks_to_disk(chunks, chunks_dir)
        
        chunk_time = time.time() - chunk_start_time
        logger.info(f"Created {len(chunks)} smart chunks from document in {chunk_time:.2f} seconds")
        
        # Step 2: Process each chunk with the AI agent
        all_policies = []
        policy_page_map = {}  # Map policy names to page ranges
        successful_chunks = 0
        failed_chunks = 0
        
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            logger.info(f"\n===== Processing chunk {i+1}/{len(chunks)} - {chunk['policy_name']} =====")
            logger.info(f"Pages: {chunk['pages']}, Tokens: {chunk['tokens']}")
            
            # Prepare source information
            source_info = {
                "filename": filename,
                "folder_path": folder_path,
                "page_numbers": chunk["pages"],
                "chunk_id": chunk["chunk_id"]
            }
            
            # Process the chunk
            logger.info(f"Sending chunk to AI for extraction...")
            try:
                response = await self.process_chunk(chunk, source_info)
                
                # If policies were found, add them to results
                if response.extraction_result and response.extraction_result.has_policy and response.extraction_result.policies:
                    # Update the policy page map
                    for policy_name, page_range in response.extraction_result.policy_page_ranges.items():
                        if policy_name in policy_page_map:
                            policy_page_map[policy_name].extend(page_range)
                            # Remove duplicates and sort
                            policy_page_map[policy_name] = sorted(list(set(policy_page_map[policy_name])))
                        else:
                            policy_page_map[policy_name] = page_range
                    
                    for policy in response.extraction_result.policies:
                        # Get the pages for this policy
                        policy_pages = response.extraction_result.policy_page_ranges.get(
                            policy.policy_name, 
                            chunk["pages"]
                        )
                        
                        # Add source information with specific pages for this policy
                        policy.source = DocumentSource(
                            filename=filename,
                            folder_path=folder_path,
                            page_numbers=policy_pages
                        )
                        
                        # Only add policies with a name
                        if policy.policy_name and policy.policy_name.strip():
                            all_policies.append(policy)
                            logger.info(f"✓ Extracted policy: {policy.policy_name}")
                            logger.info(f"  Found on pages: {policy_pages}")
                            # Log fields extracted
                            filled_fields = [field for field, value in policy.get_valid_fields().items() 
                                           if value is not None and value != ""]
                            logger.info(f"  Fields extracted: {', '.join(filled_fields)}")
                        else:
                            logger.warning(f"✗ Policy detected but name was missing")
                    
                    successful_chunks += 1
                else:
                    if response.error:
                        logger.warning(f"Error in chunk {i+1}: {response.error}")
                        failed_chunks += 1
                    else:
                        logger.warning(f"No policies detected in chunk {i+1}")
                        successful_chunks += 1  # Count as successful even if no policies found
                
                chunk_time = time.time() - chunk_start_time
                logger.info(f"Completed chunk {i+1}/{len(chunks)} in {chunk_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Unhandled exception processing chunk {i+1}: {str(e)}")
                traceback.print_exc()
                failed_chunks += 1
        
        # Step 3: Deduplicate and merge policies
        logger.info("\nDeduplicating and merging policies...")
        merged_policies = self.deduplicate_policies(all_policies)
        
        # Update page ranges from the policy page map
        for policy in merged_policies:
            if policy.policy_name in policy_page_map:
                policy.source.page_numbers = policy_page_map[policy.policy_name]
        
        total_time = time.time() - start_time
        logger.info(f"Document processing complete in {total_time:.2f} seconds")
        logger.info(f"Chunks processed: {len(chunks)}, Successful: {successful_chunks}, Failed: {failed_chunks}")
        logger.info(f"Final result: {len(merged_policies)} unique policies extracted from {len(all_policies)} raw extractions")
        
        return merged_policies
    
    def deduplicate_policies(self, policies: List[Policy]) -> List[Policy]:
        """Deduplicate policies and merge information from duplicates"""
        # Use a dictionary to track unique policies by name
        unique_policies = {}
        not_policy_entries = []  # Special handling for "NOT A POLICY" entries
        
        for policy in policies:
            # Handle "NOT A POLICY" entries separately
            if policy.policy_name == "NOT A POLICY":
                if policy.source and policy.source.filename:
                    key = f"NOT_A_POLICY_{policy.source.filename}"
                    not_policy_entries.append((key, policy))
                continue
            
            # Normalize policy name for comparison
            norm_name = policy.policy_name.lower().strip()
            
            if norm_name not in unique_policies:
                # First time seeing this policy
                unique_policies[norm_name] = policy
            else:
                # We already have this policy, merge information
                existing = unique_policies[norm_name]
                
                # Merge source page numbers
                if policy.source and existing.source:
                    existing.source.page_numbers = sorted(list(set(
                        existing.source.page_numbers + policy.source.page_numbers
                    )))
                
                # For each field, prefer non-empty values
                for field_name, field_value in policy.get_valid_fields().items():
                    existing_value = getattr(existing, field_name, None)
                    if not existing_value and field_value:
                        setattr(existing, field_name, field_value)
                    elif existing_value and field_value:
                        # Check if values are strings before comparing lengths
                        if isinstance(field_value, str) and isinstance(existing_value, str):
                            if len(field_value) > len(existing_value):
                                # If both have string values but new one is more detailed, prefer it
                                setattr(existing, field_name, field_value)
                        # For numeric values like confidence_score (float), prefer the higher value
                        elif isinstance(field_value, (int, float)) and isinstance(existing_value, (int, float)):
                            if field_value > existing_value:
                                setattr(existing, field_name, field_value)
        
        # Add "NOT A POLICY" entries to the result
        not_policy_dict = dict(not_policy_entries)
        
        # Combine regular policies with "NOT A POLICY" entries
        result = list(unique_policies.values()) + list(not_policy_dict.values())
        
        return result
    
    def save_to_csv(self, policies: List[Policy], output_path: str) -> None:
        """Save policies to CSV file"""
        if not policies:
            logger.warning("No policies to save.")
            return
        
        try:
            # Get all fields from the Policy model
            all_fields = list(policies[0].model_dump().keys())
            
            # Prepare field names for CSV
            field_names = []
            source_fields = ['source_filename', 'source_folder', 'source_pages']
            
            # Add all non-source fields
            for field in all_fields:
                if field != 'source':
                    field_names.append(field)
            
            # Add source fields at the end
            field_names.extend(source_fields)
            
            # Write to CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
                
                # Write each policy
                for policy in policies:
                    # Convert policy to dict for CSV writing
                    policy_dict = policy.model_dump()
                    
                    # Handle nested source field
                    if policy.source:
                        # Extract source fields
                        policy_dict['source_filename'] = policy.source.filename
                        policy_dict['source_folder'] = policy.source.folder_path
                        policy_dict['source_pages'] = ', '.join(map(str, policy.source.page_numbers))
                    else:
                        policy_dict['source_filename'] = ""
                        policy_dict['source_folder'] = ""
                        policy_dict['source_pages'] = ""
                    
                    # Remove the original source field
                    if 'source' in policy_dict:
                        del policy_dict['source']
                    
                    writer.writerow(policy_dict)
            
            logger.info(f"✓ Saved {len(policies)} policies to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            traceback.print_exc()
    
    def save_to_excel(self, policies: List[Policy], output_path: str) -> None:
        """Save policies to Excel file for better handling of Arabic text"""
        if not policies:
            logger.warning("No policies to save.")
            return
        
        try:
            # Create a list of dictionaries for pandas DataFrame
            policies_data = []
            
            for policy in policies:
                # Convert policy to dict
                policy_dict = policy.model_dump()
                
                # Handle nested source field
                if policy.source:
                    policy_dict['source_filename'] = policy.source.filename
                    policy_dict['source_folder'] = policy.source.folder_path
                    policy_dict['source_pages'] = ', '.join(map(str, policy.source.page_numbers))
                else:
                    policy_dict['source_filename'] = ""
                    policy_dict['source_folder'] = ""
                    policy_dict['source_pages'] = ""
                
                # Remove the original source field
                if 'source' in policy_dict:
                    del policy_dict['source']
                
                policies_data.append(policy_dict)
            
            # Get all fields from the Policy model
            if policies_data:
                all_fields = list(policies_data[0].keys())
                
                # Prepare field names in order
                field_names = []
                source_fields = ['source_filename', 'source_folder', 'source_pages']
                
                # Add all non-source fields
                for field in all_fields:
                    if field not in source_fields:
                        field_names.append(field)
                
                # Add source fields at the end
                field_names.extend(source_fields)
                
                # Create DataFrame and save to Excel
                df = pd.DataFrame(policies_data)
                
                # Ensure columns are in correct order
                df = df[field_names]
                
                # Save to Excel with UTF-8 encoding for proper Arabic text handling
                df.to_excel(output_path, index=False, engine='openpyxl')
                
                logger.info(f"✓ Saved {len(policies)} policies to {output_path}")
            else:
                logger.warning("No policy data to save to Excel")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {str(e)}")
            traceback.print_exc()
            
            # Fallback to CSV if Excel saving fails
            csv_path = output_path.replace('.xlsx', '.csv')
            logger.info(f"Falling back to CSV format: {csv_path}")
            self.save_to_csv(policies, csv_path)


# Main function for CLI usage
def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract policies from PDF documents with smart chunking')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input PDF file path')
    parser.add_argument('--output', '-o', type=str, default="Output",
                        help='Output directory for results')
    parser.add_argument('--format', '-f', type=str, choices=['csv', 'excel'], default='excel',
                        help='Output format (csv or excel)')
    args = parser.parse_args()
    
    # Validate input path exists
    if not os.path.exists(args.input):
        logger.error(f"Error: Input path '{args.input}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize extractor with config
    config = Config()
    config.OUTPUT_DIR = args.output
    extractor = PolicyExtractor(config)
    
    # Process the document
    logger.info(f"Processing document: {args.input}")
    
    # Run the async process_document in an event loop
    import asyncio
    
    async def async_main():
        policies = await extractor.process_document(args.input)
        
        # Save the results
        if policies:
            if args.format == 'excel':
                output_path = os.path.join(args.output, "extracted_policies.xlsx")
                extractor.save_to_excel(policies, output_path)
            else:
                output_path = os.path.join(args.output, "extracted_policies.csv")
                extractor.save_to_csv(policies, output_path)
            
            logger.info(f"\nExtraction complete. Saved {len(policies)} policies to {output_path}")
        else:
            logger.warning("\nNo policies were extracted from the document.")
    
    # Run the async main function
    try:
        asyncio.run(async_main())
    except RuntimeError as e:
        # Handle case where we might already be in an event loop
        logger.warning(f"Error with asyncio.run(): {e}")
        loop = asyncio.get_event_loop()
        if loop.is_running():
            logger.info("Using existing running event loop")
            future = asyncio.ensure_future(async_main())
            # Wait until it's done
            while not future.done():
                time.sleep(0.1)
        else:
            logger.info("Using existing event loop")
            loop.run_until_complete(async_main())


if __name__ == "__main__":
    main() 