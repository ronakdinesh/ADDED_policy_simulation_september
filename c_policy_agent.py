#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy JSON Processor

A tool for extracting structured policy information from policy JSON files
output by b_smart_chunking_agent.py using Azure OpenAI and Pydantic.

This script:
1. Processes policy JSON files created by b_smart_chunking_agent.py
2. Uses Azure OpenAI and Pydantic AI to extract structured information from each policy
3. Combines results into a coherent Excel output

Usage:
    python policy_json_processor.py --input <input_directory> --output <output_file.xlsx>
"""

import os
import time
import logging
import argparse
import pandas as pd
import traceback
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio
import httpx
import orjson
import random
from glob import glob

# OpenAI dependencies
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field, ConfigDict, model_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
# Load environment variables
load_dotenv()

# Configure logging
OUTPUT_DIR = "Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'policy_processing.log'))
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


# Policy model
class Policy(BaseModel):
    """Pydantic model for policy information extraction"""
    # Main identification fields
    policy_name: str = Field(description="The official name of the policy as it appears in the document. Look for standard naming formats such as 'Law No. (XX) of Year YYYY concerning...', 'Chairman of the Executive Council Resolution No. (XX) of Year YYYY...', 'Executive Council Resolution No. (XX)...', 'Amiri Decree No. (XX)...', or 'Circular No. (XX)...'")
    issuing_year: Optional[str] = Field(None, description="The year the policy was officially issued. Extract this information either from the policy name (e.g., 'Law No. (XX) of Year 2020') or from the publication date found within the document. Format as a 4-digit year (e.g., '2020').")
    relevance: Optional[str] = Field(None, description="Label as 'relevant' if this is an economic policy according to the ECONOMIC POLICY IDENTIFICATION GUIDELINES, or 'not relevant' if it's not an economic policy.")
    relevance_rationale: Optional[str] = Field(None, description="Provide a clear explanation of why this document was classified as 'relevant' or 'not relevant'. For relevant policies, explain which economic components (government spending, investment, consumption, net exports) it affects. For non-relevant policies, explain why it doesn't qualify as an economic policy (e.g., purely governance, social without economic objectives, etc.).")
    has_tables: Optional[str] = Field(None, description="""Answer "Yes" only if the policy contains tables with financial or numerical data (e.g., budgets, statistics, metrics, quantitative targets). Answer "No" if:
    1. The document has no tables at all
    2. The document has tables but they only contain text or qualitative information
    3. The tables are purely organizational/structural without meaningful numerical data
    Remember to check the actual content of tables, not just their presence.""")
    
    # Content fields
    policy_summary: Optional[str] = Field(None, description="""Create a comprehensive summary of this policy document using ONLY terminology and concepts explicitly mentioned in the text.
    1. Do not introduce new terms or infer information not directly stated
    2. Include: 
    - The exact name and number of the policy
    - Direct quotes of key purposes and objectives
    - Specific bodies/committees established (with exact names)
    - Explicitly stated implementation mechanisms
    3. Be thorough in reviewing the entire document to create a complete summary
    4. Use 3-5 sentences while maintaining substantive content
    5. If the document has an executive summary or introduction section, pay special attention to these areas""")

    goals: Optional[str] = Field(None, description="""Exhaustively extract ALL high-level qualitative aims explicitly stated in the policy. 
    1. Search through the entire document for any statements of intent, purpose, or long-term aspirations
    2. Pay special attention to sections with headers like "Goals", "Aims", "Vision", "Mission", or similar
    3. Include direct quotes where possible and maintain the original terminology
    4. Format as a numbered or bulleted list with each goal on a new line
    5. Look for aspirational statements that describe desired outcomes without specific metrics
    6. Extract ALL stated goals, even if they appear in different sections of the document
    7. If no goals are explicitly stated, indicate "No explicit goals stated in the document"
    8. Do not include objectives that have specific targets or metrics (these belong in the objectives field)""")

    objectives: Optional[str] = Field(None, description="""Extract ALL specific, actionable, quantitative targets explicitly stated in the policy document. 
    1. Exhaustively search through the entire document for statements that include concrete, measurable targets or specific actions to be implemented
    2. Pay special attention to sections with headers like "Objectives", "Targets", "Implementation Plan", or similar
    3. Include numeric targets, percentages, timelines, and specific deliverables
    4. Format as a numbered or bulleted list with each objective on a new line
    5. Use direct quotes or closely paraphrased text to maintain the exact meaning
    6. Do not include general aspirational statements without specific metrics (these belong in the goals field)
    7. If no specific objectives are found, indicate "No specific objectives stated in the document"
    8. Be thorough - objectives may be scattered throughout different sections of the document""")
    
    situation_analysis: Optional[str] = Field(None, description="""Analyze the context and circumstances that would have led to this policy being drafted. Consider:
    1. What problems or opportunities was this policy responding to?
    2. What was the economic, social, or political situation that necessitated this policy?
    3. What broader goals and objectives was this policy trying to achieve beyond its explicitly stated aims?
    You may need to make informed inferences based on the content, but ground your analysis in the text. If information cannot be determined, provide an empty string.""")
    
    # Classification fields
    legislative_instrument: Optional[str] = Field(None, description="""The legal tool used to enforce the policy. 
    Must be one of the following options only:
    1. UAE Constitution: The highest source of all legislation in the country, establishing the general foundations of the legal and legislative system
    2. Federal Law: Issued by the Federal National Council and ratified by the President of the State
    3. Decree by Law: Issued directly by the President of the State in urgent cases that do not allow for delays, and they carry the force of law
    4. Federal Regulation and Decision: Issued by the Cabinet or relevant federal regulatory authorities to implement and interpret federal laws
    5. Local Law of the Emirate of Abu Dhabi: Issued by the Ruler of the Emirate to regulate local matters, provided they do not conflict with federal laws
    6. Amiri Decree: Issued by the Ruler of Abu Dhabi to address specific matters within the Emirate
    7. Executive Council Decisions for the Emirate of Abu Dhabi: Issued by the Executive Council, addressing public policies and executive procedures within the Emirate
    8. Executive Regulation and Local Regulatory Decision: Issued by relevant government entities to regulate economic and administrative activities""")
    
    federal_emirate: Optional[str] = Field(None, description="""Whether the policy is developed on the federal level or just the Emirate of Abu Dhabi's level.
    Must be classified as either 'Federal' or 'Emirate'.""")
    
    region: Optional[str] = Field(None, description="""The geographical area covered by the policy.
    Must be one of the following options:
    1. All (if policy refers to entire Emirate of Abu Dhabi or doesn't specify a region)
    2. Abu Dhabi City
    3. Al Ain
    4. Al Dhafra
    5. Abu Dhabi City & Al Ain
    6. Al Ain & Al Dhafra
    7. Abu Dhabi City & Al Dhafra

    Guidelines for classification:
    - If the policy explicitly states it applies to a specific geographical region, select that region.
    - If the policy title or name includes a specific regional reference (such as a named entity or facility that is clearly associated with a specific region), classify it for that region.
    - If multiple regions are mentioned in the policy scope, select the appropriate combined region option.
    - Only select "All" if the policy applies to the entire Emirate or doesn't specify regional limitations.""")
    
    jurisdiction: Optional[str] = Field(None, description="""The legal or regulatory scope within which the policy applies.
        Must be one of the following options:
        1. All (Use this if: a) the policy explicitly states it applies to the entire Emirate of Abu Dhabi; OR b) the policy is issued by an emirate-wide authority without specifying geographical limitations; OR c) the policy governs an entity/activity with emirate-wide scope)
        2. Mainland (Use this ONLY if the policy explicitly limits its application to mainland areas or specific mainland regions AND does not have emirate-wide applicability)
        3. Freezone (Use this ONLY if the policy specifically limits its application to freezones or any of the listed freezone areas)
        4. Island (Use this ONLY if the policy specifically limits its application to islands or any of the listed island areas)

        DEFAULT ASSUMPTION: If a policy mentions a location (like Abu Dhabi City) merely as a headquarters or office location, but the policy's applicability is emirate-wide, classify as "All" not "Mainland".

        Examples of mainland areas include: Ain Al Faydah, Al Dhahir, Al Falah, Al Khibeesi, Al Ain, Abu Dhabi City, Al Dhafra, Al Rawdah Al Sharqiyah, Al Shamkhah, Bani Yas, Madinat Al Riyad, Mayzad, Mbazzarah Al Khadra, Mohamed Bin Zayed City, Shakhbout City, Shi'bat Al Wutah, Abu Dhabi Island

        Examples of freezone areas include: Abu Dhabi Global Market (ADGM), Khalifa Economic Zones Abu Dhabi (KEZAD), Masdar City Free Zone, Twofour54, Industrial City of Abu Dhabi (ICAD), Higher Corporation for Specialized Economic Zones (ZonesCorp), Abu Dhabi Airport Free Zone (ADAFZ)

        Examples of island areas include: Al Khalidiyah, Al Manhal, Hadbat Al Za'faranah""")
    
    sector: Optional[str] = Field(None, description="""The sector this policy affects.
        Must be one of the following options or "Others":
        1. Multi-sector (if it affects more than one sector)
        2. Mining and Quarrying
        3. Wholesale and Retail Trade
        4. Manufacturing
        5. Financial and Insurance Services
        6. Construction
        7. Real Estate Activities
        8. Public Administration and Defense
        9. Transportation and Storage
        10. Professional, Scientific and Technical Activities and Administrative and Support Services
        11. Information and Communication
        12. Electricity, gas, and Water Supply; Waste Management Activities
        13. Accommodation and Food Service Activities
        14. Education
        15. Human Health and Social work Activities
        16. Agriculture, Forestry and Fishing
        17. Activities of Households as Employers
        18. Arts, Recreation and Other Service Activities""")
    
    subsector: Optional[str] = Field(None, description="""The subsector this policy affects. 
        Must align with the selected sector from above:
        - Mining and Quarrying: Oil and gas extraction; Mining, except oil and gas; Mining support service activities
        - Wholesale and Retail Trade: Motor trades; Wholesale trade; Retail trade
        - Manufacturing: Manufacture of food, beverages and tobacco; Manufacture of textiles, wearing apparel and leather; Manufacture of wood and paper products and printing; Manufacture of coke, refined petroleum and chemicals; Manufacture of pharmaceutical products; Manufacture of rubber, plastic and non-metallic minerals; Manufacture of basic and fabricated metal products; Manufacture of computer, electronic and optical products; Manufacture of electrical equipment; Manufacture of machinery and equipment; Manufacture of transport equipment; Other manufacturing; Repair and installation of machinery and equipment
        - Financial and Insurance Services: Financial service activities; Insurance and pension funding; Activities auxiliary to finance and insurance
        - Construction: Construction of buildings and civil engineering services; Specialized construction
        - Real Estate Activities: Owner-occupiers' imputed rental; Real estate activities, excluding imputed rental
        - Public Administration and Defence: Public administration and defence
        - Transportation and Storage: Transportation - land, water and air transport; Warehousing, and postal and courier activities
        - Professional/Scientific/Technical: Legal and accounting activities; Head offices and management consultancy; Architectural and engineering activities; Scientific R&D; Advertising/market research; Other professional/scientific; Veterinary; Rental/leasing; Employment; Travel agency; Security/investigation; Building services; Office administration
        - Information and Communication: Media activities; Telecommunications; Information technology services
        - Utilities and Waste: Electricity, gas, steam and air conditioning supply; Water supply and sewerage; Waste collection, treatment and disposal, and waste management services
        - Accommodation and Food: Accommodation; Food and beverage service activities
        - Education: Education
        - Health and Social Work: Human health activities; Residential care and social work activities
        - Agriculture, Forestry and Fishing: Agriculture and hunting, and forestry and logging; Fishing and aquaculture
        - Activities of Households: Activities of households
        - Arts, Recreation and Services: Arts, entertainment and recreation; Other service activities""")
    
    # Performance and ownership fields
    metrics: Optional[str] = Field(None, description="Extract ONLY the key performance indicators (KPIs) or metrics that are EXPLICITLY mentioned in the document. Look for phrases like 'key performance indicators', 'KPIs', 'performance metrics', 'success measures', 'performance targets', 'impact metrics'. Do NOT ideate, create, or infer metrics that aren't directly stated in the text. If no metrics are explicitly mentioned, leave this field NULL.")
    policy_owner: Optional[str] = Field(None, description="""The government entity that owns the policy. 
        IMPORTANT: The policy owner MUST be selected ONLY from this list of entities: (If High confidence, select from this list, if not, leave "Others")
        - Abu Dhabi Department of Economic Development (ADDED/DED)
        - Abu Dhabi Investment Office (ADIO)
        - Department of Municipalities and Transport (DMT)
        - Abu Dhabi National Oil Company (ADNOC)
        - Abu Dhabi Pension Fund (ADPF)
        - Abu Dhabi Department of Health (AD DoH)
        - Abu Dhabi Executive Office (ADEO)/ Executive Council
        - Supreme Council for Financial and Economic Affairs (SCFEA)
        - Abu Dhabi Accountability Authority (ADAA)
        - Abu Dhabi Centre for Projects and Infrastructure (ADPIC)
        - Creative Media Authority
        - Abu Dhabi Developmental Holding Company (ADQ)
        - Abu Dhabi Judicial Department
        - Abu Dhabi Media Office
        - Abu Dhabi Sports Council
        - Department of Government Support
        - Al Ain Farms for Livestock Production
        - Abu Dhabi Securities Exchange Company (PJSC)
        - Abu Dhabi Human Resources Authority
        - Advanced Technology Research Council (ATRC)
        - Abu Dhabi Ports Company (AD Ports)
        - Abu Dhabi Department of Education and Knowledge (ADEK)
        - Abu Dhabi Department of Energy (AD DoE)
        - Endowments and Minors' Funds Management Authority
        - Mubadala
        - Musanada
        - Abu Dhabi Global Market (ADGM)
        - Abu Dhabi Police (AD Police) / Abu Dhabi Police General Headquarters
        - Environment Agency Abu Dhabi (EAD)
        - Sorbonne University Abu Dhabi
        - Emirates College for Advanced Education
        - Khalifa University for Science and Technology
        - Abu Dhabi Department of Finance (AD DoF)
        - International Fund for Houbara Conservation
        - Ruler's Representative Court - Al-Dhafra Region
        - Rabdan Academy
        - Dar Zayed For Family Care
        - Mohammed V University - Abu Dhabi
        - Abu Dhabi Center for Arabic Language
        - Abu Dhabi Department of Community Development (DCD)
        - Emirates Nuclear Energy Corporation
        - Abu Dhabi Housing Authority
        - Zayed Higher Organization for People of Determination
        - Zayed bin Sultan Al Nahyan Charitable & Humanitarian Foundation
        - Khalifa Fund for Enterprise Development (KFED)
        - National Rehabilitation Centre - Abu Dhabi
        - Abu Dhabi Digital Authority
        - Abu Dhabi Centre for Sheltering & Humanitarian Care (EWAA)
        - Abu Dhabi Civil Defense Authority
        - Integrated Transport Centre (ITC)
        - Abu Dhabi Equestrian Club
        - Mohamed bin Zayed University for Humanities
        - Mohamed bin Zayed University of Artificial Intelligence
        - Al Ain City Municipality
        - Crown Prince Court
        - Abu Dhabi Centre for Legal and Community Awareness
        - Qas Al Emarat Company - Abu Dhabi
        - Abu Dhabi Early Childhood Authority
        - Department of Government Enablement
        - Abu Dhabi Fund for Development
        - Abu Dhabi Public Health Centre
        - Al Ain Company
        - Statistics Centre Abu Dhabi
        - Etihad Airways Group
        - Modon Properties Company
        - Emirates Foundation
        - Abu Dhabi Chamber of Commerce and Industry (ADCCI)
        - AD Customs
        - Zoo & Aquarium Public Institution in Al Ain
        - Family Care Authority
        - Department of Culture and Tourism (DCT)
        - KEZAD Group
        - Ministry of Economy
        - Immigration Authority
        - Ministry of Human Resources and Emiratisation
        - Central Bank

Do NOT create or infer policy owners that are not on this list. If the policy owner is not explicitly mentioned or cannot be determined with high confidence, leave this field NULL.""")
    stakeholders_involved: Optional[str] = Field(None, description="Government entities that play a role in developing the policy but do not have primary ownership or responsibility")
    thematic_categorization: Optional[str] = Field(None, description="""The broad theme this policy belongs to (single-select).
Must be EXACTLY one of:
- Economic Growth
- Clusters activation & growth
- Localization & regional development
- Enablers
- Business Environment Support (BES)
- Government Optimization

Rules:
- Assign only ONE category.
- Use the policy title, objectives, or legal instrument to guide classification.
- If a policy relates to multiple areas, choose the most explicit or dominant one.
""")
    thematic_sub_categorization: Optional[str] = Field(None, description="""The specific sub-theme under the selected Thematic Categorization (single-select).
Allowed values depend on the chosen category (sub is a subset of category):
- Enablers:
  - Financial & non-financial incentives (grants, fee waivers, subsidies, exemptions, advisory support)
  - Employment & workforce productivity (upskilling, training, Emiratization, workforce programs)
  - Innovation & digital transformation (R&D, AI adoption, digitization, new technologies)
  - Physical infrastructure development (utilities, logistics, industrial zones, transport, broadband)
- Business Environment Support (BES):
  - Business regulation & process reform (licensing, permits, regulatory simplification, inspection reform)
  - SME & entrepreneurship development (SME programs, incubators, accelerators, financing, market access)
- Government Optimization:
  - Government fees & charges (adjustments, reductions, or introductions of fees and tariffs)
  - Government efficiency (process optimization, performance management, shared services, digital government)
- Economic Growth: General
- Clusters activation & growth: General
- Localization & regional development: General

Strict rules:
- Assign exactly ONE sub-categorization.
- The sub-categorization MUST be a valid subset under the selected category.
- When the category has no explicit sub-themes, use "General".
- Prioritize clear keyword matches (e.g., “incentive” → Enablers > Financial & non-financial incentives; “fee reduction” → Government Optimization > Government fees & charges).
""")
    
    # Source tracking (removed confidence_score)
    source: Optional[DocumentSource] = None
    extraction_date: str = Field(None, description="Date when the policy was extracted")
    processing_status: str = Field(default="fully_processed", description="Whether the policy was fully processed or truncated due to token limits")
    
    # Full text of the policy
    full_text: Optional[str] = Field(None, description="The complete text of the policy document")
    
    # Helper methods
    def get_valid_fields(self) -> Dict[str, Any]:
        """Return only fields that have values (not None)"""
        return {k: v for k, v in self.model_dump().items() 
                if k != 'source' and v is not None and v != ""}
    
    model_config = ConfigDict(
        extra="ignore",
        protected_namespaces=()
    )

    # Thematic subset validation map
    THEMATIC_MAP: Dict[str, List[str]] = {
        "Economic Growth": ["General"],
        "Clusters activation & growth": ["General"],
        "Localization & regional development": ["General"],
        "Enablers": [
            "Financial & non-financial incentives",
            "Employment & workforce productivity",
            "Innovation & digital transformation",
            "Physical infrastructure development",
        ],
        "Business Environment Support (BES)": [
            "Business regulation & process reform",
            "SME & entrepreneurship development",
        ],
        "Government Optimization": [
            "Government fees & charges",
            "Government efficiency",
        ],
    }

    @model_validator(mode="after")
    def _validate_thematic_subset(self):
        """Ensure sub-categorization is a subset of the chosen category; fallback to General if available."""
        category = self.thematic_categorization
        subcategory = getattr(self, 'thematic_sub_categorization', None)
        if category and subcategory:
            allowed = self.THEMATIC_MAP.get(category)
            if not allowed or subcategory not in allowed:
                if allowed and "General" in allowed:
                    self.thematic_sub_categorization = "General"
                else:
                    self.thematic_sub_categorization = ""
        return self


# Agent response models
class ExtractionResult(BaseModel):
    """Model for extraction results"""
    policies: List[Policy] = Field(default_factory=list, description="List of policies found in the document")
    error: Optional[str] = None
    has_policy: bool = Field(description="Whether a policy was found")
    policy_count: int = Field(default=0, description="Number of policies found")


class AgentResponse(BaseModel):
    """Main agent response model"""
    message: Optional[str] = Field(None, description="Human-readable message")
    error: Optional[str] = Field(None, description="Error message if any")
    extraction_result: Optional[ExtractionResult] = Field(None, description="Extraction results")
    suggested_next_action: Optional[str] = Field(None, description="Suggested next action")


class Config:
    """Configuration for the Policy JSON Processor"""
    # Azure OpenAI settings - use values from .env
    AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://admin-0752-resource.cognitiveservices.azure.com/") 
    AZURE_OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", "2024-12-01-preview")
    AZURE_OPENAI_DEPLOYMENT = os.environ.get("OPENAI_MODEL_NAME", "gpt-5")

    # Processing settings
    MAX_TOKENS = 50000  # Maximum context window for GPT-4o
    SYSTEM_OVERHEAD = 15000  # Estimate for system prompt overhead
    MAX_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "8"))  # Optimized concurrency
    
    # Timeout settings from environment (optimized for speed)
    HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "180"))
    AGENT_TIMEOUT = int(os.environ.get("AGENT_TIMEOUT", "300"))
    FILE_TIMEOUT = int(os.environ.get("FILE_TIMEOUT", "450"))
    MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))
    RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "30"))


class TrackingFileManager:
    """Manages tracking file operations and validation"""
    
    def __init__(self, tracking_path: str):
        self.tracking_path = tracking_path
        self.required_columns = ["Policy ID", "Policy Name", "Relative Path"]
        self.tracking_columns = ["processed", "processed_date", "extraction_status", "extraction_notes"]
    
    def load_tracking_file(self) -> Optional[pd.DataFrame]:
        """Load and validate tracking file"""
        try:
            df = pd.read_excel(self.tracking_path, engine='openpyxl')
            logger.info(f"Loaded tracking file with {len(df)} entries")
            
            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None
            
            # Add tracking columns if missing
            for col in self.tracking_columns:
                if col not in df.columns:
                    df[col] = ""
                    logger.info(f"Added '{col}' column")
            
            return df
        except Exception as e:
            logger.error(f"Failed to load tracking file: {e}")
            return None
    
    def create_backup(self, df: pd.DataFrame) -> str:
        """Create backup of tracking file"""
        backup_path = self.tracking_path.replace('.xlsx', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        df.to_excel(backup_path, index=False, engine='openpyxl')
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    
    def save_tracking_file(self, df: pd.DataFrame) -> None:
        """Save updated tracking file"""
        df.to_excel(self.tracking_path, index=False, engine='openpyxl')
        logger.info(f"Updated tracking file: {self.tracking_path}")
    
    def filter_policies_to_process(self, df: pd.DataFrame, process_all: bool = False, limit: int = None) -> pd.DataFrame:
        """Filter policies based on processing criteria"""
        if process_all:
            policies_to_process = df
        else:
            policies_to_process = df[df["processed"] != "Yes"]
        
        if limit and limit > 0:
            original_count = len(policies_to_process)
            
            # Prioritize never processed, then failed
            never_processed = policies_to_process[policies_to_process["processed"] == ""]
            failed_processed = policies_to_process[policies_to_process["processed"] == "Failed"]
            other_unprocessed = policies_to_process[
                ~policies_to_process.index.isin(never_processed.index) & 
                ~policies_to_process.index.isin(failed_processed.index)
            ]
            
            # Combine in priority order and apply limit
            prioritized = pd.concat([never_processed, failed_processed, other_unprocessed])
            policies_to_process = prioritized.iloc[:limit]
            
            logger.info(f"Limited to {len(policies_to_process)}/{original_count} policies (limit={limit})")
            logger.info(f"Prioritized: {len(never_processed)} never processed, {len(failed_processed)} failed")
        
        return policies_to_process


class FilePathResolver:
    """Resolves policy file paths with fallback mechanisms"""
    
    @staticmethod
    def find_matching_policy_file(original_path: str) -> str:
        """Find matching policy file with fallback mechanisms"""
        if os.path.exists(original_path):
            return original_path
        
        try:
            dirname = os.path.dirname(original_path)
            filename = os.path.basename(original_path)
            
            if not os.path.exists(dirname):
                return original_path
            
            # Try policy ID prefix matching
            parts = filename.split('_', 2)
            if len(parts) >= 3 and parts[0] == "policy":
                prefix = f"{parts[0]}_{parts[1]}_"
                
                for file in os.listdir(dirname):
                    if file.startswith(prefix):
                        logger.info(f"Found alternative: {file} instead of {filename}")
                        return os.path.join(dirname, file)
            
            # Try file prefix matching
            name_parts = filename.split('.')
            if len(name_parts) >= 2:
                file_prefix = name_parts[0].rsplit('_', 1)[0]
                matching_files = [f for f in os.listdir(dirname) 
                                if f.startswith(file_prefix) and f.endswith('.json')]
                if matching_files:
                    logger.info(f"Found prefix match: {matching_files[0]} instead of {filename}")
                    return os.path.join(dirname, matching_files[0])
                    
        except Exception as e:
            logger.warning(f"Error finding alternative file: {e}")
        
        return original_path


class PolicyJsonProcessor:
    """Process policy JSON files and extract structured information"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None  # Will be set up in async context
        self.model = None   # Will be set up in async context
        self.agent = None   # Will be set up in async context
        
        # Semaphores for better concurrency control
        self.api_semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
        self.rate_limit_semaphore = asyncio.Semaphore(10)  # For rate limit handling
        
        # Checkpointing for resume capability
        self.checkpoint_file = Path("output/processing_checkpoint.json")
        self.processed_files = set()
        self.failed_files = set()
        self.load_checkpoint()
    
    async def setup_openai_client(self, deterministic_seed: Optional[int] = 42):
        """Set up the Azure OpenAI client with proper async context management"""
        logger.info(f"Connecting to Azure OpenAI at: {self.config.AZURE_OPENAI_ENDPOINT}")
        logger.info(f"Using deployment: {self.config.AZURE_OPENAI_DEPLOYMENT}")
        logger.info(f"API Version: {self.config.AZURE_OPENAI_API_VERSION}")
        
        # Create HTTP client with connection pooling for better performance
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=15,  # Reduced for stability
                max_keepalive_connections=8  # Reduced for stability
            ),
            timeout=httpx.Timeout(self.config.HTTP_TIMEOUT, connect=30.0)  # Increased timeouts
        )
        
        self.client = AsyncAzureOpenAI(
            api_key=self.config.AZURE_OPENAI_KEY,
            api_version=self.config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
            max_retries=5,  # Increased retries
            http_client=http_client
        )
        
        # Set up OpenAI model for Pydantic AI
        self.model = OpenAIModel(
            self.config.AZURE_OPENAI_DEPLOYMENT,
            provider='azure'
        )
        
        # Model settings with deterministic seed
        self.model_settings = {'seed': deterministic_seed} if deterministic_seed else {}
        
        return http_client  # Return for context management
    
    async def cleanup_client(self):
        """Properly close HTTP connections"""
        if hasattr(self, 'client') and self.client and hasattr(self.client, '_client'):
            await self.client._client.aclose()
            logger.info("HTTP client connections closed")
    
    def load_checkpoint(self):
        """Load processing checkpoint to resume from failures"""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = orjson.loads(f.read())
                    self.processed_files = set(checkpoint_data.get('processed', []))
                    self.failed_files = set(checkpoint_data.get('failed', []))
                    logger.info(f"Loaded checkpoint: {len(self.processed_files)} processed, {len(self.failed_files)} failed")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            self.processed_files = set()
            self.failed_files = set()
    
    def save_checkpoint(self):
        """Save current processing state"""
        try:
            # Ensure output directory exists
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint_data = {
                'processed': list(self.processed_files),
                'failed': list(self.failed_files),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.checkpoint_file, 'wb') as f:
                f.write(orjson.dumps(checkpoint_data, option=orjson.OPT_INDENT_2))
                
            logger.debug(f"Checkpoint saved: {len(self.processed_files)} processed, {len(self.failed_files)} failed")
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")
    
    def should_skip_file(self, file_path: str) -> bool:
        """Check if file should be skipped based on checkpoint"""
        return file_path in self.processed_files
    
    def mark_file_processed(self, file_path: str):
        """Mark file as successfully processed"""
        self.processed_files.add(file_path)
        if file_path in self.failed_files:
            self.failed_files.remove(file_path)
    
    def mark_file_failed(self, file_path: str):
        """Mark file as failed"""
        self.failed_files.add(file_path)
        if file_path in self.processed_files:
            self.processed_files.remove(file_path)
    
    def setup_agent(self) -> None:
        """Set up the AI agent with system prompt"""
        # System prompt with detailed instructions
        self.system_prompt = """
You are an expert policy analyst specializing in economic policy extraction and analysis. Your task is to extract structured information about the economic policy from the provided document.

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

## DOCUMENT PROCESSING INSTRUCTIONS:
1. Carefully read the entire document to understand its content and structure.
2. Determine if this document contains an economic policy based on the guidelines above.
3. IMPORTANT: Never return NULL values for any field. Instead:
   - Use "Others" for categorical fields with predefined options when none clearly apply
   - Use empty string "" for free-text fields when information cannot be determined
   - Use "All" for region when no specific region is mentioned
   - Use "All" for jurisdiction when no specific legal scope is mentioned
   - For has_tables, use "Yes" ONLY if tables contain financial or numerical data, not just because tables exist
4. Provide a clear rationale for your relevance classification, explaining why this policy is economically relevant or not.
5. For Arabic text, provide both the original Arabic text and an English translation for important fields.
6. IMPORTANT: Even when a policy is classified as "not relevant", you MUST still extract and provide information for ALL fields:
   - policy_summary, goals, objectives, and situation_analysis should still be populated based on the document content
   - sector and thematic_categorization (broad parent) and thematic_sub_categorization (subset of the parent) should be determined based on the document's focus area
   - policy_owner should be identified if possible
   Do not skip any fields just because the policy is not economically relevant.

## PARAMETER DEFINITIONS:
Refer to the field descriptions provided in the Policy model definition for detailed guidance on each parameter. For all fields:
- Extract information directly from the document where possible
- Follow the specific options and formatting requirements for each field
- Make informed inferences only when necessary and ground them in the text
- Provide meaningful content for each field or use the appropriate default value as specified above

## THEMATIC CLASSIFICATION TAXONOMY (JSON contract):
Return the thematic classification as JSON ONLY (no prose) with the keys below. The sub-category must be a valid subset of the selected category.

{
  "thematic_categorization": "<one_of>",
  "thematic_sub_categorization": "<subset_of_thematic_categorization>"
}

Allowed values:
{
  "Economic Growth": ["General"],
  "Clusters activation & growth": ["General"],
  "Localization & regional development": ["General"],
  "Enablers": [
    "Financial & non-financial incentives",
    "Employment & workforce productivity",
    "Innovation & digital transformation",
    "Physical infrastructure development"
  ],
  "Business Environment Support (BES)": [
    "Business regulation & process reform",
    "SME & entrepreneurship development"
  ],
  "Government Optimization": [
    "Government fees & charges",
    "Government efficiency"
  ]
}

Assignment rules:
- Assign exactly one category and one sub-category.
- Use the policy title, objectives, or legal instrument to guide classification.
- If a policy relates to multiple areas, choose the most explicit or dominant one.
- Keyword cues: “incentive” → Enablers/Financial & non-financial incentives; “fee reduction” → Government Optimization/Government fees & charges.

## OUTPUT FORMAT:
Provide your analysis in a structured JSON format that matches the Policy model definition. Include all required fields and ensure every field has a value - either a meaningful extracted value, "Others", or an empty string "".

Remember to extract factual information present in the document, not to make assumptions or generate content that isn't explicitly stated.
"""
        
        self.agent = Agent(
            self.model,
            output_type=AgentResponse,
            instructions=self.system_prompt
        )
    
    async def process_policy_json(self, json_path: str) -> AgentResponse:
        """Process a single policy JSON file with improved error handling"""
        if not os.path.exists(json_path):
            error_msg = f"Policy JSON file not found: {json_path}"
            logger.error(error_msg)
            return self._create_error_response(error_msg)
        
        try:
            logger.info(f"Processing policy JSON file: {json_path}")
            
            # Load and validate JSON data
            policy_data = await self._load_policy_data(json_path)
            if policy_data is None:
                return self._create_error_response(f"Failed to load policy data from {json_path}")
            
            # Prepare context with metadata
            text_with_context, was_truncated = await self._prepare_context(policy_data)
            
            # Process with AI agent
            return await self._process_with_agent(text_with_context, json_path, was_truncated)
            
        except Exception as e:
            error_msg = f"Error processing policy JSON {json_path}: {str(e)}"
            logger.error(error_msg)
            return self._create_error_response(error_msg)
    
    async def _load_policy_data(self, json_path: str) -> Optional[Dict[str, Any]]:
        """Load and validate policy data from JSON file"""
        try:
            with open(json_path, 'rb') as f:
                policy_data = orjson.loads(f.read())
                
            # Basic validation
            if not isinstance(policy_data, dict):
                logger.error(f"Invalid JSON structure in {json_path}")
                return None
                
            return policy_data
        except (orjson.JSONDecodeError, UnicodeDecodeError, IOError) as e:
            logger.error(f"Failed to load policy data from {json_path}: {e}")
            return None
    
    async def _prepare_context(self, policy_data: Dict[str, Any]) -> tuple[str, bool]:
        """Prepare context with metadata and handle token limits"""
        # Extract metadata safely
        metadata = policy_data.get('metadata', {})
        tables_info = metadata.get('contains_tables', False)
        tables_note = ("Yes, the document contains tables, but you must analyze if they contain financial/numerical data" 
                      if tables_info else "No tables detected in the document")
        
        # Create context header
        context_header = f"""
Policy Document Information:
- Policy Name: {policy_data.get('policy_name', 'Unknown')}
- Arabic Policy Name: {policy_data.get('policy_name_arabic', 'Not provided')}
- Pages: {', '.join(map(str, policy_data.get('page_range', [])))}
- Language: {metadata.get('language', 'Unknown')}
- Year: {metadata.get('year', 'Unknown')}
- Policy Type: {metadata.get('policy_type', 'Unknown')}
- Contains Tables: {tables_note}

Document Content:
==================
"""
        
        # Combine header with content
        text_with_context = context_header + policy_data.get('content', '')
        
        # Handle token limits
        token_count = self.count_tokens(text_with_context)
        available_tokens = self.config.MAX_TOKENS - self.config.SYSTEM_OVERHEAD
        was_truncated = False
        
        if token_count > available_tokens:
            logger.warning(f"Content exceeds context window ({token_count} > {available_tokens} tokens)")
            was_truncated = True
            text_with_context = self.smart_truncate_content(text_with_context, available_tokens - 1000)
            new_token_count = self.count_tokens(text_with_context)
            logger.info(f"Smart truncation: {token_count} -> {new_token_count} tokens")
        
        return text_with_context, was_truncated
    
    async def _process_with_agent(self, text_with_context: str, json_path: str, was_truncated: bool) -> AgentResponse:
        """Process text with AI agent and handle response"""
        logger.info("Starting AI processing...")
        start_time = time.time()
        
        try:
            async def agent_operation():
                async with self.api_semaphore:
                    return await asyncio.wait_for(
                        self.agent.run(text_with_context, model_settings=self.model_settings),
                        timeout=self.config.AGENT_TIMEOUT
                    )
            
            result = await self.api_call_with_retry(agent_operation)
            elapsed_time = time.time() - start_time
            logger.info(f"AI processing completed in {elapsed_time:.2f}s")
            
            return self._process_agent_result(result, json_path, was_truncated)
            
        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            error_msg = f"AI processing timed out after {elapsed_time:.2f}s"
            logger.error(error_msg)
            return self._create_error_response(error_msg)
    
    def _create_error_response(self, error_msg: str) -> AgentResponse:
        """Create standardized error response"""
        return AgentResponse(
            error=error_msg,
            extraction_result=ExtractionResult(
                error=error_msg,
                has_policy=False,
                policy_count=0
            )
        )
    
    def _process_agent_result(self, result, json_path: str, was_truncated: bool) -> AgentResponse:
        """Process agent result and add metadata"""
        agent_output = getattr(result, "output", None)
        if not agent_output:
            return self._create_error_response("Agent returned no output")
        
        # Add source information and processing status
        if (agent_output.extraction_result and 
            agent_output.extraction_result.policies):
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            for policy in agent_output.extraction_result.policies:
                policy.source = DocumentSource(
                    filename=os.path.basename(json_path),
                    folder_path=os.path.dirname(json_path),
                    page_numbers=[],  # Will be populated from policy_data if available
                    language=None
                )
                policy.extraction_date = current_date
                
                if was_truncated:
                    policy.processing_status = "partially_processed"
                    logger.info("Policy marked as partially processed due to truncation")
        
        return agent_output
    def count_tokens(self, text: str) -> int:
        """Count tokens in a string"""
        try:
            import tiktoken
            # Use cl100k_base encoding for GPT-4 models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            # Fallback to approximation
            return len(text) // 4  # Rough estimate
    
    def smart_truncate_content(self, text_with_context: str, available_tokens: int) -> str:
        """
        Intelligently truncate content while preserving important policy information
        
        Args:
            text_with_context: The full text content
            available_tokens: Maximum tokens allowed
            
        Returns:
            Truncated content that preserves key policy sections
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            
            # If content is within limits, return as-is
            current_tokens = len(encoding.encode(text_with_context))
            if current_tokens <= available_tokens:
                return text_with_context
            
            # Split into header and content sections
            lines = text_with_context.split('\n')
            
            # Find the content section (after the header)
            content_start_idx = 0
            for i, line in enumerate(lines):
                if line.strip() == "Document Content:" or line.strip() == "==================":
                    content_start_idx = i + 1
                    break
            
            # Preserve header and some content
            header_lines = lines[:content_start_idx]
            content_lines = lines[content_start_idx:]
            
            if len(content_lines) < 50:
                # For short content, use simple truncation
                tokens_per_line = current_tokens // len(lines)
                target_lines = available_tokens // tokens_per_line
                truncated_lines = lines[:target_lines]
                truncated_lines.append("\n[NOTE: Document was truncated to fit within context window.]")
                return '\n'.join(truncated_lines)
            
            # For longer content, preserve beginning and end sections
            keep_start_pct = 0.6  # Keep 60% from start
            keep_end_pct = 0.2    # Keep 20% from end
            
            start_keep = int(len(content_lines) * keep_start_pct)
            end_keep = int(len(content_lines) * keep_end_pct)
            
            # Build truncated content
            preserved_content = (
                header_lines +
                content_lines[:start_keep] +
                ["\n[CONTENT TRUNCATED - MIDDLE SECTION REMOVED FOR TOKEN LIMITS]\n"] +
                content_lines[-end_keep:] if end_keep > 0 else []
            )
            
            truncated_text = '\n'.join(preserved_content)
            
            # Verify it fits within token limits
            truncated_tokens = len(encoding.encode(truncated_text))
            if truncated_tokens > available_tokens:
                # If still too large, fall back to simple truncation
                tokens_ratio = available_tokens / truncated_tokens
                target_chars = int(len(truncated_text) * tokens_ratio * 0.9)  # 10% buffer
                truncated_text = truncated_text[:target_chars] + "\n[NOTE: Document was truncated to fit within context window.]"
            
            logger.info(f"Smart truncation: {current_tokens} -> {len(encoding.encode(truncated_text))} tokens")
            return truncated_text
            
        except Exception as e:
            logger.error(f"Error in smart truncation: {e}")
            # Fallback to simple truncation
            tokens_ratio = available_tokens / self.count_tokens(text_with_context)
            target_chars = int(len(text_with_context) * tokens_ratio * 0.9)
            return text_with_context[:target_chars] + "\n[NOTE: Document was truncated to fit within context window.]"
    
    def calculate_optimal_batch_size(self, policy_files: List[str]) -> int:
        """
        Calculate optimal batch size based on file complexity and system resources
        
        Args:
            policy_files: List of JSON file paths to process
            
        Returns:
            Optimal batch size for processing
        """
        if not policy_files:
            return self.config.MAX_CONCURRENCY
        
        try:
            # Calculate average file size
            total_size = 0
            valid_files = 0
            
            for file_path in policy_files[:min(10, len(policy_files))]:  # Sample first 10 files
                try:
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        valid_files += 1
                except Exception:
                    continue
            
            if valid_files == 0:
                return self.config.MAX_CONCURRENCY
            
            avg_file_size = total_size / valid_files
            
            # Adaptive batch sizing based on file complexity
            if avg_file_size < 50_000:  # < 50KB - Simple files
                batch_size = min(20, len(policy_files))
                complexity = "simple"
            elif avg_file_size < 200_000:  # < 200KB - Medium files  
                batch_size = min(15, len(policy_files))
                complexity = "medium"
            elif avg_file_size < 500_000:  # < 500KB - Complex files
                batch_size = min(10, len(policy_files))
                complexity = "complex"
            else:  # > 500KB - Very complex files
                batch_size = min(6, len(policy_files))
                complexity = "very complex"
            
            logger.info(f"Dynamic batching: {complexity} files (avg {avg_file_size/1000:.1f}KB) -> batch size {batch_size}")
            return batch_size
            
        except Exception as e:
            logger.warning(f"Error calculating batch size: {e}. Using default.")
            return self.config.MAX_CONCURRENCY
    
    async def api_call_with_retry(self, operation_func, max_retries: int = 3, base_delay: float = 1.0):
        """
        Execute an API operation with exponential backoff retry logic
        
        Args:
            operation_func: Async function to execute (should be a coroutine)
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            
        Returns:
            Result of the operation_func
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Use rate limit semaphore to prevent overwhelming the API
                async with self.rate_limit_semaphore:
                    return await operation_func()
                    
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if it's a retryable error
                is_rate_limit = any(keyword in error_str for keyword in [
                    'rate limit', 'too many requests', '429', 'quota', 'throttle'
                ])
                is_timeout = any(keyword in error_str for keyword in [
                    'timeout', 'connection', 'network'
                ])
                is_server_error = any(keyword in error_str for keyword in [
                    '500', '502', '503', '504', 'internal server error'
                ])
                
                # Don't retry on final attempt or non-retryable errors
                if attempt >= max_retries or not (is_rate_limit or is_timeout or is_server_error):
                    raise e
                
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                # Honor Retry-After header if present
                try:
                    response = getattr(e, 'response', None)
                    if response is not None:
                        ra = response.headers.get('Retry-After') or response.headers.get('retry-after')
                        if ra:
                            delay = float(ra)
                except Exception:
                    pass
                
                # Log the retry
                if is_rate_limit:
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Waiting {delay:.2f}s before retry...")
                elif is_timeout:
                    logger.warning(f"Timeout error (attempt {attempt + 1}/{max_retries + 1}). Waiting {delay:.2f}s before retry...")
                elif is_server_error:
                    logger.warning(f"Server error (attempt {attempt + 1}/{max_retries + 1}). Waiting {delay:.2f}s before retry...")
                
                # Wait before retrying
                await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        raise last_exception
    
    async def process_file_wrapper(self, json_path: str, file_index: int, total_files: int) -> tuple:
        """Process a single file with improved error handling and checkpointing"""
        filename = os.path.basename(json_path)
        
        # Check if file should be skipped based on checkpoint
        if self.should_skip_file(json_path):
            logger.info(f"Skipping already processed file: {filename}")
            return (json_path, [], True, "skipped")
        
        logger.info(f"\n===== Processing file {file_index+1}/{total_files}: {filename} =====")
        
        try:
            response = await asyncio.wait_for(
                self.process_policy_json(json_path),
                timeout=self.config.FILE_TIMEOUT  # Configurable timeout per file
            )
            
            result = self._handle_processing_response(json_path, response)
            
            # Mark as processed on success
            if result[0] is not None:  # If policy was extracted successfully
                self.mark_file_processed(json_path)
            else:
                self.mark_file_failed(json_path)
                
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"File processing timed out after {self.config.FILE_TIMEOUT} seconds"
            logger.error(f"{filename}: {error_msg}")
            self.mark_file_failed(json_path)  # Mark as failed in checkpoint
            return json_path, [], False, error_msg
        except Exception as e:
            error_msg = f"Unhandled exception: {str(e)}"
            logger.error(f"{filename}: {error_msg}")
            self.mark_file_failed(json_path)  # Mark as failed in checkpoint
            return json_path, [], False, error_msg
    
    def _handle_processing_response(self, json_path: str, response: AgentResponse) -> tuple:
        """Handle and validate processing response"""
        filename = os.path.basename(json_path)
        extracted_policies = []
        
        if response.extraction_result and response.extraction_result.has_policy:
            for policy in response.extraction_result.policies:
                if policy.policy_name and policy.policy_name.strip():
                    extracted_policies.append(policy)
                    logger.info(f"✓ {filename}: Extracted '{policy.policy_name}'")
                    
                    # Log key fields extracted
                    valid_fields = policy.get_valid_fields()
                    field_count = len([v for v in valid_fields.values() if v not in ["", None]])
                    logger.info(f"  {field_count} fields populated")
            
            return json_path, extracted_policies, True, None
        else:
            if response.error:
                logger.warning(f"{filename}: Processing error - {response.error}")
                return json_path, [], False, response.error
            else:
                logger.info(f"{filename}: No policies detected")
                return json_path, [], True, None
    
    async def process_directory(self, input_dir: str, concurrency_limit: int = None) -> List[Policy]:
        """Process all policy JSON files in a directory with parallel processing"""
        all_policies = []
        
        # Find all policy JSON files
        policy_files = glob(os.path.join(input_dir, "**/*.json"), recursive=True)
        
        if not policy_files:
            logger.warning(f"No policy JSON files found in {input_dir}")
            return []
        
        # Use dynamic batching if not specified, otherwise use provided limit
        if concurrency_limit is None:
            concurrency_limit = self.calculate_optimal_batch_size(policy_files)
        
        logger.info(f"Found {len(policy_files)} policy JSON files to process with batch size of {concurrency_limit}")
        
        # Initialize progress tracking
        try:
            from progress_tracker import ProgressTracker
            progress_tracker = ProgressTracker(len(policy_files), "Policy Extraction")
        except ImportError:
            progress_tracker = None
            logger.warning("Progress tracker not available")
        
        # Process files in parallel with concurrency control
        successful = 0
        failed = 0
        
        # Process files in batches based on concurrency limit
        for i in range(0, len(policy_files), concurrency_limit):
            batch_files = policy_files[i:i+concurrency_limit]
            batch_tasks = [
                # Fixed indexing by using i+j where j is the batch index
                self.process_file_wrapper(json_path, i+j, len(policy_files)) 
                for j, json_path in enumerate(batch_files)
            ]
            
            # Process batch in parallel
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Process results from the batch
            for json_path, policies, success, error in batch_results:
                filename = os.path.basename(json_path)
                if success:
                    successful += 1
                    if policies:
                        logger.info(f"Adding {len(policies)} policies from {filename} to the collection")
                        all_policies.extend(policies)
                        if progress_tracker:
                            progress_tracker.update(filename, f"Extracted {len(policies)} policies", True)
                    else:
                        if progress_tracker:
                            progress_tracker.update(filename, "No policies found", True)
                else:
                    failed += 1
                    if progress_tracker:
                        progress_tracker.update(filename, f"Failed: {error}", False)
        
        # Complete progress tracking
        if progress_tracker:
            progress_tracker.complete(f"Extracted {len(all_policies)} policies from {successful} files")
        
        # Log summary
        logger.info(f"\nProcessing complete. Files processed: {len(policy_files)}, Successful: {successful}, Failed: {failed}")
        logger.info(f"Total policies extracted: {len(all_policies)}")
        
        # Add a sanity check
        if all_policies:
            logger.info(f"First policy name: {all_policies[0].policy_name}")
            if len(all_policies) > 1:
                logger.info(f"Second policy name: {all_policies[1].policy_name}")
        
        return all_policies
    
    def save_to_excel(self, policies: List[Policy], output_path: str) -> None:
        """Save policies to Excel file by appending to existing file if it exists"""
        if not policies:
            logger.warning("No policies to save.")
            return
        
        try:
            # Log the number of policies we're about to save
            logger.info(f"Preparing to save {len(policies)} policies to Excel")
            
            # Create a list of dictionaries for pandas DataFrame
            policies_data = []
            
            for i, policy in enumerate(policies):
                # Log every 10th policy to avoid excessive logs
                if i < 10 or i % 10 == 0:
                    logger.info(f"Processing policy {i+1}/{len(policies)}: {policy.policy_name}")
                
                # Convert policy to dict
                policy_dict = policy.model_dump()
                
                # Handle nested source field
                if policy.source:
                    policy_dict['source_filename'] = policy.source.filename
                    policy_dict['source_folder'] = policy.source.folder_path
                    policy_dict['source_pages'] = ', '.join(map(str, policy.source.page_numbers))
                    policy_dict['language'] = policy.source.language
                else:
                    policy_dict['source_filename'] = ""
                    policy_dict['source_folder'] = ""
                    policy_dict['source_pages'] = ""
                    policy_dict['language'] = ""
                
                # Remove the original source field
                if 'source' in policy_dict:
                    del policy_dict['source']
                
                # Make sure relevance fields are included
                if 'relevance' not in policy_dict:
                    policy_dict['relevance'] = None
                
                if 'relevance_rationale' not in policy_dict:
                    policy_dict['relevance_rationale'] = None
                
                policies_data.append(policy_dict)
            
            # Create DataFrame with new policies
            if policies_data:
                logger.info(f"Created policies_data list with {len(policies_data)} entries")
                
                # Define field order
                main_fields = [
                    'policy_name', 'relevance', 'relevance_rationale', 'issuing_year', 'has_tables',
                    'policy_summary', 'goals', 'objectives', 'situation_analysis',
                    'legislative_instrument', 'federal_emirate', 'region', 'jurisdiction', 'sector', 'subsector',
                    'metrics', 'policy_owner', 'stakeholders_involved', 'thematic_categorization', 'thematic_sub_categorization'
                ]
                
                # Add source fields at the end (removed confidence_score)
                source_fields = ['source_filename', 'source_folder', 'source_pages', 'language',
                               'extraction_date', 'processing_status', 'full_text']
                
                # Combine all fields in order
                all_fields = main_fields + source_fields
                
                # Create DataFrame for new policies
                new_df = pd.DataFrame(policies_data)
                
                # Check if output file already exists
                existing_df = None
                if os.path.exists(output_path):
                    try:
                        # Try to read existing Excel file
                        logger.info(f"Found existing output file: {output_path}. Will append to it.")
                        existing_df = pd.read_excel(output_path, engine='openpyxl')
                        logger.info(f"Read {len(existing_df)} existing policies from {output_path}")
                    except Exception as e:
                        logger.warning(f"Error reading existing file: {e}. Will create new file.")
                
                # After reading the tracking file but before processing
                if os.path.exists(output_path):
                    # Check which policies already exist in the output file
                    existing_df = pd.read_excel(output_path, engine='openpyxl')
                    existing_names = set(existing_df['policy_name'].dropna().tolist())
                    
                    # Mark policies that are already in the output file as processed
                    for idx, row in new_df.iterrows():
                        if "Policy Name" in row and row["Policy Name"] in existing_names:
                            if "processed" in new_df.columns and new_df.at[idx, "processed"] != "Yes":
                                new_df.at[idx, "processed"] = "Yes"
                                new_df.at[idx, "processed_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                new_df.at[idx, "extraction_status"] = "Already in output"
                                new_df.at[idx, "extraction_notes"] = "Policy exists in output file"
                
                # Combine existing and new data
                if existing_df is not None and not existing_df.empty:
                    # Create a dictionary to map policy names to their corresponding row index in existing_df
                    policy_name_to_idx = {name: idx for idx, name in enumerate(existing_df['policy_name'].dropna())}
                    
                    # Keep track of policies to update vs append
                    policies_to_update = []
                    policies_to_append = []
                    
                    for idx, row in new_df.iterrows():
                        policy_name = row['policy_name']
                        if policy_name in policy_name_to_idx:
                            # Update existing policy
                            existing_idx = policy_name_to_idx[policy_name]
                            for col in new_df.columns:
                                if col in existing_df.columns:
                                    existing_df.at[existing_idx, col] = row[col]
                            policies_to_update.append(policy_name)
                        else:
                            # New policy to append
                            policies_to_append.append(policy_name)
                    
                    # For appending, create a new DataFrame with only the new policies
                    append_df = new_df[new_df['policy_name'].isin(policies_to_append)]
                    
                    # Combine existing (now updated) DataFrame with new policies
                    if not append_df.empty:
                        combined_df = pd.concat([existing_df, append_df], ignore_index=True)
                    else:
                        combined_df = existing_df
                    
                    logger.info(f"Updated {len(policies_to_update)} existing policies and added {len(policies_to_append)} new policies")
                else:
                    # No existing data, just use new data
                    combined_df = new_df
                    logger.info(f"Creating new Excel file with {len(combined_df)} policies")
                
                # Log columns for debugging
                logger.info(f"Final DataFrame has {len(combined_df)} rows")
                logger.info(f"DataFrame columns: {combined_df.columns.tolist()}")
                
                # Ensure columns are in correct order (only include fields that exist)
                existing_fields = [field for field in all_fields if field in combined_df.columns]
                other_fields = [field for field in combined_df.columns if field not in all_fields]
                final_fields = existing_fields + other_fields
                
                logger.info(f"Fields to be included in Excel: {len(final_fields)} columns")
                
                # Reorder columns if needed
                if final_fields:
                    combined_df = combined_df[final_fields]
                
                # Save to Excel with UTF-8 encoding for proper Arabic text handling
                combined_df.to_excel(output_path, index=False, engine='openpyxl')
                
                logger.info(f"✓ Saved {len(combined_df)} policies to {output_path} ({len(new_df)} newly added)")
            else:
                logger.warning("No policy data to save to Excel")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {str(e)}")
            traceback.print_exc()

    # Add a direct lookup method to find a policy JSON by name
    def find_policy_by_name(self, policy_name, output_dir="output"):
        """Find a policy file based on policy name keywords"""
        if not policy_name or not os.path.exists(output_dir):
            return None
            
        # Extract keywords from policy name
        policy_keywords = []
        for keyword in ["Law", "Resolution", "Decision", "Circular", "Executive", "Council"]:
            if keyword in policy_name:
                policy_keywords.append(keyword)
                
        if not policy_keywords:
            return None
            
        # Search for JSON files that might match the policy name using pathlib
        output_path = Path(output_dir)
        json_files = list(output_path.glob("**/*.json"))
        
        for json_file in json_files:
            if any(keyword in json_file.name for keyword in policy_keywords):
                try:
                    with open(json_file, 'rb') as f:
                        data = orjson.loads(f.read())
                        if data.get('policy_name') == policy_name:
                            logger.info(f"Found policy file by name match: {json_file}")
                            return str(json_file)
                except Exception:
                    pass  # Skip files with JSON errors
                        
        logger.warning(f"No file found for policy: {policy_name}")
        return None

    def find_policy_file(self, relative_path):
        """
        Try to find the policy file by trying different filename patterns
        if the exact path doesn't work.
        
        Args:
            relative_path: The relative path from all_policies.xlsx
            
        Returns:
            The correct path if found, or the original path if not found
        """
        # First try the exact path
        full_path = os.path.join(os.getcwd(), relative_path.lstrip('./'))
        if os.path.exists(full_path):
            return full_path
        
        # If path doesn't exist, try different truncation patterns
        try:
            # Parse the path to extract components
            dir_path = os.path.dirname(full_path)
            filename = os.path.basename(full_path)
            
            # Extract the prefix and policy name
            parts = filename.split('_', 2)  # Split at most twice to get: ['policy', '001', 'rest_of_name.json']
            if len(parts) >= 3:
                prefix = f"{parts[0]}_{parts[1]}_"  # e.g., "policy_001_"
                
                # If directory exists, try to find files with the same prefix
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        if file.startswith(prefix):
                            logger.info(f"Found alternative file: {file} instead of {filename}")
                            policy_path = os.path.join(dir_path, file)
                            logger.info(f"Using matched file: {policy_path}")
                            return policy_path
            
            # If we got here, we couldn't find a match
            logger.warning(f"Could not find matching policy file for {relative_path}")
            return full_path
        except Exception as e:
            logger.error(f"Error looking for policy file: {e}")
            return full_path

# Main function for CLI usage
def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract policies from JSON files')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory with policy JSON files')
    parser.add_argument('--output', '-o', type=str, default="Output/extracted_policies.xlsx",
                        help='Output Excel file path')
    parser.add_argument('--concurrency', '-c', type=int, default=None,
                        help='Maximum number of concurrent API calls')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Process all policy folders in the input directory')
    parser.add_argument('--tracking', '-t', type=str, default="Output/all_policies.xlsx",
                        help='Excel file for tracking policies')
    parser.add_argument('--process-all', '-a', action='store_true',
                        help='Process all policies regardless of their processing status')
    parser.add_argument('--use-tracking', action='store_true',
                        help='Explicitly use the tracking file for processing')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit the number of policies to process from the tracking file')
    args = parser.parse_args()
    
    # Validate input path exists
    if not os.path.exists(args.input):
        logger.error(f"Error: Input path '{args.input}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Determine processing mode - only process files one way
    if args.batch:
        logger.info("Using batch processing mode")
        process_mode = "batch"
    else:
        # Default to tracking mode always
        logger.info(f"Using tracking file mode with file: {args.tracking}")
        
        # Check if tracking file exists (but don't create it automatically)
        if not os.path.exists(args.tracking):
            logger.error(f"Tracking file {args.tracking} does not exist. Please provide a valid tracking file.")
            return
        
        process_mode = "tracking"
    
    # Run the async process in an event loop
    try:
        if process_mode == "batch":
            asyncio.run(batch_process(args.input, args.output, args.concurrency))
            logger.info(f"\nBatch extraction complete. Results saved to {args.output}")
        elif process_mode == "tracking":
            asyncio.run(process_from_tracking(args.input, args.output, args.tracking, args.concurrency, args.process_all, args.limit))
            logger.info(f"\nTracked extraction complete. Results saved to {args.output}")
        else:
            asyncio.run(run_async(args.input, args.output, args.concurrency))
            logger.info(f"\nExtraction complete. Results saved to {args.output}")
    except RuntimeError as e:
        # Handle case where we might already be in an event loop
        logger.warning(f"RuntimeError encountered: {e}")
        loop = asyncio.get_event_loop()
        if loop.is_running():
            logger.info("Using existing running event loop")
            if process_mode == "batch":
                asyncio.ensure_future(batch_process(args.input, args.output, args.concurrency))
            elif process_mode == "tracking":
                asyncio.ensure_future(process_from_tracking(args.input, args.output, args.tracking, args.concurrency, args.process_all, args.limit))
            else:
                asyncio.ensure_future(run_async(args.input, args.output, args.concurrency))
        else:
            logger.info("Using existing event loop")
            if process_mode == "batch":
                loop.run_until_complete(batch_process(args.input, args.output, args.concurrency))
            elif process_mode == "tracking":
                loop.run_until_complete(process_from_tracking(args.input, args.output, args.tracking, args.concurrency, args.process_all, args.limit))
            else:
                loop.run_until_complete(run_async(args.input, args.output, args.concurrency))


# Helper function to run async code
async def run_async(input_dir: str, output_path: str, concurrency: int = None):
    config = Config()
    processor = PolicyJsonProcessor(config)
    
    # Process all policy JSON files
    policies = await processor.process_directory(input_dir, concurrency)
    
    # Save to Excel
    processor.save_to_excel(policies, output_path)


# Process policies using the tracking Excel file
async def process_from_tracking(base_dir: str, output_path: str, tracking_path: str, concurrency: int = None, process_all: bool = False, limit: int = None):
    """Process policies using information from tracking Excel file"""
    logger.info(f"Using tracking file: {tracking_path}")
    
    # Helper function to find a matching policy file
    def find_matching_policy_file(original_path):
        """Find a matching policy file with a similar name if the exact path doesn't exist"""
        if os.path.exists(original_path):
            return original_path
            
        try:
            # Get directory and filename
            dirname = os.path.dirname(original_path)
            filename = os.path.basename(original_path)
            
            # Check if directory exists
            if not os.path.exists(dirname):
                return original_path
                
            # Try to extract the policy ID prefix (e.g., "policy_001_")
            parts = filename.split('_', 2)
            if len(parts) >= 3 and parts[0] == "policy":
                prefix = f"{parts[0]}_{parts[1]}_"
                
                # Look for files with the same prefix
                for file in os.listdir(dirname):
                    if file.startswith(prefix):
                        logger.info(f"Found alternative file: {file} instead of {filename}")
                        return os.path.join(dirname, file)
                        
            # If we got here, we couldn't find a match with policy ID
            # Try a simpler approach with file prefix
            name_parts = filename.split('.')
            if len(name_parts) >= 2:
                file_prefix = name_parts[0].rsplit('_', 1)[0]
                matching_files = [f for f in os.listdir(dirname) if f.startswith(file_prefix) and f.endswith('.json')]
                if matching_files:
                    logger.info(f"Found prefix match: {matching_files[0]} instead of {filename}")
                    return os.path.join(dirname, matching_files[0])
        except Exception as e:
            logger.warning(f"Error trying to find alternative file: {e}")
            
        # Default to original path if no match found
        return original_path
    
    try:
        # Read the tracking file with all columns preserved
        tracking_df = pd.read_excel(tracking_path, engine='openpyxl')
        logger.info(f"Successfully read tracking file with {len(tracking_df)} entries")
        
        # Check if the required columns exist
        required_columns = ["Policy ID", "Policy Name", "Relative Path"]
        missing_columns = [col for col in required_columns if col not in tracking_df.columns]
        
        if missing_columns:
            logger.error(f"Tracking file is missing required columns: {', '.join(missing_columns)}")
            return
        
        # Add tracking columns if they don't exist
        for col in ["processed", "processed_date", "extraction_status", "extraction_notes"]:
            if col not in tracking_df.columns:
                tracking_df[col] = ""
                logger.info(f"Added '{col}' column to tracking file")
        
        # Make a backup of the original tracking file
        backup_path = tracking_path.replace('.xlsx', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        tracking_df.to_excel(backup_path, index=False, engine='openpyxl')
        logger.info(f"Created backup of tracking file at {backup_path}")
        
        # Keep a copy of the original tracking data to preserve non-tracking columns
        original_tracking_df = tracking_df.copy()
        
        # After reading the tracking file but before processing
        if os.path.exists(output_path):
            # Check which policies already exist in the output file
            existing_df = pd.read_excel(output_path, engine='openpyxl')
            existing_names = set(existing_df['policy_name'].dropna().tolist())
            
            # Mark policies that are already in the output file as processed
            for idx, row in tracking_df.iterrows():
                if "Policy Name" in row and row["Policy Name"] in existing_names:
                    if "processed" in tracking_df.columns and tracking_df.at[idx, "processed"] != "Yes":
                        tracking_df.at[idx, "processed"] = "Yes"
                        tracking_df.at[idx, "processed_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        tracking_df.at[idx, "extraction_status"] = "Already in output"
                        tracking_df.at[idx, "extraction_notes"] = "Policy exists in output file"
        
        # Save updated tracking file
        tracking_df.to_excel(tracking_path, index=False, engine='openpyxl')
        
        # Refilter policies to process
        if process_all:
            policies_to_process = tracking_df
        else:
            policies_to_process = tracking_df[(tracking_df["processed"] != "Yes")]
        
        # After reading the tracking file and before applying the limit
        if limit is not None and limit > 0:
            original_count = len(policies_to_process)
            
            # Create priority groups
            never_processed = policies_to_process[policies_to_process["processed"] == ""]
            failed_processed = policies_to_process[policies_to_process["processed"] == "Failed"]
            other_unprocessed = policies_to_process[~policies_to_process.index.isin(never_processed.index) & 
                                                  ~policies_to_process.index.isin(failed_processed.index)]
            
            # Combine in priority order
            prioritized_policies = pd.concat([never_processed, failed_processed, other_unprocessed])
            
            # Apply limit to the prioritized list
            policies_to_process = prioritized_policies.iloc[:limit]
            
            logger.info(f"Limiting processing to {len(policies_to_process)} policies out of {original_count} (limit={limit})")
            logger.info(f"Prioritized: {len(never_processed)} never processed, {len(failed_processed)} previously failed")
        
        if len(policies_to_process) == 0:
            logger.info("No policies need to be processed")
            return
        
        # Create configuration
        config = Config()
        processor = PolicyJsonProcessor(config)
        
        # Process each policy
        all_policies = []
        processed_indices = []
        skipped_files = []
        failed_files = []
        no_policy_files = []
        
        # Group policies into batches for concurrent processing
        total_policies = len(policies_to_process)
        concurrency_limit = concurrency or config.MAX_CONCURRENCY
        logger.info(f"Processing {total_policies} policies with concurrency limit of {concurrency_limit}")
        
        # Process in batches
        for batch_start in range(0, total_policies, concurrency_limit):
            batch_end = min(batch_start + concurrency_limit, total_policies)
            batch = policies_to_process.iloc[batch_start:batch_end]
            batch_tasks = []
            batch_indices = []
            
            for idx, row in batch.iterrows():
                policy_id = row["Policy ID"]
                policy_name = row["Policy Name"]
                relative_path = row["Relative Path"]
                
                if pd.isna(relative_path):
                    logger.warning(f"Skipping policy {policy_id}: {policy_name} - No path information")
                    skipped_files.append((policy_id, policy_name, "Missing path"))
                    # Update status in tracking DataFrame
                    tracking_df.at[idx, "extraction_status"] = "Skipped"
                    tracking_df.at[idx, "extraction_notes"] = "Missing path information"
                    continue
                
                # Get the full path to the specific JSON file
                policy_path = os.path.join(base_dir, relative_path)
                
                # Try to find a matching file if the exact path doesn't exist
                policy_path = find_matching_policy_file(policy_path)
                
                # If still not found, skip this policy
                if not os.path.exists(policy_path):
                    logger.warning(f"Skipping policy {policy_id}: {policy_name} - File not found: {policy_path}")
                    skipped_files.append((policy_id, policy_name, "File not found"))
                    # Update status in tracking DataFrame
                    tracking_df.at[idx, "extraction_status"] = "Skipped"
                    tracking_df.at[idx, "extraction_notes"] = "File not found"
                    continue
                
                logger.info(f"Processing policy {policy_id}: {policy_name}")
                # Create task to process the specific JSON file directly
                batch_tasks.append(processor.process_policy_json(policy_path))
                batch_indices.append(idx)
            
            # Run batch of tasks concurrently
            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks)
                
                # Process results
                for i, (result, idx) in enumerate(zip(batch_results, batch_indices)):
                    row = policies_to_process.loc[idx]
                    policy_id = row["Policy ID"]
                    policy_name = row["Policy Name"]
                    relative_path = row["Relative Path"]
                    policy_path = os.path.join(base_dir, relative_path)
                    
                    if result.extraction_result and result.extraction_result.has_policy:
                        policies = result.extraction_result.policies
                        if policies:
                            logger.info(f"Successfully extracted {len(policies)} policies from {policy_name}")
                            
                            # Add original policy text to each policy if needed
                            for policy in policies:
                                try:
                                    policy_filename = policy.source.filename
                                    logger.info(f"Attempting to find and load content for policy: {policy_filename}")
                                    
                                    # First try using our specialized finder function
                                    file_path = processor.find_policy_by_name(policy_name)
                                    
                                    if file_path and os.path.exists(file_path):
                                        try:
                                            with open(file_path, 'rb') as f:
                                                policy_data = orjson.loads(f.read())
                                                policy.full_text = policy_data.get('content', '')
                                                if policy.full_text:
                                                    logger.info(f"Successfully loaded content from file: {file_path}")
                                                else:
                                                    logger.warning(f"File found but no content field: {file_path}")
                                                    policy.full_text = f"File found but no content field available"
                                        except Exception as e:
                                            logger.error(f"Error reading found policy file {file_path}: {str(e)}")
                                            policy.full_text = f"Error reading policy file: {str(e)}"
                                    else:
                                        # Fall back to the original method if our finder fails
                                        dir_path = policy.source.folder_path
                                        logger.info(f"File not found by finder, trying original path: {dir_path}")
                                        
                                        # Fix incorrect path structure
                                        if dir_path.startswith("./"):
                                            dir_path = dir_path[2:]
                                        
                                        if not dir_path.startswith("output/"):
                                            dir_path = "output/" + dir_path
                                        
                                        # If directory still doesn't exist, look for common policy directories
                                        if not os.path.exists(dir_path):
                                            logger.warning(f"Directory not found: {dir_path}")
                                            
                                            # Extract just the policy directory name without hardcoding
                                            dir_components = dir_path.split(os.sep)
                                            policies_idx = -1
                                            
                                            # Find the "policies" component in the path
                                            for idx, component in enumerate(dir_components):
                                                if component == "policies":
                                                    policies_idx = idx
                                                    break
                                            
                                            if policies_idx > 0 and policies_idx < len(dir_components):
                                                # Get the parent folder of the policies directory
                                                parent_folder = dir_components[policies_idx-1] if policies_idx > 0 else ""
                                                logger.info(f"Looking for policies in parent folder: {parent_folder}")
                                                
                                                # Search for any policies directory in output using pathlib
                                                output_path = Path("output")
                                                for policies_dir in output_path.glob("**/policies"):
                                                    if policies_dir.is_dir():
                                                        # Check if the parent folder is similar
                                                        if parent_folder and parent_folder.lower() in str(policies_dir.parent).lower():
                                                            # Found a matching directory structure
                                                            potential_dir = str(policies_dir)
                                                            logger.info(f"Found potential matching directory: {potential_dir}")
                                                            if policies_dir.exists():
                                                                dir_path = potential_dir
                                                                break
                                    
                                    # If still not found, just use any policies directory as fallback
                                    if not os.path.exists(dir_path):
                                        logger.warning("No matching directory found, trying any policies directory")
                                        output_path = Path("output")
                                        for policies_dir in output_path.glob("**/policies"):
                                            if policies_dir.is_dir():
                                                dir_path = str(policies_dir)
                                                logger.info(f"Using fallback directory: {dir_path}")
                                                break
                                except Exception as e:
                                    logger.error(f"Error adding full text to policy: {e}")
                                    policy.full_text = "Error reading full text"
                            
                            # Add the policies to the master list and track the processed index
                            all_policies.extend(policies)
                            processed_indices.append(idx)
                            
                            # Mark this policy as successfully processed in the tracking DataFrame
                            tracking_df.at[idx, "processed"] = "Yes"
                            tracking_df.at[idx, "processed_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            tracking_df.at[idx, "extraction_status"] = "Success"
                            tracking_df.at[idx, "extraction_notes"] = f"Successfully extracted {len(policies)} policies"
                        else:
                            logger.warning(f"No policies extracted from {policy_name}")
                            no_policy_files.append((policy_id, policy_name, "No policies found"))
                            # Update status in tracking DataFrame
                            tracking_df.at[idx, "processed"] = "Yes" 
                            tracking_df.at[idx, "processed_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            tracking_df.at[idx, "extraction_status"] = "Empty"
                            tracking_df.at[idx, "extraction_notes"] = "No policies found in file"
                    else:
                        error_msg = result.error if result.error else "Unknown error"
                        logger.warning(f"Failed to extract policy from {policy_name}: {error_msg}")
                        failed_files.append((policy_id, policy_name, error_msg))
                        # Update status in tracking DataFrame
                        tracking_df.at[idx, "processed"] = "Failed"
                        tracking_df.at[idx, "processed_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        tracking_df.at[idx, "extraction_status"] = "Failed"
                        tracking_df.at[idx, "extraction_notes"] = error_msg
            
            # Only update the tracking columns in the original DataFrame
            tracking_columns = ["processed", "processed_date", "extraction_status", "extraction_notes"]
            
            # Copy only the tracking columns from the updated tracking_df to original_tracking_df
            for idx in tracking_df.index:
                if idx in original_tracking_df.index:
                    for col in tracking_columns:
                        if col in tracking_df.columns and col in original_tracking_df.columns:
                            original_tracking_df.at[idx, col] = tracking_df.at[idx, col]
            
            # Save the original DataFrame with updated tracking columns
            original_tracking_df.to_excel(tracking_path, index=False, engine='openpyxl')
            logger.info(f"Updated tracking file after batch {batch_start//concurrency_limit + 1}")
        
        # Log summary of processing results
        logger.info("\n===== PROCESSING SUMMARY =====")
        logger.info(f"Total files to process: {total_policies}")
        logger.info(f"Successfully processed: {len(processed_indices)}")
        logger.info(f"Skipped files: {len(skipped_files)}")
        logger.info(f"Failed extractions: {len(failed_files)}")
        logger.info(f"Files with no policies: {len(no_policy_files)}")
        logger.info(f"Total policies extracted: {len(all_policies)}")
        
        # Log details of skipped/failed files
        if skipped_files:
            logger.info("\nSkipped files:")
            for policy_id, policy_name, reason in skipped_files:
                logger.info(f"  - {policy_id}: {policy_name} - {reason}")
        
        if failed_files:
            logger.info("\nFailed extractions:")
            for policy_id, policy_name, error in failed_files:
                logger.info(f"  - {policy_id}: {policy_name} - {error}")
        
        if no_policy_files:
            logger.info("\nFiles with no policies detected:")
            for policy_id, policy_name, note in no_policy_files:
                logger.info(f"  - {policy_id}: {policy_name} - {note}")
        
        # Save the extracted policies
        if all_policies:
            processor.save_to_excel(all_policies, output_path)
            logger.info(f"Saved {len(all_policies)} extracted policies to {output_path}")
        else:
            logger.warning("No policies were extracted")
    
    except Exception as e:
        logger.error(f"Error processing from tracking file: {str(e)}")
        traceback.print_exc()


# Batch processing function to process multiple policy folders
async def batch_process(base_dir: str, output_path: str, concurrency: int = None):
    """Process all policy folders in the base directory with proper async context management"""
    config = Config()
    processor = PolicyJsonProcessor(config)
    all_policies = []
    
    # Set up client with proper async context management
    http_client = await processor.setup_openai_client()
    processor.setup_agent()
    
    try:
        # Find all subdirectories that contain a 'policies' folder using pathlib
        base_path = Path(base_dir)
        source_dirs = [str(policies_dir) for policies_dir in base_path.glob("**/policies") if policies_dir.is_dir()]
        
        if not source_dirs:
            logger.warning(f"No 'policies' folders found in {base_dir}")
            return
        
        logger.info(f"Found {len(source_dirs)} policy directories to process")
        
        # Process each policies directory
        for i, policy_dir in enumerate(source_dirs):
            logger.info(f"\n==== Processing directory {i+1}/{len(source_dirs)}: {policy_dir} ====")
            
            # Process all policy JSON files in this directory
            policies = await processor.process_directory(policy_dir, concurrency)
            
            if policies:
                logger.info(f"Added {len(policies)} policies from {policy_dir}")
                
                # Add full text to each policy
                for policy in policies:
                    if policy.source and policy.source.filename and policy.source.folder_path:
                        try:
                            policy_filename = policy.source.filename
                            logger.info(f"Attempting to find and load content for policy: {policy_filename}")
                            
                            # First try using our specialized finder function
                            file_path = processor.find_policy_by_name(policy_filename)
                            
                            if file_path and os.path.exists(file_path):
                                try:
                                    with open(file_path, 'rb') as f:
                                        policy_data = orjson.loads(f.read())
                                        policy.full_text = policy_data.get('content', '')
                                        if policy.full_text:
                                            logger.info(f"Successfully loaded content from file: {file_path}")
                                        else:
                                            logger.warning(f"File found but no content field: {file_path}")
                                            policy.full_text = f"File found but no content field available"
                                except Exception as e:
                                    logger.error(f"Error reading found policy file {file_path}: {str(e)}")
                                    policy.full_text = f"Error reading policy file: {str(e)}"
                            else:
                                # Fall back to the original method if our finder fails
                                dir_path = policy.source.folder_path
                                logger.info(f"File not found by finder, trying original path: {dir_path}")
                                
                                # Fix incorrect path structure
                                if dir_path.startswith("./"):
                                    dir_path = dir_path[2:]
                                
                                if not dir_path.startswith("output/"):
                                    dir_path = "output/" + dir_path
                                
                                # If directory still doesn't exist, look for common policy directories
                                if not os.path.exists(dir_path):
                                    logger.warning(f"Directory not found: {dir_path}")
                                    
                                    # Extract just the policy directory name without hardcoding
                                    dir_components = dir_path.split(os.sep)
                                    policies_idx = -1
                                
                                # Find the "policies" component in the path
                                for idx, component in enumerate(dir_components):
                                    if component == "policies":
                                        policies_idx = idx
                                        break
                                
                                if policies_idx > 0 and policies_idx < len(dir_components):
                                    # Get the parent folder of the policies directory
                                    parent_folder = dir_components[policies_idx-1] if policies_idx > 0 else ""
                                    logger.info(f"Looking for policies in parent folder: {parent_folder}")
                                    
                                    # Search for any policies directory in output using pathlib
                                    output_path = Path("output")
                                    for policies_dir in output_path.glob("**/policies"):
                                        if policies_dir.is_dir():
                                            # Check if the parent folder is similar
                                            if parent_folder and parent_folder.lower() in str(policies_dir.parent).lower():
                                                # Found a matching directory structure
                                                potential_dir = str(policies_dir)
                                                logger.info(f"Found potential matching directory: {potential_dir}")
                                                if policies_dir.exists():
                                                    dir_path = potential_dir
                                                    break
                                    
                                    # If still not found, just use any policies directory as fallback
                                    if not os.path.exists(dir_path):
                                        logger.warning("No matching directory found, trying any policies directory")
                                        output_path = Path("output")
                                        for policies_dir in output_path.glob("**/policies"):
                                            if policies_dir.is_dir():
                                                dir_path = str(policies_dir)
                                                logger.info(f"Using fallback directory: {dir_path}")
                                                break
                        except Exception as e:
                            logger.error(f"Error adding full text to policy: {e}")
                            policy.full_text = "Error reading full text"
            
                all_policies.extend(policies)
            else:
                logger.warning(f"No policies extracted from {policy_dir}")
            
            # Save checkpoint after each directory (every 10 directories or so)
            if (i + 1) % 10 == 0 or i == len(source_dirs) - 1:
                processor.save_checkpoint()
                logger.info(f"Checkpoint saved after processing {i+1} directories")
    
        # Save all policies to a single Excel file
        if all_policies:
            logger.info(f"\nTotal policies collected: {len(all_policies)}")
            processor.save_to_excel(all_policies, output_path)
        else:
            logger.warning("No policies were extracted from any directory")
            
        # Save final checkpoint
        processor.save_checkpoint()
        logger.info(f"Processing complete. Processed: {len(processor.processed_files)}, Failed: {len(processor.failed_files)}")
    
    finally:
        # Ensure proper cleanup of HTTP connections
        await processor.cleanup_client()


if __name__ == "__main__":
    main() 