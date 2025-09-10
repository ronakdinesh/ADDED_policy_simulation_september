import os
import re
import csv
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PDF Processing
import fitz  # PyMuPDF
import pandas as pd

# Azure OpenAI - updated imports to match bot_new.py
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

# Define configuration
class Config:
    """Configuration for the Policy Extraction Agent"""
    # Azure OpenAI settings - use values from .env with correct API version
    AZURE_OPENAI_KEY = os.environ.get("Azure_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT = os.environ.get("Azure_OPENAI_ENDPOINT", "https://teams-ai-agent.openai.azure.com")
    AZURE_OPENAI_API_VERSION = "2024-10-21"  # Updated to match bot_new.py
    AZURE_OPENAI_DEPLOYMENT = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")

    # Processing settings - updated for GPT-4o's larger context window
    MAX_TOKENS = 128000  # Maximum context window for GPT-4o (128K tokens)
    CHUNK_SIZE = 100000  # Much larger chunks to minimize policy splitting
    CHUNK_OVERLAP = 5000  # Increased overlap for better context preservation
    OUTPUT_DIR = "output"  # Directory to store outputs

# Document source information
class DocumentSource(BaseModel):
    """Source information for a document"""
    filename: str
    folder_path: str
    page_numbers: List[int] = Field(default_factory=list)
    language: Optional[str] = None

# Policy model based on ParameterDefinition.csv
class Policy(BaseModel):
    """Pydantic model for policy information extraction"""
    # Main identification fields
    policy_name: str = Field(description="The official name of the policy, usually found at the beginning of each policy as a header")
    issuing_year: Optional[str] = Field(None, description="The year the policy was officially issued. It may be stated in the policy name or identified from its publication date found within the document")
    
    # Content fields
    policy_summary: Optional[str] = Field(None, description="A concise overview of the policy, outlining its purpose, key objectives, intended impact, etc. It provides a high-level understanding of what the policy aims to address")
    goals: Optional[str] = Field(None, description="The high-level aims the policy wants to achieve (more qualitative); e.g. to attract foreign investment")
    objectives: Optional[str] = Field(None, description="Specific, actionable, and measurable targets that the policy aims to achieve (more quantitative); e.g. increase funding available for incentive programs by AED x; increase FDI growth by 2%, etc.")
    situation_analysis: Optional[str] = Field(None, description="The background or reasons why the policy was needed establishing the context for it")
    
    # Classification fields
    legislative_instrument: Optional[str] = Field(None, description="""The legal tool used to enforce the policy. 
    Must be one of the following options:
    1. UAE Constitution: The highest source of all legislation in the country, establishing the general foundations of the legal and legislative system
    2. Federal Law: Issued by the Federal National Council and ratified by the President of the State
    3. Decree by Law: Issued directly by the President of the State in urgent cases that do not allow for delays, and they carry the force of law
    4. Federal Regulation and Decision: Issued by the Cabinet or relevant federal regulatory authorities to implement and interpret federal laws
    5. Local Law of the Emirate of Abu Dhabi: Issued by the Ruler of the Emirate to regulate local matters, provided they do not conflict with federal laws
    6. Amiri Decree: Issued by the Ruler of Abu Dhabi to address specific matters within the Emirate
    7. Executive Council Decisions for the Emirate of Abu Dhabi: Issued by the Executive Council, addressing public policies and executive procedures within the Emirate
    8. Executive Regulation and Local Regulatory Decision: Issued by relevant government entities to regulate economic and administrative activities""")
    
    federal_emirate: Optional[str] = Field(None, description="""Whether the policy is developed on the federal level or just the Emirate of Abu Dhabi's level.
    Must be one of the following options:
    1. Federal
    2. Emirate""")
    
    region: Optional[str] = Field(None, description="""The geographical area covered by the policy.
    Must be one of the following options:
    1. All
    2. Abu Dhabi City
    3. Al Ain
    4. Al Dhafra
    5. Abu Dhabi City & Al Ain
    6. Al Ain & Al Dhafra
    7. Abu Dhabi City & Al Dhafra""")
    
    jurisdiction: Optional[str] = Field(None, description="""The legal or regulatory scope within which the policy applies.
    Must be one of the following options:
    1. All
    2. Mainland
    3. Freezone
    4. Island""")
    
    sector: Optional[str] = Field(None, description="""The sector this policy affects.
    Must be one of the following options:
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
    
    # Performance and ownership fields
    metrics: Optional[str] = Field(None, description="Key performance indicators (KPIs) used to measure policy performance")
    policy_owner: Optional[str] = Field(None, description="The government entity that owns the policy (should be a single entity)")
    stakeholders_involved: Optional[str] = Field(None, description="Government entities that play a role in developing the policy but do not have primary ownership or responsibility")
    thematic_categorization: Optional[str] = Field(None, description="""The general theme the policy falls under.
    Must be one of the following options:
    1. Non oil GDP growth
    2. Priority clusters growth
    3. Digital economy expansion
    4. Non oil exports growth
    5. Regional development
    6. FDI growth
    7. Incentive programs
    8. Government revenue growth
    9. Government efficiency (spending)
    10. Ease of doing business
    11. Cost of doing business
    12. (High) skilled workforce growth
    13. Employment (job creation)
    14. Emiratization level
    15. Finance access improvement
    16. Innovation and R&D investment
    17. Mega infrastructure development
    18. Cost of living""")
    
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
    has_policy: bool = Field(description="Whether a policy was found in the document chunk")
    multiple_policies: bool = Field(default=False, description="Whether multiple policies were found in the document chunk")
    policy_count: int = Field(default=0, description="Number of policies found in the document chunk")
    policy_names_extracted: List[str] = Field(default_factory=list, description="Names of the extracted policies if found")
    
class AgentResponse(BaseModel):
    """Main agent response model"""
    message: str = Field(description="Human-readable message describing the result")
    result: ExtractionResult = Field(description="Extraction results")
    suggested_next_action: Optional[str] = Field(None, description="Suggested next action if needed")

# Document processing functions
def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF and return a list of page contents"""
    pages = []
    
    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():  # Only add pages with content
                pages.append({
                    "page_number": i + 1,  # 1-indexed page numbers
                    "text": text,
                    "language": detect_language(text)
                })
        doc.close()
    except Exception as e:
        print(f"Error extracting text from {file_path}: {str(e)}")
        traceback.print_exc()
    
    return pages

def detect_language(text: str) -> str:
    """Simple language detection for Arabic vs. English"""
    # Arabic Unicode range (simplified check)
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    
    # Count Arabic characters
    arabic_count = len(arabic_pattern.findall(text))
    
    # If significant Arabic content is found
    if arabic_count > len(text) * 0.3:  # If more than 30% is Arabic
        return "Arabic"
    return "English"

def extract_field(text: str, field_prefix: str) -> str:
    """Extract a field value from text based on a prefix"""
    try:
        # Find the field in the text
        start_idx = text.find(field_prefix)
        if start_idx == -1:
            return ""
        
        # Move to the end of the prefix
        start_idx += len(field_prefix)
        
        # Find the end of the line or the next field
        end_idx = text.find("\n", start_idx)
        if end_idx == -1:
            end_idx = len(text)
        
        # Extract and clean the value
        value = text[start_idx:end_idx].strip()
        return value
    except Exception:
        return ""

def chunk_document(pages: List[Dict[str, Any]], chunk_size: int = 12000, overlap: int = 1000) -> List[Dict[str, Any]]:
    """Create overlapping chunks from document pages"""
    chunks = []
    current_chunk = ""
    current_pages = []
    
    for page in pages:
        page_text = page["text"]
        page_num = page["page_number"]
        
        # If adding this page would exceed chunk size, save current chunk and start new one
        if len(current_chunk) + len(page_text) > chunk_size and current_chunk:
            # Add overlap from the end of the current chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            
            chunks.append({
                "text": current_chunk,
                "pages": current_pages.copy()
            })
            
            # Start new chunk with overlap
            current_chunk = overlap_text
            # Find the pages that contribute to the overlap text
            current_pages = []  # Reset and will be populated properly below
        
        current_chunk += page_text
        current_pages.append(page_num)
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append({
            "text": current_chunk,
            "pages": current_pages
        })
    
    return chunks

# Policy Extraction Agent
class PolicyExtractionAgent:
    """Agent to extract policy information from documents"""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_openai_client()
        self.load_parameter_definitions()
        self.setup_agent()
    
    def setup_openai_client(self):
        """Set up the Azure OpenAI client using async client like in bot_new.py"""
        print(f"Connecting to Azure OpenAI at: {self.config.AZURE_OPENAI_ENDPOINT}")
        print(f"Using deployment: {self.config.AZURE_OPENAI_DEPLOYMENT}")
        print(f"API Version: {self.config.AZURE_OPENAI_API_VERSION}")
        
        # Use AsyncAzureOpenAI like in bot_new.py
        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
            api_version=self.config.AZURE_OPENAI_API_VERSION,
            api_key=self.config.AZURE_OPENAI_KEY
        )
        
        # Set up OpenAI model for Pydantic AI - without temperature setting
        self.model = OpenAIModel(
            self.config.AZURE_OPENAI_DEPLOYMENT,
            openai_client=self.client,
        )
        
        # Set model settings to be used during run
        self.model_settings = {'temperature': 0.0}  # Store settings to use at runtime
    
    def load_parameter_definitions(self):
        """Load parameter definitions from CSV for the agent prompt"""
        try:
            # Get the script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            param_file = os.path.join(script_dir, "ParameterDefinition.csv")
            
            if not os.path.exists(param_file):
                print(f"Warning: Parameter definition file not found at {param_file}")
                self.parameter_definitions = ""
                return
                
            params_df = pd.read_csv(param_file)
            
            # Format as a string for inclusion in the prompt
            param_text = []
            for _, row in params_df.iterrows():
                param_text.append(f"{row['Parameter']}: {row['Definition']}")
            
            self.parameter_definitions = "\n".join(param_text)
            
        except Exception as e:
            print(f"Error loading parameter definitions: {str(e)}")
            self.parameter_definitions = ""
    
    def setup_agent(self):
        """Set up the Pydantic AI agent"""
        system_prompt = f"""
        You are a Policy Extraction AI Agent specialized in analyzing government policy documents.
        Your task is to extract structured information about ECONOMIC POLICIES from text according to defined parameters.
        
        # PARAMETER DEFINITIONS:
        {self.parameter_definitions}
        
        # ECONOMIC POLICY FOCUS:
        You are specifically looking for economic policies, which typically:
        - Address economic development, trade, investment, economic growth, or financial matters
        - Involve business regulations, economic incentives, industrial development, or fiscal measures
        - Are issued by government entities to guide economic activities or development
        - Have objectives related to GDP growth, employment, business environment, or financial stability
        
        # INSTRUCTIONS:
        1. Carefully read the provided document chunk. You have a large context window (128K tokens), so you can process extensive documents.
        2. First gain a holistic understanding of the entire document before beginning analysis. Consider how different sections might relate to form complete policies.
        3. Determine whether the document contains one or more ECONOMIC POLICIES. If it does not contain any economic policies, indicate this clearly.
        4. If economic policies are present, identify ALL policies in the chunk, not just the most prominent one.
        5. Extract all information according to the parameter definitions for EACH policy you identify.
        6. For fields with fixed options (Legislative instrument, Federal/Emirate, Region, Jurisdiction, Sector, Thematic categorization):
           - You MUST select EXACTLY ONE of the predefined options only - do not modify, paraphrase or create new options
           - Return ONLY the option text WITHOUT the numbering prefix (e.g., "Digital economy expansion" not "3. Digital economy expansion")
           - If none of the options clearly apply, return NULL rather than selecting the closest match
           - Do not try to map similar terms to options - use only the exact wording from the options provided
        7. If a parameter isn't found in the text for a specific policy, leave it as NULL rather than making assumptions.
        8. Arabic text may be present - focus on content you can understand and extract from.
        9. Take advantage of your large context window to understand the document structure and the relationships between sections.
        10. DO NOT try to extract document source information (like filename or folder path) - this will be added by the system.
        11. It's better to return NULL or indicate uncertainty than to hallucinate or guess information not clearly present in the document.
        
        # MULTIPLE POLICY HANDLING:
        When extracting multiple policies from the same document:
        - Ensure each policy is distinct and complete
        - Do not duplicate information between policies unless it genuinely applies to both
        - Include all policies that meet the economic policy criteria
        - Return each complete policy as a separate entity in your response
        
        Remember that your goal is to extract factual information present in the document,
        not to make assumptions or generate content that isn't explicitly stated.
        """
        
        self.agent = Agent(
            self.model,
            result_type=AgentResponse,
            system_prompt=system_prompt
        )
    
    async def process_chunk(self, chunk_text: str, source_info: Dict[str, Any]) -> AgentResponse:
        """Process a single document chunk with the agent"""
        try:
            # Add document source information at the beginning of the text
            source_header = f"""
                    Document Information:
                    - Filename: {source_info['filename']}
                    - Location: {source_info['folder_path']}
                    - Pages: {', '.join(map(str, source_info['page_numbers']))}
                    - Chunk: {source_info['chunk_number']} of {source_info['total_chunks']}

                    Document Content:
                    ==================
                    """
            # Prepend source information to the document text
            text_with_source = source_header + chunk_text

            # Run the agent with the modified text and model_settings for temperature control
            result = await self.agent.run(
                text_with_source,
                model_settings=self.model_settings  # Pass temperature here
            )
            
            # If policies were returned, make sure each one has required fields
            if result.data.result.has_policy and result.data.result.policies:
                # Add source info to each policy
                for policy in result.data.result.policies:
                    if not hasattr(policy, 'source') or policy.source is None:
                        # Source info will be added later in process_document
                        pass
            
            return result.data
            
        except Exception as e:
            # Handle errors gracefully
            print(f"Error processing chunk: {str(e)}")
            traceback.print_exc()
            
            # Return an error response
            return AgentResponse(
                message=f"Error processing document chunk: {str(e)}",
                result=ExtractionResult(
                    error=str(e),
                    has_policy=False,
                    policy_count=0
                )
            )
    
    async def extract_document_info(self, document_text: str, source_info: Dict[str, Any]) -> Dict[str, str]:
        """Extract general information about a document that doesn't contain economic policies"""
        try:
            # Create a prompt specifically for document classification
            prompt = f"""
            You are analyzing a document that does NOT contain economic policies.
            Please extract the following detailed information about this document:
            
            1. Document Type: What type of document is this? (e.g., circular, announcement, report, guideline, administrative decision, etc.)
            2. Main Subject: What is the main subject or topic of this document? Be specific and detailed.
            3. Document Purpose: What is the primary purpose or objective of this document?
            4. Key Points: What are 2-3 key points or main provisions in this document?
            5. Issuing Entity: Which organization or entity issued this document? Include department if specified.
            6. Target Audience: Who is the intended audience for this document? Be specific.
            7. Effective Date: When does this document take effect (if mentioned)?
            8. Reason Not Economic Policy: Explain specifically why this is not considered an economic policy document.
            
            Document: {source_info['filename']}
            
            Provide your answers in a concise but informative format. Include specific details from the document.
            """
            
            # Run a separate query to get document information
            result = await self.agent.run(
                prompt + "\n\n" + document_text[:15000],  # Use more of the document for better analysis
                model_settings=self.model_settings
            )
            
            # Extract information from the response
            message = result.data.message
            
            # Parse the response to extract key information
            info = {
                "document_type": extract_field(message, "Document Type:"),
                "main_subject": extract_field(message, "Main Subject:"),
                "document_purpose": extract_field(message, "Document Purpose:"),
                "key_points": extract_field(message, "Key Points:"),
                "issuing_entity": extract_field(message, "Issuing Entity:"),
                "target_audience": extract_field(message, "Target Audience:"),
                "effective_date": extract_field(message, "Effective Date:"),
                "reason_not_policy": extract_field(message, "Reason Not Economic Policy:")
            }
            
            return info
            
        except Exception as e:
            print(f"Error extracting document info: {str(e)}")
            return {
                "document_type": "Unknown",
                "main_subject": "Unknown",
                "document_purpose": "Unknown",
                "key_points": "Unknown",
                "issuing_entity": "Unknown",
                "target_audience": "Unknown",
                "effective_date": "Unknown",
                "reason_not_policy": f"Error during analysis: {str(e)}"
            }
    
    async def process_document(self, file_path: str) -> List[Policy]:
        """Process a complete document and extract policies"""
        print(f"\n===== Processing document: {file_path} =====")
        print(f"Using context window of {self.config.MAX_TOKENS} tokens")
        print(f"Chunk size: {self.config.CHUNK_SIZE} tokens with {self.config.CHUNK_OVERLAP} token overlap")
        
        # Extract document information
        filename = os.path.basename(file_path)
        folder_path = os.path.dirname(file_path)
        
        # Extract text from PDF
        print(f"Extracting text from PDF...")
        pages = extract_text_from_pdf(file_path)
        if not pages:
            print(f"No content extracted from {file_path}")
            return []
        
        total_text_length = sum(len(page["text"]) for page in pages)
        estimated_tokens = total_text_length // 4  # Rough estimate of token count
        print(f"Extracted {len(pages)} pages with approximately {estimated_tokens} tokens")
        
        # Create chunks from document
        chunks = chunk_document(
            pages, 
            chunk_size=self.config.CHUNK_SIZE, 
            overlap=self.config.CHUNK_OVERLAP
        )
        print(f"Document split into {len(chunks)} chunks for processing")
        
        # Process the document based on its size
        policies = []
        
        # Combine all pages into a single text for document-level analysis if needed later
        full_document_text = "\n\n".join([page["text"] for page in pages])
        
        # Try to process entire document if it fits in context window
        if estimated_tokens <= self.config.MAX_TOKENS * 0.9:  # Leave 10% margin
            print("\nDocument fits in context window - processing entire document holistically...")
            
            # Process as a single chunk
            source_info = {
                "filename": filename,
                "folder_path": folder_path,
                "page_numbers": list(range(1, len(pages) + 1)),
                "chunk_number": 1,
                "total_chunks": 1
            }
            
            print(f"Sending entire document to AI for extraction...")
            response = await self.process_chunk(full_document_text, source_info)
            
            # If policies were found, add them to the results
            if response.result.has_policy and response.result.policies:
                # Add source information to each policy
                for policy in response.result.policies:
                    # Add source information
                    policy.source = DocumentSource(
                        filename=filename,
                        folder_path=folder_path,
                        page_numbers=source_info["page_numbers"]
                    )
                    
                    # Only add if we have a policy name (required field)
                    if policy.policy_name and policy.policy_name.strip():
                        policies.append(policy)
                        print(f"✓ Extracted policy: {policy.policy_name}")
                        # Print fields that were successfully extracted
                        filled_fields = [field for field, value in policy.get_valid_fields().items() 
                                       if value is not None and value != ""]
                        print(f"  Fields extracted: {', '.join(filled_fields)}")
                    else:
                        print(f"✗ Policy detected but name was missing")
                
                # Print information about the number of policies
                if response.result.multiple_policies:
                    print(f"✓ Extracted {len(response.result.policies)} policies from document")
            else:
                print(f"No policies detected in document")
        
        else:
            print("\nDocument too large for single context window - processing in chunks...")
            # First, create a document summary by combining first paragraphs from each chunk
            summary_text = ""
            for i, chunk in enumerate(chunks[:min(len(chunks), 5)]):  # Use first 5 chunks for summary
                chunk_start = chunk["text"][:500]  # Take first 500 chars from each chunk
                summary_text += f"Chunk {i+1} start: {chunk_start}\n\n"
            
            print("\nCreating document overview to guide extraction...")
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_text_length = len(chunk["text"]) 
                print(f"\nProcessing chunk {i+1}/{len(chunks)}... (approx. {chunk_text_length // 4} tokens, pages {chunk['pages']})")
                
                # Prepare source information
                source_info = {
                    "filename": filename,
                    "folder_path": folder_path,
                    "page_numbers": chunk["pages"],
                    "chunk_number": i+1,
                    "total_chunks": len(chunks)
                }
                
                # Prepend document summary to each chunk for context
                enhanced_chunk_text = f"""
DOCUMENT OVERVIEW (to help understand context):
Document: {filename}
Total Pages: {len(pages)}
This chunk: Pages {', '.join(map(str, chunk['pages']))} (Chunk {i+1} of {len(chunks)})

{summary_text}

--- CURRENT CHUNK CONTENT (Pages {', '.join(map(str, chunk['pages']))}) ---
{chunk["text"]}
"""
                
                # Process the chunk
                print(f"Sending to AI for extraction...")
                response = await self.process_chunk(enhanced_chunk_text, source_info)
                
                # If policies were found, add them to the results
                if response.result.has_policy and response.result.policies:
                    # Add source information to each policy
                    for policy in response.result.policies:
                        # Add source information
                        policy.source = DocumentSource(
                            filename=filename,
                            folder_path=folder_path,
                            page_numbers=chunk["pages"]
                        )
                        
                        # Only add if we have a policy name (required field)
                        if policy.policy_name and policy.policy_name.strip():
                            policies.append(policy)
                            print(f"✓ Extracted policy: {policy.policy_name}")
                            # Print fields that were successfully extracted
                            filled_fields = [field for field, value in policy.get_valid_fields().items() 
                                            if value is not None and value != ""]
                            print(f"  Fields extracted: {', '.join(filled_fields)}")
                        else:
                            print(f"✗ Policy detected but name was missing")
                
                    # Print information about the number of policies
                    if response.result.multiple_policies:
                        print(f"✓ Extracted {len(response.result.policies)} policies from chunk {i+1}")
                else:
                    print(f"No policy detected in chunk {i+1}")
        
        # If no policies were found, create an enhanced "NOT A POLICY" entry with additional information
        if not policies:
            print(f"No economic policies found. Extracting general document information...")
            
            # Extract additional information about the document
            source_info = {
                "filename": filename,
                "folder_path": folder_path,
                "page_numbers": list(range(1, len(pages) + 1))
            }
            
            # Get document information
            doc_info = await self.extract_document_info(full_document_text, source_info)
            
            print(f"Creating enhanced 'NOT A POLICY' entry for document: {filename}")
            print(f"  Document Type: {doc_info['document_type']}")
            print(f"  Main Subject: {doc_info['main_subject']}")
            
            # Create a NOT A POLICY entry with the additional information
            not_policy = Policy(
                policy_name="NOT A POLICY",
                policy_summary=f"Document Type: {doc_info['document_type']} | Main Subject: {doc_info['main_subject']} | Purpose: {doc_info['document_purpose']} | Key Points: {doc_info['key_points']}",
                goals=doc_info['document_purpose'],
                objectives=doc_info['key_points'],
                situation_analysis=doc_info['reason_not_policy'],
                issuing_year=doc_info['effective_date'] if doc_info['effective_date'] and doc_info['effective_date'] != "Not mentioned" else None,
                policy_owner=doc_info['issuing_entity'],
                stakeholders_involved=doc_info['target_audience'],
                sector="Multi-sector" if doc_info['document_type'] else None,  # Use Multi-sector as default if we have a document type
                source=DocumentSource(
                    filename=filename,
                    folder_path=folder_path,
                    page_numbers=list(range(1, len(pages) + 1))
                )
            policies.append(not_policy)
        
        print(f"\nExtraction complete. Found {len(policies)} policies in document.")
        return policies

    def deduplicate_policies(self, policies: List[Policy]) -> List[Policy]:
        """Remove duplicate policies based on policy name"""
        unique_policies = {}
        not_policy_entries = []  # Separate list for "NOT A POLICY" entries
        
        for policy in policies:
            # Handle "NOT A POLICY" entries separately
            if policy.policy_name == "NOT A POLICY":
                # Store with filename as key to avoid duplicates from same file
                if policy.source and policy.source.filename:
                    key = f"NOT_A_POLICY_{policy.source.filename}"
                    not_policy_entries.append((key, policy))
                continue
            
            # Normal policy deduplication
            norm_name = policy.policy_name.lower().strip()
            
            if norm_name not in unique_policies:
                unique_policies[norm_name] = policy
            else:
                # If we already have this policy, merge missing fields
                existing = unique_policies[norm_name]
                
                # Merge source page numbers
                if policy.source and existing.source:
                    existing.source.page_numbers = sorted(list(set(
                        existing.source.page_numbers + policy.source.page_numbers
                    )))
                
                # For each field, prefer the existing value unless it's empty
                for field_name, field_value in policy.get_valid_fields().items():
                    existing_value = getattr(existing, field_name, None)
                    if not existing_value and field_value:
                        setattr(existing, field_name, field_value)
        
        # Add "NOT A POLICY" entries to the result
        # Use a dict to deduplicate by filename
        not_policy_dict = dict(not_policy_entries)
        
        # Combine regular policies with "NOT A POLICY" entries
        result = list(unique_policies.values()) + list(not_policy_dict.values())
        
        return result

    def save_policies_to_csv(self, policies: List[Policy], output_filepath: str) -> None:
        """Save policies to CSV file"""
        if not policies:
            print(f"No policies to save.")
            return
        
        # First, get all fields from the Policy model
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
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
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
        
        print(f"✓ Saved {len(policies)} policies to {output_filepath}")

# Main function for CLI usage
async def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract policies from PDF documents')
    parser.add_argument('--input', type=str, default="Document Repo Temp",
                        help='Input directory containing PDF files or a single PDF file')
    parser.add_argument('--output', type=str, default="output",
                        help='Output directory for CSV files')
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    # Validate input path exists
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize agent with config
    config = Config()
    agent = PolicyExtractionAgent(config)
    
    # Check if input is a directory or single file
    all_policies = []
    
    if os.path.isdir(input_path):
        print(f"Processing all PDF files in: {input_path}")
        
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(input_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {input_path}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process.")
        
        # Process each PDF file
        for pdf_file in pdf_files:
            file_path = os.path.join(input_path, pdf_file)
            
            # Process the file
            file_policies = await agent.process_document(file_path)
            all_policies.extend(file_policies)
            
            print(f"Extracted {len(file_policies)} policies from {pdf_file}")
    
    else:
        # Process single file
        print(f"Processing single file: {input_path}")
        
        if not input_path.lower().endswith('.pdf'):
            print(f"Error: Input file must be a PDF file.")
            return
        
        all_policies = await agent.process_document(input_path)
    
    # Check if any policies were found
    if not all_policies:
        print("\nNo policies were extracted from any of the processed documents.")
        return
    
    # Deduplicate policies based on policy name
    unique_policies = agent.deduplicate_policies(all_policies)
    
    # Save all results to a single CSV file
    if unique_policies:
        output_filepath = os.path.join(output_path, "extracted_policies.csv")
        agent.save_policies_to_csv(unique_policies, output_filepath)
        print(f"\nExtraction complete. Saved {len(unique_policies)} unique policies to {output_filepath}")
    else:
        print("\nNo unique policies were found after deduplication.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 