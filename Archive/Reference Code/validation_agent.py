from typing import List, Optional, Union, Dict, Any, cast
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from datetime import datetime, date
import traceback

class SimilarRequest(BaseModel):
    id: int
    objective: str
    business_domain: str
    similarity_score: float

class ActionData(BaseModel):
    title: str
    data: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True

class ValidationResponse(BaseModel):
    valid: bool = Field(description="Whether the request is valid")
    message: str = Field(description="Validation message")
    similar_requests: List[SimilarRequest] = Field(default_factory=list)
    department_access: bool = Field(description="Whether user has department access")
    store_success: bool = Field(default=False)
    needs_confirmation: bool = Field(default=False, description="Whether user confirmation is needed")
    suggested_actions: List[Union[str, ActionData]] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True

class SimilarityResponse(BaseModel):
    """Response model for similarity analysis"""
    message: str = Field(description="Analysis of similarities between requests")

class ValidationAgent:
    def __init__(self, model, db):
        with open(r"/Users/karthik/Projects/Github/teams-ai-agent/teams/src/prompts/validation_agent_prompt.txt") as f:
            system_prompt = f.read()
            
        self.model = model
        self.agent = Agent(
            model,
            result_type=ValidationResponse,
            system_prompt=system_prompt
        )
        self.db = db
        self.user_id = 0  # Default user ID
        self._setup_tools()
    
    def _setup_tools(self):
        @self.agent.tool
        async def find_similar_requests(ctx: RunContext, objective: str) -> List[SimilarRequest]:
            """Find similar existing requests using text similarity"""
            try:
                query = """
                SELECT id, objective, business_domain,
                       similarity(objective, $1) as score
                FROM report_requests 
                WHERE similarity(objective, $1) > 0.3
                ORDER BY score DESC
                LIMIT 5;
                """
                results = await self.db.fetch(query, objective)
                return [
                    SimilarRequest(
                        id=r['id'],
                        objective=r['objective'],
                        business_domain=r['business_domain'],
                        similarity_score=r['score']
                    ) for r in results
                ]
            except Exception as e:
                print(f"Error finding similar requests: {e}")
                return []

        @self.agent.tool
        async def check_department_access(
            ctx: RunContext, 
            user_id: int, 
            department: str
        ) -> bool:
            """Check if user has access to create requests for given department"""
            try:
                query = """
                SELECT d.name as dept_name
                FROM users u
                JOIN departments d ON u.department_id = d.id
                WHERE u.id = $1;
                """
                result = await self.db.fetch(query, user_id)
                if not result:
                    return False
                return result[0]['dept_name'].lower() == department.lower()
            except Exception as e:
                print(f"Error checking department access: {e}")
                # For testing, allow all access
                return True

        @self.agent.tool
        async def store_request(ctx: RunContext, request_data: dict) -> bool:
            """Store the validated request in database"""
            try:
                print(f"Attempting to store request with data: {request_data}")
                
                # Validate required fields
                required_fields = ['business_domain', 'objective']
                for field in required_fields:
                    if not request_data.get(field):
                        print(f"Missing required field: {field}")
                        return False

                # Clean up the data before storing
                cleaned_data = {
                    'business_domain': request_data.get('business_domain'),
                    'business_domain_other': request_data.get('business_domain_other'),
                    'objective': request_data.get('objective'),
                    'platform': request_data.get('platform'),
                    'frequency': request_data.get('frequency'),
                    'data_sources': request_data.get('data_sources'),
                    'stakeholders': request_data.get('stakeholders'),
                    'required_by': None,  # Initialize as None
                    'requester_id': 1  # For testing, use a default user ID
                }

                # Convert date string to proper format if present
                if request_data.get('required_by'):
                    try:
                        # Parse the date string to datetime object
                        required_by = datetime.strptime(request_data['required_by'], '%Y-%m-%d')
                        cleaned_data['required_by'] = required_by
                    except ValueError as e:
                        print(f"Error parsing date: {e}")
                        cleaned_data['required_by'] = None

                query = """
                INSERT INTO report_requests (
                    business_domain, business_domain_other, objective,
                    platform, frequency, data_sources, stakeholders,
                    required_by, requester_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id;
                """
                
                await self.db.execute(
                    query,
                    cleaned_data['business_domain'],
                    cleaned_data['business_domain_other'],
                    cleaned_data['objective'],
                    cleaned_data['platform'],
                    cleaned_data['frequency'],
                    cleaned_data['data_sources'],
                    cleaned_data['stakeholders'],
                    cleaned_data['required_by'],
                    cleaned_data['requester_id']
                )
                return True
                
            except Exception as e:
                print(f"Error storing request: {e}")
                print(f"Request data was: {cleaned_data}")
                return False

    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string to date object"""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"Invalid date format: {date_str}")
            return None

    async def validate_request(self, request_data: dict) -> ValidationResponse:
        """Validate a new report request"""
        try:
            print("\n=== Validation Process Started ===")
            
            # First check department access
            print("Checking department access...")
            has_access, policy_message = await self._check_department_access(
                request_data.get('requester_id'),
                request_data.get('business_domain')
            )
            print(f"Department access result: {has_access}")
            
            if not has_access:
                print("Department access denied - returning policy citation")
                return ValidationResponse(
                    valid=True,
                    needs_confirmation=True,
                    message=(
                        f"{policy_message}\n\n"
                        "Would you like to:"
                    ),
                    department_access=False,
                    suggested_actions=[
                        ActionData(
                            title="Submit anyway (Not Recommended)",
                            data={
                                "verb": "cross_department_submit",
                                "policy_violation": True
                            }
                        ),
                        ActionData(
                            title="❌ Cancel",
                            data={
                                "verb": "cross_department_cancel",
                                "policy_cited": True
                            }
                        )
                    ]
                )

            # Then check for similar requests
            print("Checking for similar requests...")
            similar_requests = await self._find_similar_requests(
                request_data.get('objective', ''),
                request_data.get('business_domain', '')
            )
            print(f"Found {len(similar_requests)} similar requests")
            
            if similar_requests:
                print("Similar requests found - returning confirmation request")
                similarity_details = await self._analyze_similarities(
                    request_data.get('objective', ''),
                    similar_requests
                )
                return ValidationResponse(
                    valid=True,
                    needs_confirmation=True,
                    message=(
                        "📋 I found similar existing report requests:\n\n"
                        f"{similarity_details}\n\n"
                        "Would you like to:"
                    ),
                    similar_requests=similar_requests,
                    department_access=True,
                    suggested_actions=[
                        "Submit as New Request", 
                        "View Existing Reports", 
                        ActionData(
                            title="❌ Cancel",
                            data={
                                "verb": "validation_cancel",
                                "from_validation": True,
                                "action_taken": True,
                                "message": similarity_details
                            }
                        )
                    ]
                )

            # If all checks pass, proceed with saving
            print("All validation checks passed")
            return ValidationResponse(
                valid=True,
                needs_confirmation=False,
                message="",
                department_access=True
            )

        except Exception as e:
            print(f"Error in validate_request: {e}")
            traceback.print_exc()
            return ValidationResponse(
                valid=False,
                message="⚠️ Error validating request. Please try again.",
                department_access=False
            ) 

    async def _check_department_access(self, user_id: str, department: str) -> tuple[bool, str]:
        """Check if user has access to create requests for given department"""
        try:
            print(f"\n=== Checking Department Access ===")
            print(f"User ID: {user_id}")
            print(f"Requested Department: {department}")
            
            # For HR department, always grant access since it's a central function
            if department.upper() == "HR":
                print("HR department access granted automatically")
                return True, ""
            
            # Get user's department from database
            query = """
            SELECT department 
            FROM users 
            WHERE id = $1
            """
            result = await self.db.fetchrow(query, self.user_id)
            user_department = result['department'] if result else "HR"  # Default to HR if not found
            
            # Check if request is for their department
            has_access = user_department.lower() == department.lower()
            
            if not has_access:
                policy_citation = (
                    "🔒 Cross-Department Access Policy (SC-3)\n\n"
                    "According to our Data Access Governance Policy:\n"
                    "• Each department maintains ownership of their data and reporting requirements\n"
                    "• Cross-department requests require explicit coordination with the target department\n"
                    "• This policy ensures:\n"
                    "  - Proper data handling and security\n"
                    "  - Alignment with department-specific requirements\n"
                    "  - Clear ownership and accountability\n\n"
                    "Reference: Data Governance Framework v2.1, Section SC-3"
                )
                return False, policy_citation
            
            return True, ""
            
        except Exception as e:
            print(f"Error checking department access: {e}")
            traceback.print_exc()
            return False, "Error checking department access"

    async def _find_similar_requests(self, objective: str, business_domain: str) -> List[SimilarRequest]:
        """Find similar existing requests"""
        try:
            print(f"\n=== Finding Similar Requests ===")
            print(f"Objective: {objective}")
            print(f"Business Domain: {business_domain}")
            
            # Simple query to get potential matches from same business domain
            query = """
            SELECT 
                id,
                objective,
                business_domain
            FROM report_requests
            WHERE business_domain = $1
            ORDER BY created_at DESC
            LIMIT 10;
            """
            
            results = await self.db.fetch(query, business_domain)
            print(f"Found {len(results)} potential requests to analyze")
            
            if not results:
                return []

            # Let LLM determine similarity
            similarity_agent = Agent(
                self.model,
                result_type=SimilarityResponse,
                system_prompt="""
                Analyze if the new request is similar to any existing requests.
                Consider:
                1. Similar objectives or metrics being tracked
                2. Similar business purposes
                3. Similar data needs or reporting requirements
                
                Return only the truly similar requests with a similarity score between 0.0 and 1.0.
                Higher scores (closer to 1.0) indicate more similarity.
                """
            )
            
            similar_requests = []
            for r in results:
                score_result = await similarity_agent.run(
                    f"""
                    Compare these two requests and determine their similarity:
                    
                    New request: {objective}
                    Existing request: {r['objective']}
                    
                    Return only a number between 0.0 and 1.0 representing similarity,
                    where 1.0 means identical and 0.0 means completely different.
                    Just return the number, e.g. "0.8"
                    """
                )
                
                try:
                    score = float(score_result.data.message.strip())
                    if score > 0.5:  # Only include if similarity is significant
                        similar_requests.append(
                            SimilarRequest(
                                id=r['id'],
                                objective=r['objective'],
                                business_domain=r['business_domain'],
                                similarity_score=score
                            )
                        )
                        print(f"Similar request found: {r['objective']} (score: {score})")
                except ValueError:
                    continue
            
            # Sort by similarity score
            similar_requests.sort(key=lambda x: x.similarity_score, reverse=True)
            return similar_requests
            
        except Exception as e:
            print(f"Error finding similar requests: {e}")
            traceback.print_exc()
            return []

    async def store_request(self, request_data: dict) -> bool:
        """Store the request in the database"""
        try:
            # Check if user exists
            user_exists = await self.db.fetchrow(
                "SELECT id FROM users WHERE id = $1",
                self.user_id
            )

            if not user_exists:
                print(f"Creating default user with ID {self.user_id}")
                await self.db.execute("""
                    INSERT INTO users (id, name, email, department)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO NOTHING
                """, self.user_id, "Default User", f"user{self.user_id}@company.com", "HR")

            # Parse the required_by date
            required_by_date = self._parse_date(request_data.get('required_by'))

            # Store the report request
            await self.db.execute("""
                INSERT INTO report_requests (
                    business_domain,
                    business_domain_other,
                    objective,
                    platform,
                    frequency,
                    data_sources,
                    stakeholders,
                    required_by,
                    requester_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, 
            request_data['business_domain'],
            request_data.get('business_domain_other'),
            request_data['objective'],
            request_data.get('platform'),
            request_data.get('frequency'),
            request_data.get('data_sources'),
            request_data.get('stakeholders'),
            required_by_date,  # Now passing a proper date object
            self.user_id)

            return True

        except Exception as e:
            print(f"Error storing request: {e}")
            traceback.print_exc()
            return False
    async def get_all_requests(self) -> List[dict]:
        """Fetch all report requests from the database"""
        query = """
        SELECT 
            id, business_domain, business_domain_other, objective,
            platform, frequency, data_sources, stakeholders,
            required_by, requester_id, created_at
        FROM report_requests
        ORDER BY created_at DESC;
        """
        try:
            results = await self.db.fetch(query)
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error fetching requests: {e}")
            return []

    async def get_request_by_id(self, request_id: int) -> Optional[dict]:
        """Fetch a specific report request by ID"""
        query = """
        SELECT 
            id, business_domain, business_domain_other, objective,
            platform, frequency, data_sources, stakeholders,
            required_by, requester_id, created_at
        FROM report_requests
        WHERE id = $1;
        """
        try:
            result = await self.db.fetchrow(query, request_id)
            return dict(result) if result else None
        except Exception as e:
            print(f"Error fetching request {request_id}: {e}")
            return None 

    async def get_request_status(self, requester_id: Union[str, int], limit: int = 5) -> List[dict]:
        """Get status of recent report requests for a user"""
        try:
            # Ensure requester_id is an integer
            if isinstance(requester_id, str):
                try:
                    requester_id = int(''.join(filter(str.isdigit, requester_id)))
                except ValueError:
                    print(f"Invalid requester_id format: {requester_id}")
                    return []

            query = """
            SELECT 
                r.id,
                r.business_domain,
                r.objective,
                r.platform,
                r.frequency,
                r.data_sources,
                r.stakeholders,
                r.status,
                r.created_at,
                u.name as requester_name,
                CASE 
                    WHEN r.status = 'pending' THEN 'Pending Review'
                    WHEN r.status = 'approved' THEN 'Approved'
                    WHEN r.status = 'in_progress' THEN 'In Progress'
                    WHEN r.status = 'completed' THEN 'Completed'
                    ELSE r.status
                END as status_display,
                to_char(r.created_at, 'YYYY-MM-DD HH24:MI:SS') as formatted_date
            FROM report_requests r
            LEFT JOIN users u ON r.requester_id = u.id
            WHERE r.requester_id = $1
            ORDER BY r.created_at DESC
            LIMIT $2;
            """
            
            results = await self.db.fetch(query, requester_id, limit)
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error fetching request status: {e}")
            traceback.print_exc()
            return []

    async def get_request_details(self, request_number: int, requester_id: Union[str, int]) -> Optional[dict]:
        """Get detailed information about a specific request"""
        try:
            print(f"Fetching details for request #{request_number} for user {requester_id}")
            
            # Ensure requester_id is an integer
            if isinstance(requester_id, str):
                try:
                    requester_id = int(''.join(filter(str.isdigit, requester_id)))
                except ValueError:
                    print(f"Invalid requester_id format: {requester_id}")
                    return None

            query = """
            WITH numbered_requests AS (
                SELECT 
                    r.*,
                    u.name as requester_name,
                    CASE 
                        WHEN r.status = 'pending' THEN 'Pending Review'
                        WHEN r.status = 'approved' THEN 'Approved'
                        WHEN r.status = 'in_progress' THEN 'In Progress'
                        WHEN r.status = 'completed' THEN 'Completed'
                        ELSE r.status
                    END as status_display,
                    to_char(r.created_at, 'YYYY-MM-DD HH24:MI:SS') as formatted_date,
                    ROW_NUMBER() OVER (ORDER BY r.created_at DESC) as row_num
                FROM report_requests r
                LEFT JOIN users u ON r.requester_id = u.id
                WHERE r.requester_id = $1
            )
            SELECT * FROM numbered_requests WHERE row_num = $2;
            """
            
            result = await self.db.fetchrow(query, requester_id, request_number)
            if result:
                return dict(result)
            print(f"No request found with number {request_number} for user {requester_id}")
            return None
            
        except Exception as e:
            print(f"Error fetching request details: {e}")
            traceback.print_exc()
            return None

    async def _analyze_similarities(self, new_objective: str, similar_requests: List[SimilarRequest]) -> str:
        """Use LLM to analyze and explain similarities between requests"""
        try:
            # Create a new agent for similarity analysis
            similarity_agent = Agent(
                self.model,
                result_type=SimilarityResponse,
                system_prompt="""
                Analyze the similarities between report requests and explain them clearly.
                Focus on:
                1. Common objectives or metrics
                2. Similar business purposes
                3. Overlapping data needs
                
                Format the response as a bulleted list.
                """
            )
            
            result = await similarity_agent.run(
                f"""
                New request: {new_objective}
                
                Existing similar requests:
                {chr(10).join(f'- {r.objective}' for r in similar_requests)}
                
                Explain the similarities:
                """
            )
            return result.data.message
        except Exception as e:
            print(f"Error analyzing similarities: {e}")
            traceback.print_exc()
            return "Similar reports found based on objective and domain." 
