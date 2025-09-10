from typing import Union, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from datetime import datetime
import traceback

class SQLQuery(BaseModel):
    """SQL query to be executed"""
    query: str = Field(description="The SQL query to execute")
    explanation: str = Field(description="Explanation of what the query does")

class AdminResponse(BaseModel):
    """Response from the admin agent"""
    message: str = Field(description="Response message to user")
    sql_query: Optional[SQLQuery] = Field(None, description="SQL query if database access needed")
    error: Optional[str] = Field(None, description="Error message if something went wrong")

class AdminAgent:
    def __init__(self, model, db):
        with open(r"/Users/karthik/Projects/Github/teams-ai-agent/teams/src/prompts/admin_agent_prompt.txt") as f:
            system_prompt = f.read()
            
        self.model = model
        self.agent = Agent(
            model,
            result_type=AdminResponse,
            system_prompt=system_prompt
        )
        self.db = db
        self._setup_tools()

    def _setup_tools(self):
        @self.agent.tool
        async def execute_query(ctx: RunContext, query: str) -> List[dict]:
            """Execute a SQL query and return results"""
            try:
                results = await self.db.fetch(query)
                return [dict(row) for row in results]
            except Exception as e:
                print(f"Error executing query: {e}")
                return []

    def _format_results(self, rows: List[dict]) -> str:
        """Format query results into readable text"""
        if not rows:
            return "No results found"
            
        # Get column names and calculate widths
        columns = list(rows[0].keys())
        widths = {col: max(len(col), max(len(str(row[col])) for row in rows)) for col in columns}
        
        # Create header with proper spacing
        header = " | ".join(f"{col.title():{widths[col]}}" for col in columns)
        separator = "-" * len(header)
        
        # Format each row
        formatted_rows = []
        for row in rows:
            # Format datetime objects
            row_data = {}
            for col, value in row.items():
                if isinstance(value, datetime):
                    row_data[col] = value.strftime("%Y-%m-%d %H:%M")
                else:
                    row_data[col] = str(value)
            
            # Create formatted row
            formatted_row = " | ".join(
                f"{row_data[col]:{widths[col]}}" for col in columns
            )
            formatted_rows.append(formatted_row)
        
        # Combine all parts with proper spacing
        table = f"\n{header}\n{separator}\n" + "\n".join(formatted_rows)
        
        # Add summary
        summary = f"\nFound {len(rows)} {'report' if len(rows) == 1 else 'reports'}"
        
        return f"{table}\n{summary}"

    async def process_admin_query(self, query: str) -> AdminResponse:
        """Process an admin query and return response"""
        try:
            print(f"\n=== Processing Admin Query ===")
            print(f"Query: {query}")
            
            current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            context = f"Current datetime: {current_dt}\nQuery: {query}"
            
            print("Generating SQL query...")
            result = await self.agent.run(context)
            print(f"Generated Response: {result.data}")
            
            # If SQL query is generated, execute it
            if result.data.sql_query:
                try:
                    print(f"Executing SQL: {result.data.sql_query.query}")
                    query_results = await self.db.fetch(result.data.sql_query.query)
                    print(f"Query returned {len(query_results) if query_results else 0} results")
                    
                    # Format results into readable message
                    if query_results:
                        rows = [dict(row) for row in query_results]
                        result.data.message = (
                            f"📊 {result.data.message}\n"
                            f"{self._format_results(rows)}"
                        )
                    else:
                        result.data.message = "No results found for your query."
                        
                except Exception as e:
                    print(f"Error executing query: {e}")
                    traceback.print_exc()
                    result.data.error = f"Error executing query: {str(e)}"
            else:
                print("No SQL query generated")
            
            return result.data
            
        except Exception as e:
            print(f"Error processing admin query: {e}")
            traceback.print_exc()
            return AdminResponse(
                message="Error processing your query",
                error=str(e)
            ) 