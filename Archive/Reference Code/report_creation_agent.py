from typing import Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from datetime import datetime
import traceback
from report_request import ReportRequest
from models.responses import TeamsResponse

class ReportCreationAgent:
    def __init__(self, model):
        with open(r"/Users/karthik/Projects/Github/teams-ai-agent/teams/src/prompts/report_creation_agent_prompt.txt") as f:
            system_prompt = f.read()
            
        self.model = model
        self.agent = Agent(
            model,
            result_type=TeamsResponse,
            system_prompt=system_prompt
        )
        self.required_fields = ['business_domain', 'objective', 'frequency']
        
    async def run(self, message: str, message_history: list = None, deps: dict = None):
        """Process user message and guide report creation"""
        try:
            # Initialize response data
            response_data = {
                "message": "",
                "show_card": False,
                "business_domain": "",
                "objective": "",
                "platform": "",
                "frequency": "",
                "stakeholders": "",
                "data_sources": "",
                "required_by": ""
            }
            
            # Extract information from message history
            collected_info = self._extract_from_history(message_history or [])
            response_data.update(collected_info)
            
            # Process current message
            current_info = self._extract_from_message(message)
            
            # Process date requirement if present
            if "required_by" in current_info and current_info["required_by"]:
                processed_date = await self.process_date_requirement(current_info["required_by"])
                current_info["required_by"] = processed_date
            
            response_data.update(current_info)
            
            # Determine next question based on missing information
            missing_fields = self._get_missing_fields(response_data)
            
            if not missing_fields:
                # All required information collected
                response_data["show_card"] = True
                response_data["message"] = "Great, I'll prepare the form with these details."
            else:
                # Ask for next missing field
                response_data["message"] = self._get_next_question(missing_fields[0])
            
            return self.agent.create_result(response_data)
            
        except Exception as e:
            print(f"Error in report creation: {e}")
            traceback.print_exc()
            return self.agent.create_result({
                "message": "I apologize, but I encountered an issue. Could you please try again?",
                "show_card": False
            })

    def _extract_from_history(self, history: list) -> dict:
        """Extract information from message history"""
        info = {
            "business_domain": "",
            "objective": "",
            "platform": "",
            "frequency": "",
            "stakeholders": "",
            "data_sources": "",
            "required_by": ""
        }
        
        for msg in history:
            if isinstance(msg, dict):
                # Update info with any previously collected information
                for key in info:
                    if msg.get(key):
                        info[key] = msg[key]
        
        return info

    def _extract_from_message(self, message: str) -> dict:
        """Extract information from current message"""
        info = {}
        
        # Extract business domain
        if any(domain.lower() in message.lower() for domain in ["HR", "Finance", "Operations", "Supply Chain"]):
            for domain in ["HR", "Finance", "Operations", "Supply Chain"]:
                if domain.lower() in message.lower():
                    info["business_domain"] = domain
                    break
        
        # Extract objective
        if "headcount" in message.lower() or "head count" in message.lower():
            info["objective"] = "Track and monitor employee headcount metrics"
        
        # Extract platform
        if "power bi" in message.lower():
            info["platform"] = "Power BI"
        elif "tableau" in message.lower():
            info["platform"] = "Tableau"
            
        # Extract frequency
        if "daily" in message.lower():
            info["frequency"] = "Daily"
        elif "weekly" in message.lower():
            info["frequency"] = "Weekly"
        elif "monthly" in message.lower():
            info["frequency"] = "Monthly"
            
        # Extract stakeholders
        if "internal team" in message.lower():
            info["stakeholders"] = "Internal team"
            
        return info

    def _get_missing_fields(self, data: dict) -> list:
        """Get list of required fields that are still missing"""
        return [field for field in self.required_fields if not data.get(field)]

    def _get_next_question(self, field: str) -> str:
        """Get appropriate question for the next missing field"""
        questions = {
            "business_domain": "Which business domain does this report fall under?",
            "objective": "What would you like to monitor or analyze in this report?",
            "frequency": "How often would you like this report to be refreshed?"
        }
        return questions.get(field, "Could you provide more details about your request?")

    async def process_date_requirement(self, date_text: str) -> str:
        """Use LLM to process and standardize date requirements"""
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        prompt = f"""
        Current date and time: {current_datetime}
        User requested date: {date_text}

        Convert the user's date requirement into a standardized YYYY-MM-DD format.
        Rules:
        - If it's "ASAP" or "earliest possible", return empty string
        - For "end of month/quarter/year", use the last day of that period
        - For relative dates (e.g., "in 2 weeks"), calculate from current date
        - If date is ambiguous or invalid, return empty string
        - For month names, assume current year if not specified
        
        Return ONLY the date in YYYY-MM-DD format or empty string.
        """
        
        try:
            result = await self.agent.run(prompt)
            processed_date = result.data.strip()
            
            # Validate the returned date format
            if processed_date:
                try:
                    datetime.strptime(processed_date, "%Y-%m-%d")
                    return processed_date
                except ValueError:
                    return ""
            return ""
            
        except Exception as e:
            print(f"Error processing date: {e}")
            return "" 