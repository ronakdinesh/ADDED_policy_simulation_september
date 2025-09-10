import os
from dataclasses import dataclass
from typing import Optional, Dict
from pydantic import BaseModel, Field
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from botbuilder.core import MemoryStorage, TurnContext, MessageFactory, CardFactory
from botbuilder.schema import Activity, Attachment
from teams import Application, ApplicationOptions, TeamsAdapter
from teams.state import TurnState
from config import Config
from card_helper import create_report_form_card, create_status_card, create_disabled_confirmation_card, disable_card_actions, create_disabled_form_card, create_disabled_validation_card
import re
from report_request import ReportRequest
import traceback
from welcome_card import create_welcome_card
from datetime import datetime
from agents.validation_agent import ValidationAgent, ActionData
from db.database import Database, DatabaseConfig
from agents.report_creation_agent import ReportCreationAgent
from models.responses import MasterResponse
import json
from agents.admin_agent import AdminAgent

config = Config()

# Define response structure
class TeamsResponse(BaseModel):
    message: str = Field(description='Message to be sent to the user')
    show_card: bool = Field(default=False, description='Whether to show an adaptive card')
    card_data: Optional[dict] = Field(default=None, description='Data for the adaptive card if needed')
    # Add fields that match the report request structure
    business_domain: Optional[str] = Field(default=None, description='Business domain for the report')
    business_domain_other: Optional[str] = Field(default=None, description='Other business domain if selected')
    objective: Optional[str] = Field(default=None, description='Report objective')
    platform: Optional[str] = Field(default=None, description='Platform for the report (Power BI, Tableau, etc.)')
    frequency: Optional[str] = Field(default=None, description='Report frequency')
    data_sources: Optional[str] = Field(default=None, description='Data sources')
    stakeholders: Optional[str] = Field(default=None, description='Stakeholders')
    required_by: Optional[str] = Field(default=None, description='Required by date')

# Initialize Azure OpenAI client
client = AsyncAzureOpenAI(
    azure_endpoint="https://teams-ai-agent.openai.azure.com",
    api_version="2024-10-21",
    api_key=config.Azure_OPENAI_API_KEY
)

# Create OpenAI model instance with Azure client
model = OpenAIModel(
    'gpt-4o', 
    openai_client=client,
)

# Create storage
storage = MemoryStorage()

# Read master agent prompt
with open(r"/Users/karthik/Projects/Github/teams-ai-agent/teams/src/prompts/master_agent_prompt.txt") as f:
    MASTER_PROMPT = f.read()

# Initialize the application with our Pydantic AI agent
class TeamsAIApplication(Application[TurnState]):
    def __init__(self):
        # Initialize base Application first
        super().__init__(
            ApplicationOptions(
                bot_app_id=config.APP_ID,
                storage=storage,
                adapter=TeamsAdapter(config),
            )
        )
        
        # Initialize database
        db_config = DatabaseConfig()
        self.db = Database(db_config)
        
        # Initialize master agent
        self.model = model  # Store model instance
        self.master_agent = Agent(
            model,
            result_type=MasterResponse,
            system_prompt=MASTER_PROMPT
        )
        
        # Initialize sub-agents
        self.report_creation_agent = ReportCreationAgent(model)
        self.validation_agent = ValidationAgent(model, self.db)
        
        self._setup_tools()
        self._setup_message_handlers()
        self.conversation_states: Dict[str, ReportRequest] = {}
        self.message_histories: Dict[str, list] = {}
        self.has_sent_welcome = {}
        self.pending_submissions: Dict[str, dict] = {}  # Store pending form submissions
        self.conversation_started = {}  # Track if we're in an active conversation
        self.greeting_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in [
                r'^hi$', r'^hello$', r'^hey$', r'^hi\s+there', 
                r'^hello\s+there', r'^greetings', r'^howdy',
                r'^good\s+(morning|afternoon|evening)'
            ]
        ]
        self.admin_modes: Dict[str, bool] = {}
        
        # Initialize admin agent
        self.admin_agent = AdminAgent(model, self.db)

    def is_greeting(self, text: str) -> bool:
        """Check if the message is a greeting"""
        greeting_words = ['hi', 'hello', 'hey', 'greetings', 'howdy']
        return any(word in text.lower().split() for word in greeting_words)

    def clear_conversation_state(self, conversation_id: str):   
        """Clear all conversation state and history for a given conversation"""
        if conversation_id in self.conversation_states:
            del self.conversation_states[conversation_id]
        if conversation_id in self.message_histories:
            del self.message_histories[conversation_id]
        if conversation_id in self.conversation_started:
            del self.conversation_started[conversation_id]
        if conversation_id in self.has_sent_welcome:
            del self.has_sent_welcome[conversation_id]

    def _setup_message_handlers(self):
        async def disable_old_cards(context: TurnContext):
            """Helper to disable any previous cards"""
            try:
                # Get the conversation reference
                conversation_id = context.activity.conversation.id
                
                # If there's a previous activity, disable its card
                if context.activity.reply_to_id:
                    # Create disabled version of the card
                    disabled_card = create_welcome_card(role="Head of HR", disable_actions=True)
                    
                    # Update the previous activity with disabled card
                    await context.update_activity(Activity(
                        id=context.activity.reply_to_id,
                        type="message",
                        attachments=[disabled_card]
                    ))
                    print("Successfully disabled previous card")
                    
            except Exception as e:
                print(f"Error disabling old cards: {str(e)}")
                traceback.print_exc()

        @self.adaptive_cards.action_submit("disable_card")
        async def on_disable_card(context: TurnContext, state: TurnState, data) -> None:
            """Handle the disable card action"""
            if data.get("action_taken"):
                # Create and show disabled version of the card
                if context.activity.reply_to_id:
                    disabled_card = create_welcome_card(role="Head of HR", disable_actions=True)
                    await context.update_activity(Activity(
                        id=context.activity.reply_to_id,
                        type="message",
                        attachments=[disabled_card]
                    ))

        @self.message("/show_form")
        async def show_form(context: TurnContext, state: TurnState) -> bool:
            try:
                # Disable any previous cards
                await disable_old_cards(context)
                
                conversation_id = context.activity.conversation.id
                # Clear all conversation state and history
                self.clear_conversation_state(conversation_id)
                
                # Create a fresh response with HR department pre-filled
                response = TeamsResponse(
                    message="Here is a new report request form",
                    show_card=True,
                    business_domain="HR"
                )
                
                # Send the message first
                await context.send_activity(MessageFactory.text(response.message))
                
                # Show fresh form
                return await self.trigger_form_display(context, state, response)
                
            except Exception as e:
                print(f"Error in show_form: {str(e)}")
                traceback.print_exc()
                await context.send_activity("There was an error displaying the form. Please try again.")
                return False

        @self.message(re.compile(r"(?i)(explain|elaborate|tell me about|details for).*request.*[#]?\s*\d+"))
        async def on_explain_request(context: TurnContext, state: TurnState):
            """Handle requests to explain specific report requests"""
            try:
                # Extract request number from message
                message = context.activity.text.lower()
                # Find all numbers in the message and take the last one
                numbers = re.findall(r'\d+', message)
                if not numbers:
                    await context.send_activity("I couldn't find a request number. Please specify which request you'd like me to explain (e.g., 'explain request #5')")
                    return
                    
                request_num = int(numbers[-1])  # Take the last number found
                print(f"Explaining request #{request_num}")
                
                # Get user ID
                user_id = context.activity.from_property.id
                try:
                    user_id = int(''.join(filter(str.isdigit, user_id)))
                except ValueError:
                    user_id = 1  # Fallback to default user ID
                
                # Get request details
                request = await self.validation_agent.get_request_details(request_num, user_id)
                
                if request:
                    # Create detailed response
                    response = (
                        f"📋 Details for Request #{request_num}:\n\n"
                        f"**Objective:** {request['objective']}\n"
                        f"**Status:** {request.get('status_display', request.get('status', 'Unknown'))}\n"
                        f"**Business Domain:** {request['business_domain']}\n"
                        f"**Platform:** {request.get('platform', 'Not specified')}\n"
                        f"**Frequency:** {request.get('frequency', 'Not specified')}\n"
                        f"**Data Sources:** {request.get('data_sources', 'Not specified')}\n"
                        f"**Stakeholders:** {request.get('stakeholders', 'Not specified')}\n"
                        f"**Created:** {request['formatted_date']}\n"
                        f"**Requested by:** {request.get('requester_name', 'Unknown')}\n\n"
                        "You can ask me to explain another request or create a new one!"
                    )
                    
                    await context.send_activity(MessageFactory.text(response))
                else:
                    await context.send_activity(
                        f"I couldn't find request #{request_num}. Please check the number and try again.\n"
                        "You can see all your requests by typing 'show my requests'"
                    )
                    
            except Exception as e:
                print(f"Error explaining request: {e}")
                traceback.print_exc()
                await context.send_activity("Sorry, I had trouble retrieving those details. Please try again.")

        @self.message("/admin")
        async def on_admin_command(context: TurnContext, state: TurnState) -> bool:
            """Handle admin mode activation"""
            try:
                conversation_id = context.activity.conversation.id
                print(f"\n=== Admin Mode Activated ===")
                print(f"Conversation ID: {conversation_id}")
                print(f"Activity Type: {context.activity.type}")
                print(f"Activity Text: {context.activity.text}")
                
                # Mark conversation as in admin mode
                self.admin_modes[conversation_id] = True
                print(f"Admin mode set to: {self.admin_modes[conversation_id]}")
                
                # Send admin mode message
                await context.send_activity(
                    "🔒 Entering Admin Mode\n\n"
                    "You can now query the database directly. Examples:\n"
                    "- Show me all pending reports\n"
                    "- How many reports per department?\n"
                    "- Show reports created today\n\n"
                    "Type '/exit' to leave admin mode."
                )
                return True
                
            except Exception as e:
                print(f"Error in admin command handler: {e}")
                traceback.print_exc()
                return False

        @self.message("/exit")
        async def on_exit_admin(context: TurnContext, state: TurnState) -> bool:
            """Handle exiting admin mode"""
            conversation_id = context.activity.conversation.id
            if conversation_id in self.admin_modes:
                del self.admin_modes[conversation_id]
                await context.send_activity("✅ Exited admin mode")
            return True

        @self.message(re.compile(".*"))
        async def on_message(context: TurnContext, state: TurnState):
            try:
                print("\n=== New Message ===")
                print(f"Activity Type: {context.activity.type}")
                print(f"Activity Text: {context.activity.text}")
                print(f"Conversation ID: {context.activity.conversation.id}")
                
                # Skip if this is a form submission or system message
                if (hasattr(context.activity, 'value') and context.activity.value) or \
                   context.activity.text.startswith("system:"):
                    print("Skipping form submission or system message")
                    return True

                conversation_id = context.activity.conversation.id
                message_history = self.message_histories.get(conversation_id, [])
                message_lower = context.activity.text.lower()

                print(f"Admin Mode Status: {self.admin_modes.get(conversation_id)}")
                print(f"Conversation Started: {self.conversation_started.get(conversation_id)}")
                print(f"Message History Length: {len(message_history)}")

                # Check if in admin mode first
                if self.admin_modes.get(conversation_id):
                    print("Processing admin query...")
                    result = await self.admin_agent.process_admin_query(context.activity.text)
                    
                    if result.error:
                        print(f"Admin query error: {result.error}")
                        await context.send_activity(f"❌ Error: {result.error}")
                    else:
                        print("Admin query successful")
                        await context.send_activity(result.message)
                    return True

                # Only proceed with normal message handling if not in admin mode
                if not self.conversation_started.get(conversation_id):
                    # Check for report intent first
                    report_keywords = [
                        'report', 'dashboard', 'metrics', 'power bi', 'powerbi', 'tableau',
                        'headcount', 'head count', 'hr', 'daily', 'weekly', 'monthly',
                        'refresh', 'track', 'monitor', 'department', 'team', 'internal'
                    ]
                    contains_report_intent = any(
                        keyword in message_lower.replace('-', ' ') 
                        for keyword in report_keywords
                    )

                    print(f"Debug - Message: {message_lower}")
                    print(f"Debug - Contains report intent: {contains_report_intent}")
                    print(f"Debug - Conversation started: {self.conversation_started.get(conversation_id)}")

                    # If there's report intent, start report creation
                    if contains_report_intent:
                        print("Debug - Direct report intent detected, starting creation process")
                        self.conversation_started[conversation_id] = True

                # Check if this is a greeting or first message
                is_first_message = not message_history
                is_greeting = self.is_greeting(context.activity.text)

                # Only show welcome card if not in admin mode and is greeting/first message
                if (is_greeting or is_first_message) and not self.admin_modes.get(conversation_id):
                    print("Showing welcome card for greeting/first message")
                    # Disable any previous cards first
                    await disable_old_cards(context)
                    
                    # Create and show welcome card
                    welcome_card = create_welcome_card(role="Head of HR")
                    await context.send_activity(Activity(
                        type="message",
                        attachments=[welcome_card]
                    ))
                    return True

                # If we're in an active conversation, continue with report creation
                if self.conversation_started.get(conversation_id):
                    # Forward to report creation agent
                    creation_result = await self.report_creation_agent.agent.run(
                        context.activity.text,
                        message_history=self.message_histories.get(conversation_id, []),
                        deps={}
                    )
                    
                    # Update message history
                    self.message_histories[conversation_id] = creation_result.all_messages()
                    
                    # Send the message
                    if creation_result.data.message:
                        await context.send_activity(MessageFactory.text(creation_result.data.message))
                    
                    # Show form if needed
                    if creation_result.data.show_card:
                        if creation_result.data.card_data:
                            await context.send_activity(Activity(
                                type="message",
                                attachments=[CardFactory.adaptive_card(creation_result.data.card_data)]
                            ))
                        else:
                            await self.trigger_form_display(context, state, creation_result.data)
                    return True

                # Check if in admin mode
                if self.admin_modes.get(conversation_id):
                    # Process as admin query
                    result = await self.admin_agent.process_admin_query(context.activity.text)
                    
                    if result.error:
                        await context.send_activity(f"❌ Error: {result.error}")
                    else:
                        await context.send_activity(result.message)
                    return True

                # Add current datetime to each user message
                current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                user_message = f"[{current_dt}] {context.activity.text}"
                
                # Get response from master agent
                result = await self.master_agent.run(
                    user_message,
                    message_history=message_history
                )
                response = result.data
                print(f"Agent response: {response}")

                # Store new messages for future context
                self.message_histories[conversation_id] = result.all_messages()

                # Check for report intent
                report_keywords = [
                    'report', 'dashboard', 'metrics', 'power bi', 'powerbi', 'tableau',
                    'headcount', 'head count', 'hr', 'daily', 'weekly', 'monthly',
                    'refresh', 'track', 'monitor', 'department', 'team', 'internal'
                ]
                message_lower = context.activity.text.lower()
                contains_report_intent = any(
                    keyword in message_lower.replace('-', ' ') 
                    for keyword in report_keywords
                )

                print(f"Debug - Message: {message_lower}")
                print(f"Debug - Contains report intent: {contains_report_intent}")
                print(f"Debug - Master agent action: {response.action}")

                # If there's report intent or master agent detected CREATE_REPORT action
                if contains_report_intent or response.action == "CREATE_REPORT":
                    # Start guided creation mode
                    self.conversation_started[conversation_id] = True
                    print("Debug - Starting guided creation mode")
                    
                    # Forward to report creation agent directly with the original message
                    creation_result = await self.report_creation_agent.agent.run(
                        context.activity.text,  # Pass original message instead of timestamped version
                        message_history=[],  # Start fresh conversation
                        deps={}  # Let report creation agent handle everything
                    )
                    
                    # Update message history
                    self.message_histories[conversation_id] = creation_result.all_messages()
                    
                    # Send the message
                    if creation_result.data.message:
                        await context.send_activity(MessageFactory.text(creation_result.data.message))
                    
                    # Show form if needed
                    if creation_result.data.show_card:
                        if creation_result.data.card_data:
                            await context.send_activity(Activity(
                                type="message",
                                attachments=[CardFactory.adaptive_card(creation_result.data.card_data)]
                            ))
                        else:
                            await self.trigger_form_display(context, state, creation_result.data)
                    return True
                
                # Only show welcome card if no report intent detected and it's a first message/greeting
                if (is_greeting or is_first_message) and not context.activity.text.strip() == "/show_form":
                    # Clear state if it's a greeting
                    if is_greeting:
                        self.clear_conversation_state(conversation_id)
                    
                    # Show welcome card
                    welcome_card = create_welcome_card(role="Head of HR")
                    await context.send_activity(Activity(
                        type="message",
                        attachments=[welcome_card]
                    ))
                    return True

                # Handle other actions
                if response.action == "VALIDATE_REQUEST":
                    await self.validate_and_store_request(context, response.data.get("request_data", {}))
                elif response.action == "CHECK_STATUS":
                    await context.send_activity("Status check feature coming soon!")
                else:
                    await context.send_activity(MessageFactory.text(response.message))

            except Exception as e:
                print(f"Error in message handling: {str(e)}")
                traceback.print_exc()
                error_message = f"Sorry, I encountered an error: {str(e)}"
                await context.send_activity(MessageFactory.text(error_message))
            
            return True

        @self.adaptive_cards.action_submit("submit_form")
        async def on_submit_form(context: TurnContext, state: TurnState, data: dict) -> None:
            try:
                conversation_id = context.activity.conversation.id
                print("\n=== Form Submission Started ===")
                
                # First disable the form with submitted data
                if context.activity.reply_to_id:
                    card = {
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "Report Request Form",
                                "wrap": True,
                                "size": "Large",
                                "weight": "Bolder"
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Business Domain: {data.get('business_domain')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Objective: {data.get('objective')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Platform: {data.get('platform')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Frequency: {data.get('frequency')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Data Sources: {data.get('data_sources')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Stakeholders: {data.get('stakeholders')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Required By: {data.get('required_by')}",
                                "wrap": True
                            }
                        ],
                        "actions": [
                            {
                                "type": "Action.Submit",
                                "title": "Submit",
                                "isEnabled": False,
                                "style": "positive"
                            },
                            {
                                "type": "Action.Submit",
                                "title": "❌ Cancel",
                                "isEnabled": False
                            }
                        ]
                    }
                    
                    await context.update_activity(Activity(
                        id=context.activity.reply_to_id,
                        type="message",
                        attachments=[CardFactory.adaptive_card(card)]
                    ))

                # Continue with validation logic
                form_data = {
                    'business_domain': data.get('business_domain'),
                    'business_domain_other': data.get('business_domain_other'),
                    'objective': data.get('objective'),
                    'platform': data.get('platform'),
                    'frequency': data.get('frequency'),
                    'data_sources': data.get('data_sources'),
                    'stakeholders': data.get('stakeholders'),
                    'required_by': data.get('required_by'),
                    'requester_id': context.activity.from_property.id
                }
                
                print(f"Form Data: {json.dumps(form_data, indent=2)}")
                
                # Get the form data and validate
                print("Starting validation...")
                validation_result = await self.validation_agent.validate_request(form_data)
                print(f"Validation Result: {validation_result}")
                
                if validation_result.needs_confirmation:
                    print("Validation requires confirmation")
                    # Store form data temporarily
                    self.pending_submissions[conversation_id] = form_data
                    
                    # Create confirmation card
                    confirmation_card = {
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": validation_result.message,
                                "wrap": True
                            }
                        ],
                        "actions": [
                            {
                                "type": "Action.Submit",
                                "title": action.title if isinstance(action, ActionData) else action,
                                "data": (
                                    action.data if isinstance(action, ActionData)
                                    else {
                                        "verb": (
                                            "confirm_submit_anyway" if action.lower() == "submit anyway"
                                            else "view_existing_reports" if action.lower() == "view existing reports"
                                            else f"confirm_{action.lower().replace(' ', '_')}"
                                        ),
                                        "action_taken": True
                                    }
                                )
                            } for action in validation_result.suggested_actions
                        ]
                    }
                    await context.send_activity(Activity(
                        type="message",
                        attachments=[CardFactory.adaptive_card(confirmation_card)]
                    ))
                    return

                print("No confirmation needed, proceeding with direct storage")
                # Store in database directly
                store_success = await self.validation_agent.store_request(form_data)
                if store_success:
                    await context.send_activity("✅ Report Request Submitted\nYour request has been registered and will be processed. You will receive updates on the progress shortly.")
                else:
                    await context.send_activity("❌ Error submitting request. Please try again.")

            except Exception as e:
                print(f"Error in on_submit_form: {str(e)}")
                traceback.print_exc()
                await context.send_activity("❌ Error processing your form submission. Please try again.")

        @self.adaptive_cards.action_submit("validation_cancel")
        async def on_validation_cancel(context: TurnContext, state: TurnState, data) -> None:
            try:
                conversation_id = context.activity.conversation.id
                
                # Get the original card from the activity
                if context.activity.reply_to_id:
                    original_card = context.activity.value.get("originalCard", {})
                    
                    # Keep the original body (header text)
                    card = {
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "📋 I found similar existing report requests:",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": data.get("message", ""),
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": "Would you like to:",
                                "wrap": True
                            }
                        ],
                        "actions": [
                            {
                                "type": "Action.Submit",
                                "title": "Submit as New Request",
                                "isEnabled": False
                            },
                            {
                                "type": "Action.Submit",
                                "title": "View Existing Reports",
                                "isEnabled": False
                            },
                            {
                                "type": "Action.Submit",
                                "title": "❌ Cancel",
                                "isEnabled": False
                            }
                        ]
                    }
                    
                    await context.update_activity(Activity(
                        id=context.activity.reply_to_id,
                        type="message",
                        attachments=[CardFactory.adaptive_card(card)]
                    ))
                
                # Remove pending submission
                if conversation_id in self.pending_submissions:
                    del self.pending_submissions[conversation_id]
                    
                await context.send_activity(MessageFactory.text(
                    "❌ Cancelled. You can continue with your current request."
                ))
            except Exception as e:
                print(f"Error in validation cancel: {e}")
                traceback.print_exc()

        @self.adaptive_cards.action_submit("cancel_form")
        async def on_form_cancel(context: TurnContext, state: TurnState, data) -> None:
            try:
                conversation_id = context.activity.conversation.id
                
                # Add debug logging
                print("\n=== Form Cancel Debug ===")
                print("Context activity value:", context.activity.value)
                
                # Disable the form card
                if context.activity.reply_to_id:
                    # Reconstruct the form card using the data from activity value
                    card = {
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "Report Request Form",
                                "wrap": True,
                                "size": "Large",
                                "weight": "Bolder"
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Business Domain: {data.get('business_domain')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Objective: {data.get('objective')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Platform: {data.get('platform')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Frequency: {data.get('frequency')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Data Sources: {data.get('data_sources')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Stakeholders: {data.get('stakeholders')}",
                                "wrap": True
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Required By: {data.get('required_by')}",
                                "wrap": True
                            }
                        ],
                        "actions": [
                            {
                                "type": "Action.Submit",
                                "title": "Submit",
                                "isEnabled": False,
                                "style": "positive"
                            },
                            {
                                "type": "Action.Submit",
                                "title": "❌ Cancel",
                                "isEnabled": False
                            }
                        ]
                    }
                    
                    await context.update_activity(Activity(
                        id=context.activity.reply_to_id,
                        type="message",
                        attachments=[CardFactory.adaptive_card(card)]
                    ))
                
                # Clear conversation states but keep welcome card state
                if conversation_id in self.conversation_states:
                    del self.conversation_states[conversation_id]
                if conversation_id in self.message_histories:
                    del self.message_histories[conversation_id]
                if conversation_id in self.conversation_started:
                    del self.conversation_started[conversation_id]
                if conversation_id in self.pending_submissions:
                    del self.pending_submissions[conversation_id]
                
                # Send cancellation message for regular form cancel
                await context.send_activity(MessageFactory.text(
                    "❌ Report request cancelled.\n\nType anything to start a new request."
                ))
            except Exception as e:
                print(f"Error in form cancel: {e}")
                traceback.print_exc()
                await context.send_activity("❌ Error cancelling the form. Please try again.")

        @self.adaptive_cards.action_submit("quick_request")
        async def on_quick_request(context: TurnContext, state: TurnState, data) -> None:
            if data.get("action_taken"):
                await disable_old_cards(context)
            department = data.get("department", "")
            
            # Create a response with the department info
            response = TeamsResponse(
                message=f"Creating a quick report request for {department} department...",
                business_domain=department
            )
            
            await context.send_activity(MessageFactory.text(response.message))
            return await self.trigger_form_display(context, state, response)

        @self.adaptive_cards.action_submit("check_status")
        async def on_check_status(context: TurnContext, state: TurnState, data: dict) -> None:
            """Handle request status check"""
            try:
                # Get user ID from context and ensure it's an integer
                user_id = context.activity.from_property.id
                # Extract numeric part or use default
                try:
                    requester_id = int(''.join(filter(str.isdigit, user_id))) if user_id else 1
                except ValueError:
                    requester_id = 1  # Fallback to default user ID
                
                # Get recent requests
                requests = await self.validation_agent.get_request_status(
                    requester_id=requester_id
                )
                
                if not requests:
                    await context.send_activity("No report requests found. You can create a new request using the Quick Report Request option.")
                    return

                # Create and send status card
                status_card = create_status_card(requests)
                await context.send_activity(Activity(
                    type="message",
                    attachments=[status_card]
                ))
            except Exception as e:
                print(f"Error checking status: {e}")
                traceback.print_exc()
                await context.send_activity("Sorry, I couldn't retrieve the request status at this time. Please try again later.")

        @self.adaptive_cards.action_submit("guided_creation")
        async def on_guided_creation(context: TurnContext, state: TurnState, data) -> None:
            if data.get("action_taken"):
                await disable_old_cards(context)
            
            conversation_id = context.activity.conversation.id
            # Clear any existing state
            self.clear_conversation_state(conversation_id)
            
            # Mark that we're in an active conversation
            self.conversation_started[conversation_id] = True
            
            # Start a new conversation with the report creation agent directly
            result = await self.report_creation_agent.agent.run(
                "Start guided report creation for a new user",
                message_history=[],  # Start fresh for guided creation
                deps={}  # Let the conversation flow naturally
            )
            self.message_histories[conversation_id] = result.all_messages()
            
            # Send the message
            if result.data.message:
                await context.send_activity(MessageFactory.text(result.data.message))

        @self.adaptive_cards.action_submit("confirm_submit_new")
        @self.adaptive_cards.action_submit("confirm_submit")
        @self.adaptive_cards.action_submit("confirm_submit_anyway")
        @self.adaptive_cards.action_submit("confirm_submit_as_new_request")
        async def on_confirm_submit(context: TurnContext, state: TurnState, data) -> None:
            try:
                conversation_id = context.activity.conversation.id
                
                # Disable and highlight button
                if context.activity.reply_to_id:
                    original_card = context.activity.value.get("originalCard", {})
                    validation_message = ""
                    
                    # Extract the full validation message including similarity details
                    for body_item in original_card.get("body", []):
                        if body_item.get("type") == "TextBlock":
                            current_text = body_item.get("text", "")
                            if "The requests are essentially the same" in current_text:
                                validation_message = current_text
                                break

                    # Create disabled card with the original message and highlight Submit button
                    card = create_disabled_validation_card(
                        message=validation_message,
                        selected_action="Submit as New Request"  # This will highlight the Submit button
                    )
                    
                    print("Debug - Card actions:", json.dumps(card.get("actions", []), indent=2))
                    
                    await context.update_activity(Activity(
                        id=context.activity.reply_to_id,
                        type="message",
                        attachments=[CardFactory.adaptive_card(card)]
                    ))
                
                # Continue with submission logic
                form_data = self.pending_submissions.get(conversation_id)
                if form_data:
                    # Store in database
                    store_success = await self.validation_agent.store_request(form_data)
                    if store_success:
                        await context.send_activity("✅ Report Request Submitted\nYour request has been registered and will be processed. You will receive updates on the progress shortly.")
                        # Clear pending submission
                        del self.pending_submissions[conversation_id]
                    else:
                        await context.send_activity("❌ Error submitting request. Please try again.")
                else:
                    await context.send_activity("❌ No pending request found. Please try submitting the form again.")

            except Exception as e:
                print(f"Error in confirm submit: {e}")
                traceback.print_exc()
                await context.send_activity("❌ Error processing your confirmation. Please try again.")

        @self.adaptive_cards.action_submit("view_existing_reports")
        async def on_view_existing(context: TurnContext, state: TurnState, data: dict) -> None:
            try:
                # Disable the card buttons
                card_id = data.get("card_id")
                if card_id:
                    await self._disable_card(context, card_id, context.activity.text)
                
                conversation_id = context.activity.conversation.id
                form_data = self.pending_submissions.get(conversation_id)
                
                if form_data:
                    # Get similar reports with details
                    similar_reports = await self.validation_agent._find_similar_requests(
                        form_data.get('objective', ''),
                        form_data.get('business_domain', '')
                    )
                    
                    # Create a card to show existing reports
                    card_id = f"similar-reports-{conversation_id}"
                    card_data = {
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "id": card_id,
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "📋 Similar Existing Reports",
                                "size": "Large",
                                "weight": "Bolder"
                            }
                        ]
                    }
                    
                    for report in similar_reports:
                        card_data["body"].append({
                            "type": "Container",
                            "style": "emphasis",
                            "spacing": "medium",
                            "items": [
                                {
                                    "type": "TextBlock",
                                    "text": f"Objective: {report.objective}",
                                    "wrap": True,
                                    "weight": "Bolder"
                                },
                                {
                                    "type": "TextBlock",
                                    "text": f"Domain: {report.business_domain}",
                                    "isSubtle": True
                                },
                                {
                                    "type": "TextBlock",
                                    "text": f"Similarity Score: {report.similarity_score:.2%}",
                                    "isSubtle": True
                                }
                            ]
                        })
                    
                    # Add actions
                    card_data["actions"] = [
                        {
                            "type": "Action.Submit",
                            "title": "Submit as New Request",
                            "data": {
                                "verb": "confirm_submit_as_new_request",
                                "action_taken": True
                            }
                        },
                        {
                            "type": "Action.Submit",
                            "title": "Cancel",
                            "data": {
                                "verb": "cancel_form",
                                "action_taken": True
                            }
                        }
                    ]
                    
                    await context.send_activity(Activity(
                        type="message",
                        attachments=[CardFactory.adaptive_card(card_data)]
                    ))
                else:
                    await context.send_activity("❌ No pending request found. Please try submitting the form again.")
                    
            except Exception as e:
                print(f"Error viewing existing reports: {e}")
                traceback.print_exc()
                await context.send_activity("❌ Error retrieving similar reports. Please try again.")

        @self.adaptive_cards.action_submit("cross_department_cancel")  # Specific handler for cross-department cancel
        async def on_cross_department_cancel(context: TurnContext, state: TurnState, data) -> None:
            try:
                conversation_id = context.activity.conversation.id
                
                # Disable the current card
                if context.activity.reply_to_id:
                    card = {
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "⚠️ Cross-Department Request Warning",
                                "wrap": True,
                                "size": "Large",
                                "weight": "Bolder"
                            },
                            {
                                "type": "TextBlock",
                                "text": "Request cancelled due to cross-department access requirements.",
                                "wrap": True
                            }
                        ],
                        "actions": [
                            {
                                "type": "Action.Submit",
                                "title": "Submit anyway",
                                "data": {"verb": "cross_department_submit"},
                                "isEnabled": False
                            },
                            {
                                "type": "Action.Submit",
                                "title": "❌ Cancel",
                                "data": {"verb": "cross_department_cancel"},
                                "isEnabled": False
                            }
                        ]
                    }
                    
                    await context.update_activity(Activity(
                        id=context.activity.reply_to_id,
                        type="message",
                        attachments=[CardFactory.adaptive_card(card)]
                    ))
                
                # Clear any pending submissions
                if conversation_id in self.pending_submissions:
                    del self.pending_submissions[conversation_id]
                
                # Send cancellation message
                await context.send_activity(MessageFactory.text(
                    "❌ Request cancelled. Please create a request for your own department or coordinate with the appropriate department."
                ))
                
            except Exception as e:
                print(f"Error in cross-department cancel: {e}")
                traceback.print_exc()
                await context.send_activity("Error processing your cancellation. Please try again.")

        @self.adaptive_cards.action_submit("cross_department_submit")
        async def on_cross_department_submit(context: TurnContext, state: TurnState, data) -> None:
            try:
                conversation_id = context.activity.conversation.id
                
                # Disable the current card
                if context.activity.reply_to_id:
                    card = {
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "⚠️ Policy Override Acknowledged",
                                "wrap": True,
                                "size": "Large",
                                "weight": "Bolder",
                                "color": "Warning"
                            },
                            {
                                "type": "TextBlock",
                                "text": "You have chosen to proceed with a cross-department request despite policy guidelines.",
                                "wrap": True
                            }
                        ],
                        "actions": [
                            {
                                "type": "Action.Submit",
                                "title": "Submit anyway",
                                "isEnabled": False,
                                "style": "destructive"
                            },
                            {
                                "type": "Action.Submit",
                                "title": "❌ Cancel",
                                "isEnabled": False
                            }
                        ]
                    }
                    
                    await context.update_activity(Activity(
                        id=context.activity.reply_to_id,
                        type="message",
                        attachments=[CardFactory.adaptive_card(card)]
                    ))
                
                # Get the pending submission
                form_data = self.pending_submissions.get(conversation_id)
                if form_data:
                    # Add policy override flag
                    form_data['policy_override'] = True
                    form_data['override_type'] = 'cross_department'
                    
                    # Store in database
                    store_success = await self.validation_agent.store_request(form_data)
                    if store_success:
                        await context.send_activity(
                            "⚠️ Cross-Department Report Request Submitted\n\n"
                            "Your request has been registered with a policy override flag. "
                            "Additional approvals will be required and the department owner will be notified."
                        )
                        # Clear pending submission
                        del self.pending_submissions[conversation_id]
                    else:
                        await context.send_activity("❌ Error submitting request. Please try again.")
                else:
                    await context.send_activity("❌ No pending request found. Please try submitting the form again.")
                
            except Exception as e:
                print(f"Error in cross-department submit: {e}")
                traceback.print_exc()
                await context.send_activity("Error processing your submission. Please try again.")

    def _setup_tools(self):
        @self.master_agent.tool
        async def show_report_form(ctx: RunContext) -> dict:
            """Shows the report form adaptive card to collect report details."""
            from botbuilder.schema import Attachment  # Add import here
            
            report = ReportRequest.create_new()
            card = create_report_form_card(report)
            
            # Return the card content directly
            if isinstance(card, Attachment):
                return card.content
            return card

        @self.master_agent.tool
        async def show_adaptive_card(
            ctx: RunContext,
            title: str,
            description: str,
            buttons: list[str] = []
        ) -> dict:
            """Shows an adaptive card in Teams with the given title, description and optional buttons."""
            return {
                "type": "AdaptiveCard",
                "version": "1.0",
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "body": [
                    {
                        "type": "TextBlock",
                        "text": title,
                        "weight": "bolder",
                        "size": "medium"
                    },
                    {
                        "type": "TextBlock",
                        "text": description,
                        "wrap": True
                    }
                ],
                "actions": [
                    {
                        "type": "Action.Submit",
                        "title": button,
                        "data": {"button": button}
                    } for button in buttons
                ]
            }

    async def on_members_added_activity(
        self, members_added: list, turn_context: TurnContext
    ) -> None:
        """Send welcome message when bot is added."""
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                result = await self.master_agent.run(
                    "Send a welcome message explaining that I can help create reports. "
                    "Mention that users can ask about creating a report when ready."
                )
                response = result.data
                
                if response.show_card and response.card_data:
                    card = CardFactory.adaptive_card(response.card_data)
                    await turn_context.send_activity(MessageFactory.attachment(card))
                
                if response.message:
                    await turn_context.send_activity(MessageFactory.text(response.message))

    async def trigger_form_display(self, context: TurnContext, state: TurnState, response_data: TeamsResponse = None) -> bool:
        try:
            conversation_id = context.activity.conversation.id
            
            if conversation_id not in self.conversation_states:
                self.conversation_states[conversation_id] = ReportRequest.create_new()
            
            report = self.conversation_states[conversation_id]
            
            if response_data:
                # Update fields including processed date
                if response_data.business_domain:
                    report.business_domain = response_data.business_domain
                if response_data.objective:
                    report.objective = response_data.objective
                if response_data.platform:
                    report.platform = response_data.platform
                if response_data.frequency:
                    report.frequency = response_data.frequency
                if response_data.stakeholders:
                    report.stakeholders = response_data.stakeholders
                if response_data.data_sources:
                    report.data_sources = response_data.data_sources
                if response_data.required_by:
                    # The date should already be in YYYY-MM-DD format from the LLM processing
                    report.required_by = response_data.required_by
                    
                print(f"Updated report with collected data: {report.__dict__}")
            
            card = create_report_form_card(report)
            await context.send_activity(Activity(
                type="message",
                attachments=[card]
            ))
            
            return True
            
        except Exception as e:
            print(f"Error showing form: {str(e)}")
            traceback.print_exc()
            await context.send_activity("There was an error displaying the form. Please try again.")
            return False

    async def on_turn(self, turn_context: TurnContext):
        """Handle bot framework turn"""
        # Only show welcome card for non-message activities (like when bot is added)
        if turn_context.activity.type != "message":
            conversation_id = turn_context.activity.conversation.id
            if conversation_id not in self.has_sent_welcome:
                self.has_sent_welcome[conversation_id] = True
                welcome_card = create_welcome_card(role="Head of HR")
                await turn_context.send_activity(Activity(
                    type="message",
                    attachments=[welcome_card]
                ))
        
        # Continue with normal turn processing
        await super().on_turn(turn_context)

    async def validate_and_store_request(self, context: TurnContext, request_data: dict):
        """Validate and store a report request"""
        try:
            # Get user info
            user_id = context.activity.from_property.id
            
            # Run validation
            result = await self.validation_agent.agent.run(
                "Validate and store new report request",
                deps={"request": request_data, "user_id": user_id}
            )
            
            if not result.data.valid:
                await context.send_activity(
                    f"⚠️ {result.data.message}\n\n"
                    "Would you like to modify your request?"
                )
                return False
                
            if result.data.similar_requests:
                similar = "\n".join(
                    f"- {r.objective} ({r.business_domain})"
                    for r in result.data.similar_requests[:3]
                )
                await context.send_activity(
                    f"📝 Found similar existing requests:\n\n{similar}\n\n" +
                    "Would you like to proceed with your request anyway?"
                )
                return False
                
            if not result.data.department_access:
                await context.send_activity(
                    "⚠️ You're creating a request for a different department. " +
                    "Would you like to proceed or modify the request?"
                )
                return False
                
            if result.data.store_success:
                await context.send_activity("✅ Request successfully stored!")
                return True
                
            await context.send_activity("❌ Failed to store request. Please try again.")
            return False
            
        except Exception as e:
            print(f"Error in validate_and_store: {e}")
            await context.send_activity("An error occurred during validation. Please try again.")
            return False

    async def _disable_card(self, context: TurnContext, card_id: str, selected_action: str) -> None:
        """Helper to disable card buttons after selection"""
        try:
            if context.activity.reply_to_id:
                # Get the original card
                original_card = context.activity.value.get("originalCard", {})
                
                # Disable all buttons except the selected one
                for action in original_card.get("actions", []):
                    action["isEnabled"] = False  # Disable all buttons
                
                # Update the card
                await context.update_activity(Activity(
                    id=context.activity.reply_to_id,
                    type="message",
                    attachments=[CardFactory.adaptive_card(original_card)]
                ))
        except Exception as e:
            print(f"Error disabling card: {e}")

    async def disable_confirmation_card(self, context: TurnContext, original_message: str) -> None:
        """Helper to disable the confirmation card"""
        try:
            if context.activity.reply_to_id:
                # Create disabled version of the confirmation card
                disabled_card = {
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": original_message,
                            "wrap": True
                        }
                    ],
                    "actions": [
                        {
                            "type": "Action.Submit",
                            "title": "✅ Cancelled",
                            "isEnabled": False,
                            "style": "positive"
                        }
                    ]
                }
                
                # Update the previous activity with disabled card
                await context.update_activity(Activity(
                    id=context.activity.reply_to_id,
                    type="message",
                    attachments=[CardFactory.adaptive_card(disabled_card)]
                ))
                print("Successfully disabled confirmation card")
                
        except Exception as e:
            print(f"Error disabling confirmation card: {str(e)}")
            traceback.print_exc()

# Create the bot application instance
bot_app = TeamsAIApplication()
