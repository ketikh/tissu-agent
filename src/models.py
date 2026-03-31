from pydantic import BaseModel, Field
from typing import Optional


class CustomerContext(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    previous_messages: list[str] = Field(default_factory=list)
    product_interest: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    channel: Optional[str] = "web"
    customer_context: Optional[CustomerContext] = None
    metadata: dict = Field(default_factory=dict)


class AgentMetadata(BaseModel):
    intent: Optional[str] = None
    customer_stage: Optional[str] = None
    hot_lead: Optional[bool] = None
    next_action: Optional[str] = None
    notes: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    agent_type: str
    tool_calls_made: list[str] = Field(default_factory=list)
    agent_metadata: Optional[AgentMetadata] = None


class LeadCreate(BaseModel):
    name: str
    email: Optional[str] = None
    company: Optional[str] = None
    phone: Optional[str] = None
    source: str = "agent"
    notes: Optional[str] = None


class TicketCreate(BaseModel):
    subject: str
    description: Optional[str] = None
    priority: str = "medium"
    customer_email: Optional[str] = None


class ContentCreate(BaseModel):
    title: str
    body: str
    content_type: str
    tags: list[str] = Field(default_factory=list)
    scheduled_at: Optional[str] = None
