"""
AG-UI protocol implementation for agent-human interaction.
"""
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import json
import asyncio
from datetime import datetime

class EventType(Enum):
    CHAT_START = "chat_start"
    CHAT_MESSAGE = "chat_message"
    CHAT_STREAM = "chat_stream"
    CHAT_END = "chat_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    UI_UPDATE = "ui_update"
    STATE_SYNC = "state_sync"
    CONTEXT_UPDATE = "context_update"
    ERROR = "error"
    HUMAN_INPUT = "human_input"
    AGENT_THINKING = "agent_thinking"
    AGENT_ACTION = "agent_action"
    TASK_COMPLETE = "task_complete"

@dataclass
class AGUIEvent:
    type: EventType
    data: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

class AGUIProtocol:
    def __init__(self):
        self.subscribers = []
        self.context = {}
        
    async def emit(self, event: AGUIEvent):
        """Emit an event to all subscribers."""
        event_data = {
            "type": event.type.value,
            "data": event.data,
            "timestamp": event.timestamp
        }
        
        for subscriber in self.subscribers:
            await subscriber(event_data)
            
    def subscribe(self, callback):
        """Add a subscriber for events."""
        self.subscribers.append(callback)
        return lambda: self.subscribers.remove(callback)
        
    async def update_context(self, context_data: Dict[str, Any]):
        """Update the shared context."""
        self.context.update(context_data)
        await self.emit(AGUIEvent(
            type=EventType.CONTEXT_UPDATE,
            data={"context": self.context}
        ))
        
    async def send_message(self, content: str, role: str = "agent"):
        """Send a chat message."""
        await self.emit(AGUIEvent(
            type=EventType.CHAT_MESSAGE,
            data={
                "content": content,
                "role": role
            }
        ))
        
    async def start_stream(self):
        """Start a streaming response."""
        await self.emit(AGUIEvent(
            type=EventType.CHAT_STREAM,
            data={"status": "start"}
        ))
        
    async def stream_token(self, token: str):
        """Stream a token to the frontend."""
        await self.emit(AGUIEvent(
            type=EventType.CHAT_STREAM,
            data={"token": token}
        ))
        
    async def end_stream(self):
        """End a streaming response."""
        await self.emit(AGUIEvent(
            type=EventType.CHAT_STREAM,
            data={"status": "end"}
        ))
        
    async def update_ui(self, components: List[Dict[str, Any]]):
        """Update UI components."""
        await self.emit(AGUIEvent(
            type=EventType.UI_UPDATE,
            data={"components": components}
        ))
        
    async def report_error(self, error: str, details: Optional[Dict] = None):
        """Report an error."""
        await self.emit(AGUIEvent(
            type=EventType.ERROR,
            data={
                "message": error,
                "details": details or {}
            }
        ))
        
    async def start_thinking(self, thought: str):
        """Indicate agent is thinking."""
        await self.emit(AGUIEvent(
            type=EventType.AGENT_THINKING,
            data={"thought": thought}
        ))
        
    async def complete_task(self, result: Dict[str, Any]):
        """Mark a task as complete."""
        await self.emit(AGUIEvent(
            type=EventType.TASK_COMPLETE,
            data=result
        ))