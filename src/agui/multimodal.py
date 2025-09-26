"""
Multi-modal processing capabilities for autonomous agents.
"""
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from . import AGUIProtocol, AGUIEvent, EventType
from .autonomous import AutonomousAgent, TaskContext, Task, TaskStatus

class ContentType(Enum):
    """Supported content types for multi-modal processing."""
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    AUDIO = "audio"
    VIDEO = "video"
    DATA = "data"

@dataclass
class Content:
    """Container for multi-modal content."""
    type: ContentType
    data: Any
    metadata: Dict[str, Any]

class ProcessingPipeline:
    """Pipeline for processing multi-modal content."""
    
    def __init__(self):
        self.steps: List[callable] = []
        
    def add_step(self, processor: callable):
        """Add a processing step to the pipeline."""
        self.steps.append(processor)
        
    async def process(self, content: Content) -> Content:
        """Process content through the pipeline."""
        result = content
        for step in self.steps:
            result = await step(result)
        return result

class MultiModalProcessor:
    """Processor for handling different types of content."""
    
    def __init__(self, protocol: AGUIProtocol):
        self.protocol = protocol
        self.pipelines: Dict[ContentType, ProcessingPipeline] = {}
        self._setup_pipelines()
    
    def _setup_pipelines(self):
        """Initialize processing pipelines for each content type."""
        for content_type in ContentType:
            self.pipelines[content_type] = ProcessingPipeline()
    
    async def process_content(self, content: Content) -> Content:
        """Process content using appropriate pipeline."""
        pipeline = self.pipelines.get(content.type)
        if not pipeline:
            raise ValueError(f"No pipeline available for content type: {content.type}")
        
        await self.protocol.emit(AGUIEvent(
            type=EventType.TOOL_START,
            data={
                "content_type": content.type.value,
                "action": "processing_start",
                "metadata": content.metadata
            }
        ))
        
        try:
            result = await pipeline.process(content)
            
            await self.protocol.emit(AGUIEvent(
                type=EventType.TOOL_END,
                data={
                    "content_type": content.type.value,
                    "action": "processing_complete",
                    "metadata": result.metadata
                }
            ))
            
            return result
            
        except Exception as e:
            await self.protocol.emit(AGUIEvent(
                type=EventType.ERROR,
                data={
                    "content_type": content.type.value,
                    "error": str(e),
                    "metadata": content.metadata
                }
            ))
            raise

class MultiModalAgent(AutonomousAgent):
    """Agent capable of processing multiple types of content."""
    
    def __init__(self, protocol: AGUIProtocol):
        super().__init__(protocol)
        self.processor = MultiModalProcessor(protocol)
        
    async def process(self, content: Content) -> str:
        """Process multi-modal content."""
        task_id = await self.submit_task(
            name=f"process_{content.type.value}",
            description=f"Processing {content.type.value} content",
            context={
                "content_type": content.type.value,
                "metadata": content.metadata
            }
        )
        
        # Store content in task context
        task_context = TaskContext(
            task_id=task_id,
            session_id=content.metadata.get("session_id", "default"),
            user_id=content.metadata.get("user_id", "default"),
            metadata=content.metadata
        )
        task_context.add_artifact("content", content)
        
        # Process content
        try:
            result = await self.processor.process_content(content)
            task = self.active_tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.utcnow().isoformat()
            
        except Exception as e:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow().isoformat()
            raise
        
        return task_id

class ContentProcessor:
    """Base class for content type-specific processors."""
    
    async def process(self, content: Content) -> Content:
        """Process content. Override in subclasses."""
        raise NotImplementedError()

class TextProcessor(ContentProcessor):
    """Processor for text content."""
    
    async def process(self, content: Content) -> Content:
        if content.type != ContentType.TEXT:
            raise ValueError("Content type must be TEXT")
        
        # Add text processing logic here
        processed_text = content.data.upper()  # Example transformation
        
        return Content(
            type=ContentType.TEXT,
            data=processed_text,
            metadata={
                **content.metadata,
                "processed": True,
                "processor": "TextProcessor"
            }
        )

class CodeProcessor(ContentProcessor):
    """Processor for code content."""
    
    async def process(self, content: Content) -> Content:
        if content.type != ContentType.CODE:
            raise ValueError("Content type must be CODE")
        
        # Add code processing logic here
        # For example: parsing, formatting, analysis
        
        return Content(
            type=ContentType.CODE,
            data=content.data,
            metadata={
                **content.metadata,
                "processed": True,
                "processor": "CodeProcessor"
            }
        )

class ImageProcessor(ContentProcessor):
    """Processor for image content."""
    
    async def process(self, content: Content) -> Content:
        if content.type != ContentType.IMAGE:
            raise ValueError("Content type must be IMAGE")
        
        # Add image processing logic here
        # For example: resize, analyze, transform
        
        return Content(
            type=ContentType.IMAGE,
            data=content.data,
            metadata={
                **content.metadata,
                "processed": True,
                "processor": "ImageProcessor"
            }
        )