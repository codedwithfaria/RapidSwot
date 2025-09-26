from fastapi import APIRouter, HTTPException, Depends
from sse_starlette.sse import EventSourceResponse
from app.application.services.chat_service import ChatService
from app.application.schemas.chat import ChatRequest, ChatResponse
from typing import AsyncGenerator

router = APIRouter()
chat_service = ChatService()

@router.post("/chat")
async def create_chat(request: ChatRequest):
    try:
        session_id = await chat_service.create_session(request.user_id)
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/{session_id}/stream")
async def stream_chat(session_id: str, message: str):
    try:
        return EventSourceResponse(chat_service.stream_response(session_id, message))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tool/browser")
async def browser_action(action: dict):
    try:
        result = await chat_service.execute_browser_action(action)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tool/shell")
async def shell_command(command: dict):
    try:
        result = await chat_service.execute_shell_command(command)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))