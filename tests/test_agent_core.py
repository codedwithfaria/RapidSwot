import asyncio
from typing import Any, Dict

import pytest

from src.agent.core.agent import ActionPlan, ExecutionEngine, IntentProcessor


class DummyLlm:
    async def generate_response(self, *_: Any, **__: Any) -> str:
        raise AssertionError("generate_response should not be called in these tests")


def test_extract_json_from_fenced_block():
    processor = IntentProcessor(DummyLlm())
    response = """
    Sure, here is the plan:
    ```json
    {"steps": ["a"], "resources": {}}
    ```
    """

    extracted = processor._extract_json_block(response)

    assert extracted == '{"steps": ["a"], "resources": {}}'


def test_parse_llm_response_returns_default_for_empty():
    processor = IntentProcessor(DummyLlm())

    plan = processor._parse_llm_response("")

    assert plan == {"steps": [], "resources": {}, "duration": 0.0, "metrics": {}}


def test_parse_llm_response_normalizes_plan():
    processor = IntentProcessor(DummyLlm())
    response = """
    Plan details:
    ```json
    {
        "Steps": [
            {"tool": "search", "params": {"query": "test"}},
            "Fallback description"
        ],
        "Resources": {"gpu": true},
        "duration": 2.5,
        "metrics": {"success": "found"}
    }
    ```
    """

    plan = processor._parse_llm_response(response)

    assert plan["steps"][0]["tool"] == "search"
    assert plan["steps"][1] == {"description": "Fallback description", "tool": "", "params": {}}
    assert plan["resources"] == {"gpu": True}
    assert plan["duration"] == 2.5
    assert plan["metrics"] == {"success": "found"}


def test_execution_engine_handles_sync_and_async_tools():
    engine = ExecutionEngine()

    def sync_tool(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": params["value"] * context["multiplier"]}

    async def async_tool(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"text": params["text"].upper()}

    engine.register_tool("sync", sync_tool)
    engine.register_tool("async", async_tool)

    plan = ActionPlan(
        steps=[
            {"tool": "sync", "params": {"value": 2}},
            {"tool": "async", "params": {"text": "done"}},
        ],
        resources={},
        estimated_duration=0.0,
        success_metrics={},
    )

    context = {"multiplier": 3}

    events = []

    async def _collect_events():
        async for event in engine.execute_plan(plan, context):
            events.append(event)

    asyncio.run(_collect_events())

    assert len(events) == 2
    assert events[0].content["result"] == {"value": 6}
    assert events[1].content["result"] == {"text": "DONE"}


def test_execution_engine_emits_error_event_on_failure():
    engine = ExecutionEngine()

    def failing_tool(_: Dict[str, Any], __: Dict[str, Any]) -> None:
        raise RuntimeError("boom")

    engine.register_tool("fail", failing_tool)

    plan = ActionPlan(
        steps=[{"tool": "fail", "params": {}}],
        resources={},
        estimated_duration=0.0,
        success_metrics={},
    )

    events = []

    async def _collect_events():
        async for event in engine.execute_plan(plan, {}):
            events.append(event)

    asyncio.run(_collect_events())

    assert len(events) == 1
    assert "boom" in events[0].content["error"]
    assert events[0].actions.escalate is True
