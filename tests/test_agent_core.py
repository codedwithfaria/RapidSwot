import asyncio
from typing import Any, Dict

import pytest

from src.agent.core.agent import ActionPlan, ExecutionEngine, IntentProcessor, PlanParser
from src.sequential.planner import Step


class DummyLlm:
    async def generate_response(self, *_: Any, **__: Any) -> str:
        raise AssertionError("generate_response should not be called in these tests")


def test_extract_json_from_fenced_block():
    parser = PlanParser()
    response = """
    Sure, here is the plan:
    ```json
    {"steps": ["a"], "resources": {}}
    ```
    """

    extracted = parser._extract_json_block(response)

    assert extracted == '{"steps": ["a"], "resources": {}}'


def test_parse_llm_response_returns_default_for_empty():
    parser = PlanParser()

    plan = parser.parse("")

    assert plan == {"steps": [], "resources": {}, "duration": 0.0, "metrics": {}}


def test_parse_llm_response_normalizes_plan():
    parser = PlanParser()
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

    plan = parser.parse(response)

    assert plan["steps"][0]["tool"] == "search"
    assert plan["steps"][1] == {"description": "Fallback description", "tool": "", "params": {}}
    assert plan["resources"] == {"gpu": True}
    assert plan["duration"] == 2.5
    assert plan["metrics"] == {"success": "found"}


def test_parse_llm_response_handles_missing_steps():
    parser = PlanParser()
    response = """
    ```json
    {"Resources": {"cpu": 2}}
    ```
    """

    plan = parser.parse(response)

    assert plan == {"steps": [], "resources": {"cpu": 2}, "duration": 0.0, "metrics": {}}


def test_plan_parser_extracts_structured_text_plan():
    parser = PlanParser()
    response = """
    Plan overview:
    1. Investigate logs for recent failures (tool=search, params={"query": "error"})
    2. Summarize findings for stakeholders (tool=report)
    Resources: laptop, vpn access
    Estimated Duration: 45 minutes
    Success Metrics: {"accuracy": 0.9, "latency_reduction": "<5%"}
    """

    plan = parser.parse(response)

    assert plan["steps"][0]["description"].startswith("Investigate logs")
    assert plan["steps"][0]["tool"] == "search"
    assert plan["steps"][0]["params"] == {"query": "error"}
    assert plan["steps"][1]["tool"] == "report"
    assert plan["resources"]["items"] == ["laptop", "vpn access"]
    assert plan["duration"] == 45.0
    assert plan["metrics"] == {"accuracy": 0.9, "latency_reduction": "<5%"}


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


def test_execution_engine_validates_step_structure():
    engine = ExecutionEngine()
    engine.register_tool("noop", lambda params, ctx: None)

    plan = ActionPlan(
        steps=["invalid-step"],
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
    assert events[0].actions.escalate is True
    assert "mapping" in events[0].content["error"]


def test_execution_engine_accepts_action_alias():
    engine = ExecutionEngine()
    engine.register_tool("noop", lambda params, ctx: params["value"])

    plan = ActionPlan(
        steps=[{"action": "noop", "params": {"value": 7}}],
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
    assert events[0].content["result"] == 7
    assert events[0].content["step"]["tool"] == "noop"


def test_execution_engine_handles_step_objects():
    engine = ExecutionEngine()
    engine.register_tool("navigate", lambda params, ctx: params["url"])

    step = Step(action="navigate", params={"url": "https://example.com"}, description="go")

    plan = ActionPlan(
        steps=[step],
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
    assert events[0].content["result"] == "https://example.com"
    assert events[0].content["step"]["action"] == "navigate"
