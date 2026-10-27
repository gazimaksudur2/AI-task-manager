"""Offline provider-adapter and ID-provenance regression tests (no real API calls)."""
from types import SimpleNamespace

import pytest

import task_manager.agent as agent


class FakeStructured:
    def __init__(self, plan):
        self.plan = plan
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.plan


class FakeChatModel:
    def __init__(self, plan):
        self.structured = FakeStructured(plan)
        self.method = None

    def with_structured_output(self, _schema, method):
        self.method = method
        return self.structured


@pytest.mark.parametrize("provider,method", [
    ("openai", "function_calling"),
    ("groq", "json_mode"),
    ("google", "json_schema"),
    ("anthropic", "json_schema"),
])
def test_provider_initializes_structured_planner_offline(
    monkeypatch, provider, method,
):
    chat = FakeChatModel({
        "actions": [{"operation": "CREATE", "title": "Test"}],
    })
    monkeypatch.setattr(agent, "build_chat_model", lambda: chat)
    monkeypatch.setattr(
        agent, "get_settings",
        lambda: SimpleNamespace(llm_provider=provider, app_timezone="Asia/Dhaka"),
    )
    parser = agent.LLMPlanner()
    assert chat.method == method
    plan = parser("Create a task called Test")
    assert plan.actions[0].title == "Test"


def test_planner_rejects_invented_existing_id(monkeypatch):
    chat = FakeChatModel({
        "actions": [{"operation": "DELETE", "target_id": "TSK-000001"}],
    })
    monkeypatch.setattr(agent, "build_chat_model", lambda: chat)
    monkeypatch.setattr(
        agent, "get_settings",
        lambda: SimpleNamespace(llm_provider="openai", app_timezone="Asia/Dhaka"),
    )
    parser = agent.LLMPlanner()
    with pytest.raises(ValueError, match="not supplied"):
        parser("Delete the demonstration task")
    assert parser("Delete TSK-000001").actions[0].target_id == "TSK-000001"
