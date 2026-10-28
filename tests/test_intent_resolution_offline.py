"""Regression checks for clarification controls; no LLM or Chroma needed."""

from dataclasses import dataclass

import pytest

from task_manager.intent_resolution import (
    is_cancel_request,
    is_new_instruction,
    is_simple_list_request,
    is_unqualified_delete_request,
    partial_title_matches,
    title_search_query,
)


@dataclass
class StubTask:
    title: str


@pytest.mark.parametrize("text", [
    "show all tasks", "show all my tasks", "find all", "find all tasks",
    "find the tasks", "list tasks", "show my tasks", "show all",
    "Show all tasks!", "please show all tasks",
])
def test_simple_list_bypasses_stale_clarification(text):
    assert is_simple_list_request(text)
    assert is_new_instruction(text)


@pytest.mark.parametrize("text", [
    "Show high priority tasks", "Find documentation tasks", "search demo",
    "change task status", "create a new task", "I have a meeting tomorrow",
])
def test_filtered_or_mutating_commands_are_not_unfiltered_list(text):
    assert not is_simple_list_request(text)
    assert is_new_instruction(text)


@pytest.mark.parametrize("text", [
    "delete all tasks", "delete tasks", "remove my tasks",
    "could you delete every task?",
])
def test_unqualified_delete_requests_need_a_task_choice(text):
    assert is_unqualified_delete_request(text)


@pytest.mark.parametrize("text", [
    "delete the demo task", "remove task TSK-000001",
])
def test_targeted_deletes_are_not_unqualified(text):
    assert not is_unqualified_delete_request(text)


@pytest.mark.parametrize("text", [
    "find task that have hangout on its title",
    "find tasks that have hangout in their title",
    "show tasks with hangout in the title",
])
def test_title_search_extracts_literal_query(text):
    assert title_search_query(text) == "hangout"


@pytest.mark.parametrize("text", [
    "1", "option 2", "the first one", "TSK-000001", "2026-11-20",
    "Prepare my demo", "tomorrow", "next Friday",
])
def test_clarification_answers_do_not_look_like_new_commands(text):
    assert not is_new_instruction(text)


@pytest.mark.parametrize("text", ["cancel", "Cancel request", "never mind", "stop", "start over"])
def test_can_explicitly_cancel_a_pending_request(text):
    assert is_cancel_request(text)


def test_unique_title_fragments_are_deterministic():
    demo = StubTask("Prepare my demo")
    meeting = StubTask("Meeting with SpaceX CEO")
    tasks = [demo, meeting]
    for reference in ["demo", "demo task", "prepare my demo task", "the task that has demo in its title"]:
        assert partial_title_matches(reference, tasks) == [demo]
    assert partial_title_matches("weekly report", tasks) == []
    assert partial_title_matches("that task", tasks) == []


def test_duplicate_title_fragment_is_ambiguous_not_auto_selected():
    a, b = StubTask("Prepare my demo"), StubTask("Review client demo")
    assert partial_title_matches("demo task", [a, b]) == [a, b]
