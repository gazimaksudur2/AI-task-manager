"""Deterministic agent-to-SQLite/Chroma integration tests; no real LLM requests.

Run after ``pip install -r requirements.txt`` on Python 3.12.
"""
from datetime import timedelta

import pytest
from pydantic import ValidationError

from task_manager.agent_models import ActionPlan
from task_manager.database import TaskDatabase
from task_manager.graph import build_agent_graph, dhaka_today
from task_manager.task_service import TaskService
from task_manager.vector_store import TaskVectorStore
from tests.test_agent import FakePlanner
from tests.test_data_layer import TinyTestEmbeddings


@pytest.fixture
def service(tmp_path):
    db = TaskDatabase(tmp_path / "data" / "tasks.sqlite3")
    index = TaskVectorStore(tmp_path / "data" / "chroma", TinyTestEmbeddings())
    return TaskService(db, index)


def start(service, actions, request="test request", thread="scenario"):
    planner = FakePlanner({"actions": actions})
    graph = build_agent_graph(service, planner=planner)
    result = graph.invoke(
        {"user_request": request, "submission_id": thread + "-message-1"},
        config={"configurable": {"thread_id": thread}},
    )
    return graph, result


def answer(graph, text, thread, submission_id):
    return graph.invoke(
        {"user_request": text, "submission_id": submission_id},
        config={"configurable": {"thread_id": thread}},
    )


def test_01_02_03_create_tomorrow_survives_restart_and_semantic_search(service, tmp_path):
    graph, result = start(service, [{
        "operation": "CREATE", "title": "Prepare project demo presentation",
        "priority": "High", "due_date": "tomorrow",
    }], request="Create a high priority task to prepare the demo tomorrow")
    assert result["response"].startswith("Created TSK-")
    task = service.all_tasks()[0]
    assert task.priority.value == "High"
    assert task.due_date == dhaka_today() + timedelta(days=1)

    reopened = TaskService(
        TaskDatabase(tmp_path / "data" / "tasks.sqlite3"),
        TaskVectorStore(tmp_path / "data" / "chroma", TinyTestEmbeddings()),
    )
    assert reopened.get_task(task.id).title == task.title
    assert reopened.search_tasks("demonstration presentation")[0].task.id == task.id


def test_04_list_unfinished_documentation(service):
    service.create_task(title="Prepare documentation", status="Todo")
    service.create_task(title="Review documentation", status="In progress")
    service.create_task(title="Archive documentation", status="Done")
    _, result = start(service, [{
        "operation": "SEARCH", "query": "documentation", "unfinished": True,
    }], thread="docs")
    assert "Prepare documentation" in result["response"]
    assert "Review documentation" in result["response"]
    assert "Archive documentation" not in result["response"]


def test_05_to_10_full_lifecycle_and_vector_metadata(service):
    original = service.create_task(
        title="Demo project", description="Initial text", priority="High",
        due_date="2026-12-01",
    )
    actions = [
        {"operation": "CHANGE_STATUS", "target_id": original.id, "status": "In progress"},
        {"operation": "UPDATE", "target_id": original.id, "priority": "Low"},
        {"operation": "COMPLETE", "target_id": original.id},
        {"operation": "UPDATE", "target_id": original.id,
         "title": "Finished demonstration", "description": "Updated text"},
        {"operation": "UPDATE", "target_id": original.id, "clear_due_date": True},
    ]
    _, result = start(service, actions, thread="lifecycle")
    assert not result.get("validation_errors")
    final = service.get_task(original.id)
    assert final.status.value == "Done"
    assert final.priority.value == "Low"
    assert final.title == "Finished demonstration"
    assert final.description == "Updated text"
    assert final.due_date is None
    metadata = service.vector_store.collection.get(
        ids=[original.id], include=["metadatas", "documents"],
    )
    assert metadata["metadatas"][0]["status"] == "Done"
    assert metadata["metadatas"][0]["priority"] == "Low"
    assert metadata["metadatas"][0]["revision"] == final.revision
    assert "due_date" not in metadata["metadatas"][0]
    assert "Updated text" in metadata["documents"][0]

    _, deleted = start(service, [{
        "operation": "DELETE", "target_id": original.id,
    }], thread="deletion")
    assert "Deleted" in deleted["response"]
    assert service.get_task(original.id) is None
    assert all(task.id != original.id for task in service.list_tasks())
    assert all(item.task.id != original.id
               for item in service.search_tasks("demonstration"))
    assert service.vector_store.collection.get(ids=[original.id])["ids"] == []


def test_11_duplicate_title_requires_clarification_without_mutation(service):
    a = service.create_task(title="Weekly report")
    b = service.create_task(title="Weekly report")
    graph, result = start(service, [{
        "operation": "UPDATE", "target_title": "Weekly report", "title": "Revised report",
    }], thread="duplicate-title")
    assert "Which task" in result["response"]
    assert service.get_task(a.id).title == "Weekly report"
    assert service.get_task(b.id).title == "Weekly report"
    resolved = answer(graph, a.id, "duplicate-title", "duplicate-title-message-2")
    assert "Updated" in resolved["response"]
    assert service.get_task(a.id).title == "Revised report"
    assert service.get_task(b.id).title == "Weekly report"


def test_12_ambiguous_delete_does_not_delete_until_clarified(service):
    a = service.create_task(title="Duplicate")
    b = service.create_task(title="Duplicate")
    graph, result = start(service, [{
        "operation": "DELETE", "target_title": "Duplicate",
    }], thread="ambiguous-delete")
    assert result["pending_clarification"] is not None
    assert service.count_tasks() == 2
    result = answer(graph, "2", "ambiguous-delete", "ambiguous-delete-message-2")
    assert "Deleted" in result["response"]
    assert service.get_task(a.id) is not None
    assert service.get_task(b.id) is None


def test_13_mixed_valid_ambiguous_batch_is_all_or_nothing(service):
    original = service.create_task(title="Unambiguous")
    service.create_task(title="Duplicate")
    service.create_task(title="Duplicate")
    _, result = start(service, [
        {"operation": "COMPLETE", "target_id": original.id},
        {"operation": "DELETE", "target_title": "Duplicate"},
    ], thread="atomic")
    assert result["pending_clarification"] is not None
    assert service.get_task(original.id).status.value == "Todo"
    assert service.count_tasks() == 3


@pytest.mark.parametrize("action", [
    {"operation": "CREATE", "title": "x", "status": "Paused"},
    {"operation": "CREATE", "title": "x", "priority": "Urgent"},
    {"operation": "CREATE"},
    {"operation": "UPDATE", "target_id": "TSK-000001", "description": ""},
])
def test_14_invalid_action_plans_rejected(action):
    with pytest.raises(ValidationError):
        ActionPlan.model_validate({"actions": [action]})


def test_14_invalid_calendar_date_unknown_id_llm_and_sqlite_error(service, monkeypatch):
    _, invalid_date = start(service, [{
        "operation": "CREATE", "title": "Never created", "due_date": "2026-02-30",
    }], thread="invalid-calendar")
    assert invalid_date.get("pending_clarification") is not None
    assert service.count_tasks() == 0

    _, unknown = start(service, [{
        "operation": "DELETE", "target_id": "TSK-999999",
    }], thread="unknown-id")
    assert unknown.get("pending_clarification") is not None
    assert service.count_tasks() == 0

    planner = FakePlanner(error=RuntimeError("Simulated model outage"))
    graph = build_agent_graph(service, planner=planner)
    result = graph.invoke({"user_request": "Create important task"},
                          config={"configurable": {"thread_id": "llm-down"}})
    assert "No changes were made" in result["response"]

    def fail(_mutations, **_kwargs):
        raise OSError("Simulated database outage")
    monkeypatch.setattr(service.database, "apply_mutations", fail)
    _, result = start(service, [{
        "operation": "CREATE", "title": "Still not created",
    }], thread="db-down")
    assert "Batch execution failed" in result["response"]
    assert service.count_tasks() == 0


def test_14_chroma_failure_retains_sqlite_and_is_recoverable(service, monkeypatch):
    saved_upsert = service.vector_store.upsert_task

    def fail(_task):
        raise OSError("Index offline")

    monkeypatch.setattr(service.vector_store, "upsert_task", fail)
    _, result = start(service, [{
        "operation": "CREATE", "title": "Recovered report",
    }], thread="chroma-down")
    assert "Created" in result["response"]
    assert "synchronization is pending" in result["response"]
    task = service.all_tasks()[0]
    assert len(service.database.pending_index_changes()) == 1
    monkeypatch.setattr(service.vector_store, "upsert_task", saved_upsert)
    service.reconcile_index()
    assert not service.database.pending_index_changes()
    assert service.search_tasks("report")[0].task.id == task.id


def test_rebuild_index_from_sqlite_repairs_missing_entries(service):
    task = service.create_task(title="Prepare project demo")
    service.vector_store.remove_task(task.id)
    assert service.vector_store.collection.get(ids=[task.id])["ids"] == []
    service.reconcile_index()
    assert service.search_tasks("demonstration")[0].task.id == task.id


def test_meeting_datetime_preserves_time_without_database_migration(service):
    """Regression: model may incorrectly put 'tomorrow at 4 pm' in due_date."""
    _, result = start(service, [{
        "operation": "CREATE",
        "title": "Meeting with SpaceX CEO",
        "due_date": "tomorrow at 4 pm",
    }], request="I have a meeting tomorrow at 4 pm with CEO of SpaceX",
       thread="meeting-datetime")
    assert "Created" in result["response"]
    created = service.all_tasks()[0]
    assert created.due_date == dhaka_today() + timedelta(days=1)
    assert created.description == "Time: 4:00 PM"
    assert "Time: 4:00 PM" in service.vector_store.collection.get(
        ids=[created.id], include=["documents"]
    )["documents"][0]


def test_meeting_time_already_in_description_is_not_duplicated(service):
    _, result = start(service, [{
        "operation": "CREATE",
        "title": "Meeting with SpaceX CEO",
        "due_date": "tomorrow",
        "description": "Time: 4:00 PM",
    }], thread="meeting-time-separate")
    assert "Created" in result["response"]
    assert service.all_tasks()[0].description == "Time: 4:00 PM"


def test_update_meeting_time_preserves_existing_description(service):
    initial = service.create_task(
        title="Client meeting", description="Discuss launch timeline",
    )
    _, result = start(service, [{
        "operation": "UPDATE",
        "target_id": initial.id,
        "due_date": "next Friday at 16:00",
    }], thread="meeting-update")
    assert "Updated" in result["response"]
    updated = service.get_task(initial.id)
    assert updated.description == "Discuss launch timeline\nTime: 4:00 PM"
    assert updated.due_date is not None


def test_ambiguous_meeting_update_retains_time_through_clarification(service):
    first = service.create_task(title="Team meeting", description="Original one")
    second = service.create_task(title="Team meeting", description="Original two")
    graph, result = start(service, [{
        "operation": "UPDATE",
        "target_title": "Team meeting",
        "due_date": "tomorrow at 4 pm",
    }], thread="meeting-ambiguous")
    assert "Which task" in result["response"]
    assert service.get_task(first.id).due_date is None
    assert service.get_task(second.id).due_date is None
    selected = answer(
        graph, first.id, "meeting-ambiguous", "meeting-ambiguous-message-2",
    )
    assert "Updated" in selected["response"]
    assert service.get_task(first.id).description == "Original one\nTime: 4:00 PM"
    assert service.get_task(second.id).description == "Original two"


class RequestPlanner:
    """In-memory LLM replacement for testing instructions over one thread."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def __call__(self, request):
        self.calls.append(request)
        return ActionPlan.model_validate({"actions": self.mapping[request]})


def test_pending_clarification_does_not_lock_all_future_commands(service):
    demo = service.create_task(title="Prepare my demo")
    meeting = service.create_task(title="Meeting with SpaceX CEO")
    initial_request = "Move the demo task to In progress and mark the weekly report Done."
    planner = RequestPlanner({
        initial_request: [
            {"operation": "CHANGE_STATUS", "target_title": "demo task", "status": "In progress"},
            {"operation": "COMPLETE", "target_title": "weekly report"},
        ],
    })
    graph = build_agent_graph(service, planner=planner)
    thread = "no-stuck-chat"

    result = answer(graph, initial_request, thread, "message-1")
    # Unique partial title for 'demo task' resolves, but missing weekly report
    # blocks the WHOLE mutation batch, including the resolved first action.
    assert result["pending_clarification"] is not None
    assert "Action 2" in result["response"]
    assert service.get_task(demo.id).status.value == "Todo"

    result = answer(graph, "show all tasks", thread, "message-2")
    assert result.get("pending_clarification") is None
    assert "Task list (2)" in result["response"]
    assert demo.title in result["response"]
    assert meeting.title in result["response"]
    assert planner.calls == [initial_request]  # simple LIST needs no LLM
    assert service.get_task(demo.id).status.value == "Todo"


def test_pending_ambiguity_can_be_canceled_without_changes(service):
    a = service.create_task(title="Weekly report")
    b = service.create_task(title="Weekly report")
    planner = RequestPlanner({"complete the weekly report": [
        {"operation": "COMPLETE", "target_title": "weekly report"},
    ]})
    graph = build_agent_graph(service, planner=planner)
    thread = "cancel-stale-request"
    initial = answer(graph, "complete the weekly report", thread, "message-1")
    assert initial["pending_clarification"]

    result = answer(graph, "cancel", thread, "message-2")
    assert "Canceled" in result["response"]
    assert result.get("pending_clarification") is None
    assert service.get_task(a.id).status.value == "Todo"
    assert service.get_task(b.id).status.value == "Todo"

    followup = answer(graph, "find all", thread, "message-3")
    assert "Task list (2)" in followup["response"]


def test_new_mutation_command_replaces_previous_pending_batch(service):
    task = service.create_task(title="Prepare my demo")
    service.create_task(title="Meeting with SpaceX CEO")
    initial_request = "Complete weekly report"
    next_request = "Move the task that has demo in the title to In progress"
    planner = RequestPlanner({
        initial_request: [{"operation": "COMPLETE", "target_title": "weekly report"}],
        next_request: [{
            "operation": "CHANGE_STATUS", "target_title": "task that has demo in its title",
            "status": "In progress",
        }],
    })
    graph = build_agent_graph(service, planner=planner)
    thread = "new-mutation"
    first = answer(graph, initial_request, thread, "message-1")
    assert first["pending_clarification"]

    result = answer(graph, next_request, thread, "message-2")
    assert not result.get("pending_clarification")
    assert "Changed status" in result["response"]
    assert service.get_task(task.id).status.value == "In progress"
    assert planner.calls == [initial_request, next_request]


def test_partial_title_requiring_confirmation_when_not_unique(service):
    first = service.create_task(title="Prepare my demo")
    second = service.create_task(title="Review the demo")
    planner = RequestPlanner({"complete the demo task": [{
        "operation": "COMPLETE", "target_title": "demo task",
    }]})
    graph = build_agent_graph(service, planner=planner)
    thread = "fragment-ambiguity"
    initial = answer(graph, "complete the demo task", thread, "message-1")
    assert initial["pending_clarification"]
    assert len(initial["pending_clarification"]["candidates"]) == 2
    assert service.get_task(first.id).status.value == "Todo"
    assert service.get_task(second.id).status.value == "Todo"

    result = answer(graph, "2", thread, "message-2")
    assert result["pending_clarification"] is None
    assert service.get_task(first.id).status.value == "Todo"
    assert service.get_task(second.id).status.value == "Done"


def test_date_clarification_can_be_interrupted_by_a_read_request(service):
    planner = RequestPlanner({"create a task sometime next month": [
        {"operation": "CREATE", "title": "Submit report", "due_date": "next month"},
    ]})
    graph = build_agent_graph(service, planner=planner)
    thread = "date-pending-new-command"
    result = answer(graph, "create a task sometime next month", thread, "message-1")
    assert result["pending_clarification"]
    assert service.count_tasks() == 0

    result = answer(graph, "show all tasks", thread, "message-2")
    assert result["pending_clarification"] is None
    assert "No tasks found" in result["response"]
    assert service.count_tasks() == 0


@pytest.mark.parametrize(
    ("request_text", "selection"),
    [("delete all tasks", "2"), ("delete tasks", "TSK-000002")],
)
def test_unqualified_delete_lists_tasks_and_deletes_only_selection(
    service, request_text, selection,
):
    first = service.create_task(title="Prepare project demo")
    second = service.create_task(title="Review project report")
    planner = RequestPlanner({})
    graph = build_agent_graph(service, planner=planner)
    thread = "unqualified-delete"

    prompt = answer(graph, request_text, thread, "message-1")

    assert prompt["pending_clarification"] is not None
    assert "Which task should I delete?" in prompt["response"]
    assert first.id in prompt["response"]
    assert first.title in prompt["response"]
    assert second.id in prompt["response"]
    assert second.title in prompt["response"]
    assert service.count_tasks() == 2
    assert planner.calls == []

    result = answer(graph, selection, thread, "message-2")

    assert result["pending_clarification"] is None
    assert "Deleted TSK-000002" in result["response"]
    assert service.get_task(first.id) is not None
    assert service.get_task(second.id) is None


def test_unqualified_delete_with_no_tasks_reports_empty_list(service):
    planner = RequestPlanner({})
    graph = build_agent_graph(service, planner=planner)

    result = answer(graph, "delete tasks", "empty-delete", "message-1")

    assert result["response"] == "There are no tasks to delete."
    assert result.get("pending_clarification") is None
    assert planner.calls == []


def test_unqualified_delete_does_not_delete_task_changed_after_prompt(service):
    task = service.create_task(title="Prepare project demo")
    graph = build_agent_graph(service, planner=RequestPlanner({}))
    thread = "stale-delete-choice"

    prompt = answer(graph, "delete tasks", thread, "message-1")
    assert prompt["pending_clarification"] is not None

    service.update_task(task.id, title="Updated project demo")
    result = answer(graph, "1", thread, "message-2")

    assert "deleted or changed" in result["response"]
    assert service.get_task(task.id).title == "Updated project demo"


def test_title_search_lists_all_literal_title_matches_without_planner(service):
    first = service.create_task(title="Hangout with Tanvir")
    second = service.create_task(title="Plan a weekend hangout")
    service.create_task(title="Prepare project demo")
    planner = RequestPlanner({})
    graph = build_agent_graph(service, planner=planner)

    result = answer(
        graph,
        "find task that have hangout on its title",
        "literal-title-search",
        "message-1",
    )

    assert "Title matches (2)" in result["response"]
    assert first.title in result["response"]
    assert second.title in result["response"]
    assert "Prepare project demo" not in result["response"]
    assert planner.calls == []


def test_appropriate_description_updates_task_and_preserves_notes(
    service,
):
    task = service.create_task(
        title="Finish AI assignment",
        description="Time: 10:00 PM",
    )
    planner = RequestPlanner({})
    graph = build_agent_graph(service, planner=planner)

    result = answer(
        graph,
        "add an appropriate description for the task finish ai assignment",
        "appropriate-description",
        "message-1",
    )

    updated = service.get_task(task.id)
    assert "Updated" in result["response"]
    assert "Time: 10:00 PM" in updated.description
    assert "reviewing the finished work" in updated.description
    assert planner.calls == []


def test_related_search_phrase_uses_semantic_search_without_planner(service):
    service.create_task(title="Finish AI assignment")
    service.create_task(title="Plan a weekend hangout")
    planner = RequestPlanner({})
    graph = build_agent_graph(service, planner=planner)

    result = answer(
        graph,
        "show me task related to ai assignment",
        "related-ai-search",
        "message-1",
    )

    assert "Finish AI assignment" in result["response"]
    assert planner.calls == []


def test_related_search_phrase_respects_unfinished_filter(service):
    unfinished = service.create_task(
        title="Finish AI assignment", status="Todo"
    )
    service.create_task(title="Review AI assignment", status="Done")
    planner = RequestPlanner({})
    graph = build_agent_graph(service, planner=planner)

    result = answer(
        graph,
        "show all unfinished tasks related to ai assignment",
        "unfinished-related-search",
        "message-1",
    )

    assert unfinished.title in result["response"]
    assert "Review AI assignment" not in result["response"]
    assert planner.calls == []
