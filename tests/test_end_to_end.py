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
