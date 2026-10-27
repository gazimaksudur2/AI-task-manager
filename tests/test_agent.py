
"""Stage 3 tests: offline LLM plans, real SQLite and Chroma services."""

from datetime import date

import pytest
from pydantic import ValidationError

from task_manager.agent_models import ActionPlan
from task_manager.database import TaskDatabase
from task_manager.graph import (
    build_agent_graph,
    resolve_due_date,
    DateResolutionError,
)
from task_manager.task_service import TaskService
from task_manager.vector_store import TaskVectorStore

from tests.test_data_layer import TinyTestEmbeddings


class FakePlanner:
    def __init__(self, plan=None, error=None):
        self.plan = plan
        self.error = error
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return ActionPlan.model_validate(self.plan)


@pytest.fixture
def service(tmp_path):
    db = TaskDatabase(tmp_path / "tasks.sqlite3")
    index = TaskVectorStore(
        tmp_path / "chroma",
        TinyTestEmbeddings(),
    )
    return TaskService(db, index)


def run_graph(service, actions, thread="test"):
    planner = FakePlanner({"actions": actions})
    graph = build_agent_graph(service, planner=planner)
    result = graph.invoke(
        {"user_request": "test request"},
        config={"configurable": {"thread_id": thread}},
    )
    return graph, planner, result


@pytest.mark.parametrize(
    "action",
    [
        {"operation": "CREATE"},
        {"operation": "DELETE"},
        {"operation": "SEARCH"},
        {
            "operation": "CHANGE_STATUS",
            "target_id": "TSK-000001",
        },
        {
            "operation": "CREATE",
            "title": "x",
            "priority": "Critical",
        },
        {
            "operation": "CREATE",
            "title": "x",
            "status": "Started",
        },
        {
            "operation": "DELETE",
            "target_id": "TSK-000001",
            "query": "something",
        },
        {
            "operation": "UPDATE",
            "target_id": "TSK-000001",
            "clear_due_date": True,
            "due_date": "tomorrow",
        },
    ],
)
def test_invalid_action_schemas(action):
    with pytest.raises(ValidationError):
        ActionPlan.model_validate(
            {"actions": [action]}
        )


def test_empty_plan_rejected():
    with pytest.raises(ValidationError):
        ActionPlan.model_validate({"actions": []})


def test_create_list_and_help(service):
    _, _, result = run_graph(
        service,
        [
            {
                "operation": "CREATE",
                "title": "Write report",
                "priority": "High",
            },
            {"operation": "LIST", "priority": "High"},
            {"operation": "HELP"},
        ],
    )

    assert service.count_tasks() == 1
    assert service.get_task("TSK-000001").title == "Write report"
    assert "Created TSK-000001" in result["response"]
    assert "Task list (1)" in result["response"]
    assert "You can create" in result["response"]


def test_update_and_clear_optional_fields(service):
    task = service.create_task(
        title="Old report",
        description="Draft",
        due_date="2026-11-01",
    )

    _, _, result = run_graph(
        service,
        [{
            "operation": "UPDATE",
            "target_id": task.id,
            "title": "New report",
            "priority": "High",
            "clear_description": True,
            "clear_due_date": True,
        }],
    )

    updated = service.get_task(task.id)

    assert updated.title == "New report"
    assert updated.priority.value == "High"
    assert updated.description is None
    assert updated.due_date is None
    assert "Updated" in result["response"]


def test_status_completion_and_deletion(service):
    task = service.create_task(title="Finish assignment")

    graph, _, result = run_graph(
        service,
        [{
            "operation": "CHANGE_STATUS",
            "target_id": task.id,
            "status": "In progress",
        }],
    )

    assert service.get_task(task.id).status.value == "In progress"

    result = graph.invoke(
        {"user_request": "mark it done"},
        config={"configurable": {"thread_id": "same"}},
    )

    # A fake planner returns the same action. With no conversational
    # pronoun resolver yet, this remains the same status request.
    assert service.get_task(task.id).status.value == "In progress"

    _, _, completed = run_graph(
        service,
        [{
            "operation": "COMPLETE",
            "target_id": task.id,
        }],
        thread="completion",
    )
    assert service.get_task(task.id).status.value == "Done"
    assert "Completed" in completed["response"]

    _, _, deleted = run_graph(
        service,
        [{
            "operation": "DELETE",
            "target_id": task.id,
        }],
        thread="deletion",
    )
    assert service.get_task(task.id) is None
    assert "Deleted" in deleted["response"]


def test_multiple_actions_are_atomic(service):
    first = service.create_task(title="Demo task")
    second = service.create_task(title="Weekly report")

    _, _, result = run_graph(
        service,
        [
            {
                "operation": "CHANGE_STATUS",
                "target_title": "Demo task",
                "status": "In progress",
            },
            {
                "operation": "COMPLETE",
                "target_title": "Weekly report",
            },
        ],
    )

    assert not result.get("pending_clarification")
    assert service.get_task(first.id).status.value == "In progress"
    assert service.get_task(second.id).status.value == "Done"


def test_ambiguous_title_blocks_entire_batch(service):
    first = service.create_task(title="Demo task")
    service.create_task(title="Weekly report")
    second = service.create_task(title="Weekly report")

    planner = FakePlanner({
        "actions": [
            {
                "operation": "COMPLETE",
                "target_id": first.id,
            },
            {
                "operation": "DELETE",
                "target_title": "Weekly report",
            },
        ]
    })

    graph = build_agent_graph(service, planner=planner)
    config = {"configurable": {"thread_id": "ambiguous"}}

    result = graph.invoke(
        {"user_request": "complete demo and delete weekly report"},
        config=config,
    )

    assert "Which task" in result["response"]
    assert service.get_task(first.id).status.value == "Todo"
    assert service.count_tasks() == 3

    result = graph.invoke(
        {"user_request": "the second one"},
        config=config,
    )

    assert service.get_task(first.id).status.value == "Done"
    assert service.get_task(second.id) is None
    assert service.count_tasks() == 2
    assert planner.calls == 1
    assert "Deleted" in result["response"]


def test_semantic_candidate_requires_confirmation(service):
    service.create_task(title="Pay electricity bill")
    service.create_task(title="Pay water bill")

    planner = FakePlanner({
        "actions": [{
            "operation": "DELETE",
            "target_description": "utility invoice payment",
        }]
    })

    graph = build_agent_graph(service, planner=planner)
    config = {"configurable": {"thread_id": "semantic"}}

    result = graph.invoke(
        {"user_request": "delete the utility payment"},
        config=config,
    )

    assert result["pending_clarification"] is not None
    assert service.count_tasks() == 2

    chosen = result["pending_clarification"]["candidates"][0]["id"]

    result = graph.invoke(
        {"user_request": chosen},
        config=config,
    )

    assert service.get_task(chosen) is None
    assert service.count_tasks() == 1
    assert "Deleted" in result["response"]
    assert planner.calls == 1


def test_missing_target_never_mutates(service):
    service.create_task(title="Existing task")

    _, _, result = run_graph(
        service,
        [{
            "operation": "DELETE",
            "target_id": "TSK-999999",
        }],
    )

    assert result["pending_clarification"] is not None
    assert service.count_tasks() == 1


def test_dates_are_deterministic():
    today = date(2026, 10, 27)

    assert resolve_due_date(
        "tomorrow", today
    ) == "2026-10-28"

    assert resolve_due_date(
        "next Friday", today
    ) == "2026-11-06"

    assert resolve_due_date(
        "next Monday", today
    ) == "2026-11-02"

    assert resolve_due_date(
        "in 3 days", today
    ) == "2026-10-30"

    with pytest.raises(DateResolutionError):
        resolve_due_date("2026-02-30", today)

    with pytest.raises(DateResolutionError):
        resolve_due_date("sometime next month", today)


def test_invalid_date_asks_without_creating(service):
    planner = FakePlanner({
        "actions": [{
            "operation": "CREATE",
            "title": "Submit report",
            "due_date": "sometime next month",
        }]
    })

    graph = build_agent_graph(service, planner=planner)
    config = {"configurable": {"thread_id": "date"}}

    result = graph.invoke(
        {"user_request": "submit report next month"},
        config=config,
    )

    assert service.count_tasks() == 0
    assert "YYYY-MM-DD" in result["response"]

    result = graph.invoke(
        {"user_request": "2026-11-20"},
        config=config,
    )

    assert service.count_tasks() == 1
    assert service.all_tasks()[0].due_date.isoformat() == "2026-11-20"
    assert planner.calls == 1


def test_search_unfinished_uses_authoritative_records(service):
    service.create_task(
        title="Prepare project documentation",
        status="Todo",
    )

    service.create_task(
        title="Archive finished documentation",
        status="Done",
    )

    _, _, result = run_graph(
        service,
        [{
            "operation": "SEARCH",
            "query": "project documentation",
            "unfinished": True,
        }],
    )

    assert "Ranked semantic candidates" in result["response"]
    assert "Prepare project documentation" in result["response"]
    assert "Archive finished documentation" not in result["response"]


def test_llm_failure_does_not_mutate(service):
    planner = FakePlanner(
        error=RuntimeError("API unavailable")
    )
    graph = build_agent_graph(service, planner=planner)

    result = graph.invoke(
        {"user_request": "create important task"},
        config={"configurable": {"thread_id": "error"}},
    )

    assert service.count_tasks() == 0
    assert "No changes were made" in result["response"]


def test_db_failure_is_not_reported_as_success(
    service, monkeypatch
):
    def broken_batch(mutations):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        service.database,
        "apply_mutations",
        broken_batch,
    )

    _, _, result = run_graph(
        service,
        [{
            "operation": "CREATE",
            "title": "Should not be created",
        }],
    )

    assert "Created" not in result["response"]
    assert "Batch execution failed" in result["response"]
    assert service.count_tasks() == 0


def test_sqlite_batch_rolls_back(service):
    task = service.create_task(title="Original")

    with pytest.raises(Exception):
        service.database.apply_mutations([
            {
                "operation": "UPDATE",
                "task_id": task.id,
                "expected_revision": task.revision,
                "fields": {"title": "Changed"},
            },
            {
                "operation": "DELETE",
                "task_id": "TSK-999999",
                "expected_revision": 1,
                "fields": {},
            },
        ])

    assert service.get_task(task.id).title == "Original"


def test_sync_failure_reports_committed_data(
    service, monkeypatch
):
    def broken_upsert(task):
        raise OSError("Chroma unavailable")

    monkeypatch.setattr(
        service.vector_store,
        "upsert_task",
        broken_upsert,
    )

    _, _, result = run_graph(
        service,
        [{
            "operation": "CREATE",
            "title": "Committed task",
        }],
    )

    assert service.count_tasks() == 1
    assert "Created TSK-000001" in result["response"]
    assert "synchronization is pending" in result["response"]
    assert len(
        service.database.pending_index_changes()
    ) == 1
