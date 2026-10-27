
"""Stage 2 integration tests with isolated SQLite and ChromaDB files."""

import re
from pathlib import Path

import pytest
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings,
)
from pydantic import ValidationError

from task_manager.database import (
    TaskDatabase,
    TaskNotFoundError,
)
from task_manager.models import Priority, Status
from task_manager.task_service import (
    IndexSyncError,
    TaskService,
)
from task_manager.vector_store import TaskVectorStore


class TinyTestEmbeddings(EmbeddingFunction[Documents]):
    """Offline test vectors; production uses Chroma's local default."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "task-manager-test-embeddings"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(
        config: dict,
    ) -> "TinyTestEmbeddings":
        return TinyTestEmbeddings()

    def __call__(self, input: Documents) -> Embeddings:
        groups = (
            {"bill", "invoice", "billing"},
            {"electricity", "electric", "power"},
            {"pay", "payment", "settle"},
            {"report", "summary", "document"},
            {"quarterly", "quarter"},
            {"meeting", "call", "appointment"},
            {"groceries", "shopping", "supermarket"},
            {"project", "work", "client"},
        )

        vectors = []

        for text in input:
            words = set(
                re.findall(r"[a-z]+", text.lower())
            )
            vectors.append([
                float(len(words & group)) + 0.001
                for group in groups
            ])

        return vectors


@pytest.fixture
def db(tmp_path: Path):
    return TaskDatabase(
        tmp_path / "data" / "tasks.sqlite3"
    )


@pytest.fixture
def service(db: TaskDatabase, tmp_path: Path):
    index = TaskVectorStore(
        tmp_path / "data" / "chroma",
        TinyTestEmbeddings(),
    )
    return TaskService(db, index)


def test_create_ids_defaults(db):
    a = db.create_task(title="  Write proposal  ")
    b = db.create_task(
        title="Email professor",
        priority="High",
    )

    assert (a.id, b.id) == (
        "TSK-000001",
        "TSK-000002",
    )
    assert a.title == "Write proposal"
    assert a.status == Status.TODO
    assert a.priority == Priority.MEDIUM
    assert a.created_at.tzinfo
    assert a.updated_at.tzinfo
    assert db.count_tasks() == 2


def test_read_list_filter_and_count(db):
    first = db.create_task(
        title="First",
        status="In progress",
        priority="High",
    )
    db.create_task(
        title="Second",
        status="Done",
        priority="Low",
    )

    assert db.get_task(first.id).title == "First"
    assert db.get_task("TSK-999999") is None

    assert [
        x.title for x in db.list_tasks(
            priority="High"
        )
    ] == ["First"]

    assert [
        x.title for x in db.list_tasks(
            status="Done"
        )
    ] == ["Second"]

    assert db.count_tasks(
        status="In progress",
        priority="High",
    ) == 1

    assert len(db.all_tasks()) == 2


def test_update_all_fields_and_clear_optional(db):
    task = db.create_task(
        title="Draft",
        description="Old",
        due_date="2026-11-10",
    )

    changed = db.update_task(
        task.id,
        title="Final draft",
        description="New text",
        status="In progress",
        priority="High",
        due_date="2026-12-01",
    )

    assert changed.title == "Final draft"
    assert changed.description == "New text"
    assert changed.status == Status.IN_PROGRESS
    assert changed.priority == Priority.HIGH
    assert changed.due_date.isoformat() == "2026-12-01"
    assert changed.revision == 2

    db.update_task(task.id, title="Retitled")

    assert (
        db.get_task(task.id).due_date.isoformat()
        == "2026-12-01"
    )

    cleared = db.update_task(
        task.id,
        description=None,
        due_date=None,
    )

    assert cleared.description is None
    assert cleared.due_date is None


def test_status_transitions_completion_and_delete(db):
    task = db.create_task(
        title="Submit homework"
    )

    assert db.change_status(
        task.id, "In progress"
    ).status == Status.IN_PROGRESS

    assert db.complete_task(
        task.id
    ).status == Status.DONE

    assert db.change_status(
        task.id, "Todo"
    ).status == Status.TODO

    assert db.delete_task(task.id).id == task.id
    assert db.get_task(task.id) is None
    assert db.count_tasks() == 0

    with pytest.raises(TaskNotFoundError):
        db.delete_task(task.id)


@pytest.mark.parametrize(
    "fields",
    [
        {"title": " "},
        {"title": "x", "status": "Started"},
        {"title": "x", "priority": "Urgent"},
        {"title": "x", "due_date": "2026-02-30"},
        {"title": "x", "due_date": "next Friday"},
        {"title": "x", "unexpected": True},
    ],
)
def test_invalid_create(db, fields):
    with pytest.raises(ValidationError):
        db.create_task(**fields)

    assert db.count_tasks() == 0


@pytest.mark.parametrize(
    "fields",
    [
        {"title": "  "},
        {"title": None},
        {"status": None},
        {"priority": "Critical"},
        {"priority": None},
        {"due_date": "2026-13-01"},
        {"random_column": 1},
    ],
)
def test_invalid_update_is_atomic(db, fields):
    task = db.create_task(title="Unchanged")

    with pytest.raises((ValidationError, ValueError)):
        db.update_task(task.id, **fields)

    assert db.get_task(task.id).title == "Unchanged"
    assert db.get_task(task.id).revision == 1


def test_no_empty_patch_and_invalid_filters(db):
    task = db.create_task(title="Safe")

    with pytest.raises(ValueError):
        db.update_task(task.id)

    with pytest.raises(ValidationError):
        db.list_tasks(status="Archived")

    with pytest.raises(ValueError):
        db.get_task("1")


def test_persistence_and_autoincrement_after_reopen(
    tmp_path,
):
    path = tmp_path / "tasks.sqlite3"

    first = TaskDatabase(path).create_task(
        title="Survive restart"
    )

    second_db = TaskDatabase(path)

    assert (
        second_db.get_task(first.id).title
        == "Survive restart"
    )

    second_db.delete_task(first.id)

    assert TaskDatabase(path).create_task(
        title="Next"
    ).id == "TSK-000002"


def test_vector_upsert_search_filters_and_delete(service):
    bill = service.create_task(
        title="Pay electricity bill",
        priority="High",
    )

    service.create_task(
        title="Review quarterly financial report",
        priority="Low",
    )

    matches = service.search_tasks(
        "settle power invoice"
    )

    assert matches
    assert matches[0].task.id == bill.id

    assert service.search_tasks(
        "settle power invoice",
        priority="High",
    )[0].task.id == bill.id

    assert service.search_tasks(
        "settle power invoice",
        status="Done",
    ) == []

    assert service.vector_store.collection.get(
        ids=[bill.id]
    )["ids"] == [bill.id]

    service.delete_task(bill.id)

    assert bill.id not in [
        result.task.id
        for result in service.search_tasks(
            "settle power invoice"
        )
    ]

    assert service.vector_store.collection.get(
        ids=[bill.id]
    )["ids"] == []


def test_vector_updates_text_metadata_and_cleared_date(
    service,
):
    task = service.create_task(
        title="Shopping",
        due_date="2026-12-08",
    )

    service.update_task(
        task.id,
        title="Pay electricity bill",
        description="Settle invoice",
        priority="High",
        status="In progress",
        due_date=None,
    )

    record = service.vector_store.collection.get(
        ids=[task.id],
        include=["documents", "metadatas"],
    )

    assert "electricity" in record["documents"][0]

    metadata = record["metadatas"][0]

    assert metadata["status"] == "In progress"
    assert metadata["priority"] == "High"
    assert "due_date" not in metadata
    assert metadata["revision"] == 2

    assert service.search_tasks(
        "power invoice",
        status="In progress",
        priority="High",
    )[0].task.id == task.id


def test_index_survives_reopen(service, tmp_path):
    task = service.create_task(
        title="Pay electricity bill"
    )

    reopened_db = TaskDatabase(
        tmp_path / "data" / "tasks.sqlite3"
    )

    reopened_index = TaskVectorStore(
        tmp_path / "data" / "chroma",
        TinyTestEmbeddings(),
    )

    reopened = TaskService(
        reopened_db,
        reopened_index,
    )

    assert reopened.search_tasks(
        "settle power invoice"
    )[0].task.id == task.id


def test_full_reconcile_fixes_missing_orphans_and_outdated(
    service,
):
    task = service.create_task(
        title="Pay electricity bill",
        status="Todo",
    )

    index = service.vector_store

    index.remove_task(task.id)

    index.collection.upsert(
        ids=["TSK-999999"],
        documents=["orphan record"],
        metadatas=[{
            "task_id": "TSK-999999",
            "status": "Todo",
            "priority": "Low",
            "revision": 1,
        }],
    )

    assert service.reconcile_index() == (1, 1)

    assert index.collection.get(
        ids=[task.id]
    )["ids"] == [task.id]

    assert index.collection.get(
        ids=["TSK-999999"]
    )["ids"] == []

    assert service.reconcile_index() == (0, 0)


def test_index_failure_keeps_committed_data_and_queue(
    service,
):
    real_upsert = service.vector_store.upsert_task

    def broken_upsert(task):
        raise OSError("simulated index failure")

    service.vector_store.upsert_task = broken_upsert

    with pytest.raises(
        IndexSyncError,
        match="SQLite changes are committed",
    ):
        service.create_task(
            title="Pay electricity bill"
        )

    task = service.database.all_tasks()[0]

    assert service.database.get_task(
        task.id
    ) is not None

    assert len(
        service.database.pending_index_changes()
    ) == 1

    service.vector_store.upsert_task = real_upsert

    assert service.sync_pending() == 1
    assert service.database.pending_index_changes() == []

    assert service.search_tasks(
        "power invoice"
    )[0].task.id == task.id


def test_deletion_failure_reconcile_prevents_reappearing(
    service,
):
    task = service.create_task(
        title="Pay electricity bill"
    )

    real_remove = service.vector_store.remove_task

    def broken_remove(task_id):
        raise OSError(
            "simulated delete index failure"
        )

    service.vector_store.remove_task = broken_remove

    with pytest.raises(IndexSyncError):
        service.delete_task(task.id)

    assert service.database.get_task(
        task.id
    ) is None

    service.vector_store.remove_task = real_remove

    service.reconcile_index()

    assert service.search_tasks(
        "settle power invoice"
    ) == []

    assert service.database.pending_index_changes() == []
