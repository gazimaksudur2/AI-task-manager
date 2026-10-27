"""SQLite-only reliability checks; run without AI/Chroma/Streamlit dependencies."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from task_manager.database import TaskDatabase, TaskNotFoundError


def test_database_lifecycle_and_utc_timestamps(tmp_path: Path):
    path = tmp_path / "data" / "tasks.sqlite3"
    db = TaskDatabase(path)
    first = db.create_task(title="Prepare demo", priority="High", due_date="2026-10-28")
    assert first.id == "TSK-000001"
    assert first.created_at.utcoffset().total_seconds() == 0
    assert first.updated_at.utcoffset().total_seconds() == 0
    assert db.get_task(first.id).due_date.isoformat() == "2026-10-28"
    assert db.update_task(first.id, status="In progress", priority="Low").revision == 2
    assert db.complete_task(first.id).status.value == "Done"
    assert TaskDatabase(path).get_task(first.id).status.value == "Done"
    assert db.update_task(first.id, due_date=None).due_date is None
    db.delete_task(first.id)
    assert TaskDatabase(path).get_task(first.id) is None
    second = TaskDatabase(path).create_task(title="New task")
    assert second.id == "TSK-000002"


def test_batch_really_rolls_back_after_first_write(tmp_path):
    db = TaskDatabase(tmp_path / "tasks.sqlite3")
    first = db.create_task(title="Keep unchanged")
    second = db.create_task(title="Trigger failure")
    with sqlite3.connect(db.path) as conn:
        conn.execute("""CREATE TRIGGER reject_deletion BEFORE DELETE ON tasks
        WHEN OLD.title = 'Trigger failure'
        BEGIN SELECT RAISE(ABORT, 'injected failure'); END""")
    with pytest.raises(sqlite3.IntegrityError):
        db.apply_mutations([
            {"operation": "UPDATE", "task_id": first.id,
             "expected_revision": first.revision, "fields": {"title": "Changed"}},
            {"operation": "DELETE", "task_id": second.id,
             "expected_revision": second.revision, "fields": {}},
        ])
    assert db.get_task(first.id).title == "Keep unchanged"
    assert db.get_task(first.id).revision == 1
    assert db.count_tasks() == 2


def test_idempotent_batch_does_not_duplicate_after_restart(tmp_path):
    path = tmp_path / "tasks.sqlite3"
    db = TaskDatabase(path)
    request = [{"operation": "CREATE", "fields": {"title": "Submitted once", "priority": "High"}}]
    first = db.apply_mutations(request, request_id="ui-event-123")
    reopened = TaskDatabase(path)
    replayed = reopened.apply_mutations(request, request_id="ui-event-123")
    assert reopened.count_tasks() == 1
    assert first[0]["task"].id == replayed[0]["task"].id
    assert first[0]["task"].created_at == replayed[0]["task"].created_at
    with pytest.raises(ValueError, match="already used"):
        reopened.apply_mutations(
            [{"operation": "CREATE", "fields": {"title": "Different"}}],
            request_id="ui-event-123",
        )
    assert reopened.count_tasks() == 1


def test_idempotent_update_with_stale_revision_does_not_apply_twice(tmp_path):
    db = TaskDatabase(tmp_path / "tasks.sqlite3")
    task = db.create_task(title="Update once")
    batch = [{"operation": "UPDATE", "task_id": task.id,
              "expected_revision": task.revision, "fields": {"priority": "Low"}}]
    result = db.apply_mutations(batch, request_id="update-event")
    replay = db.apply_mutations(batch, request_id="update-event")
    assert replay[0]["task"].revision == result[0]["task"].revision == 2
    assert db.get_task(task.id).revision == 2


def test_unknown_task_id_does_not_mutate(tmp_path):
    db = TaskDatabase(tmp_path / "tasks.sqlite3")
    db.create_task(title="Safe")
    with pytest.raises(TaskNotFoundError):
        db.apply_mutations([{"operation": "DELETE", "task_id": "TSK-999999",
                             "expected_revision": 1, "fields": {}}])
    assert db.count_tasks() == 1
