
"""SQLite source of truth, with a durable Chroma sync queue."""

import sqlite3

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import (
    Priority,
    Status,
    Task,
    TaskCreate,
    TaskFilters,
    TaskUpdate,
    format_task_id,
    parse_task_id,
)


class TaskNotFoundError(LookupError):
    """The requested SQLite task does not exist."""


@dataclass(frozen=True)
class PendingIndexChange:
    task_id: str
    operation: str
    token: int


class TaskDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row

        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 10000")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self):
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")

            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def _initialize(self):
        statements = (
            """CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL
                    CHECK (length(trim(title)) > 0),
                description TEXT,
                status TEXT NOT NULL
                    CHECK (status IN ('Todo', 'In progress', 'Done')),
                priority TEXT NOT NULL
                    CHECK (priority IN ('Low', 'Medium', 'High')),
                due_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 1
            )""",
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority)",
            """CREATE TABLE IF NOT EXISTS task_index_queue (
                task_id INTEGER PRIMARY KEY,
                operation TEXT NOT NULL
                    CHECK (operation IN ('upsert', 'delete')),
                token INTEGER NOT NULL DEFAULT 1
            )""",
        )

        with self._transaction() as conn:
            for statement in statements:
                conn.execute(statement)

    @staticmethod
    def _row_to_task(row):
        data = dict(row)
        data["id"] = format_task_id(data["id"])
        return Task.model_validate(data)

    @staticmethod
    def _now():
        return datetime.now(
            ZoneInfo("Asia/Dhaka")
        ).isoformat(timespec="microseconds")

    @staticmethod
    def _enqueue(conn, number: int, operation: str):
        conn.execute(
            """
            INSERT INTO task_index_queue (task_id, operation)
            VALUES (?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                operation = excluded.operation,
                token = task_index_queue.token + 1
            """,
            (number, operation),
        )

    def create_task(self, **fields) -> Task:
        data = TaskCreate.model_validate(fields)
        now = self._now()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tasks
                (title, description, status, priority, due_date,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.title,
                    data.description,
                    data.status.value,
                    data.priority.value,
                    data.due_date.isoformat()
                    if data.due_date else None,
                    now,
                    now,
                ),
            )

            number = cursor.lastrowid
            self._enqueue(conn, number, "upsert")

            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (number,),
            ).fetchone()

        return self._row_to_task(row)

    def get_task(self, task_id: str) -> Task | None:
        number = parse_task_id(task_id)

        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (number,),
            ).fetchone()

        return self._row_to_task(row) if row else None

    @staticmethod
    def _filter_clause(status, priority):
        filters = TaskFilters(
            status=status,
            priority=priority,
        )

        clauses = []
        values = []

        if filters.status is not None:
            clauses.append("status = ?")
            values.append(filters.status.value)

        if filters.priority is not None:
            clauses.append("priority = ?")
            values.append(filters.priority.value)

        where = (
            " WHERE " + " AND ".join(clauses)
            if clauses else ""
        )

        return where, values

    def list_tasks(
        self,
        status: Status | str | None = None,
        priority: Priority | str | None = None,
    ) -> list[Task]:
        where, values = self._filter_clause(
            status, priority
        )

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks" + where + " ORDER BY id",
                values,
            ).fetchall()

        return [self._row_to_task(row) for row in rows]

    def all_tasks(self) -> list[Task]:
        return self.list_tasks()

    def count_tasks(
        self,
        status: Status | str | None = None,
        priority: Priority | str | None = None,
    ) -> int:
        where, values = self._filter_clause(
            status, priority
        )

        with self._connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM tasks" + where,
                values,
            ).fetchone()[0]

    def update_task(
        self, task_id: str, **fields
    ) -> Task:
        number = parse_task_id(task_id)
        patch = TaskUpdate.model_validate(fields)

        changes = patch.model_dump(
            exclude_unset=True
        )

        if not changes:
            raise ValueError(
                "Specify at least one field to update"
            )

        required_fields = (
            "title", "status", "priority"
        )

        if any(
            changes.get(name, "present") is None
            for name in required_fields
        ):
            raise ValueError(
                "Title, status and priority cannot be cleared"
            )

        for name, value in changes.items():
            if isinstance(value, (Status, Priority)):
                changes[name] = value.value

            elif (
                name == "due_date"
                and value is not None
            ):
                changes[name] = value.isoformat()

        # Column names come only from validated model fields.
        # User-provided values are always SQL parameters.
        assignments = ", ".join(
            f"{name} = ?" for name in changes
        )

        with self._transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM tasks WHERE id = ?",
                (number,),
            ).fetchone()

            if not exists:
                raise TaskNotFoundError(task_id)

            conn.execute(
                f"""
                UPDATE tasks
                SET {assignments},
                    updated_at = ?,
                    revision = revision + 1
                WHERE id = ?
                """,
                [
                    *changes.values(),
                    self._now(),
                    number,
                ],
            )

            self._enqueue(
                conn, number, "upsert"
            )

            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (number,),
            ).fetchone()

        return self._row_to_task(row)

    def change_status(
        self,
        task_id: str,
        status: Status | str,
    ) -> Task:
        return self.update_task(
            task_id, status=status
        )

    def complete_task(self, task_id: str) -> Task:
        return self.change_status(
            task_id, Status.DONE
        )

    def delete_task(self, task_id: str) -> Task:
        number = parse_task_id(task_id)

        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (number,),
            ).fetchone()

            if not row:
                raise TaskNotFoundError(task_id)

            conn.execute(
                "DELETE FROM tasks WHERE id = ?",
                (number,),
            )

            self._enqueue(
                conn, number, "delete"
            )

        return self._row_to_task(row)

    def pending_index_changes(
        self,
    ) -> list[PendingIndexChange]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT task_id, operation, token
                FROM task_index_queue
                ORDER BY task_id
                """
            ).fetchall()

        return [
            PendingIndexChange(
                format_task_id(row["task_id"]),
                row["operation"],
                row["token"],
            )
            for row in rows
        ]

    def acknowledge_index_change(
        self,
        change: PendingIndexChange,
    ) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                DELETE FROM task_index_queue
                WHERE task_id = ? AND token = ?
                """,
                (
                    parse_task_id(change.task_id),
                    change.token,
                ),
            )
