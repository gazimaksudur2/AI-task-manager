from dataclasses import dataclass

from .database import TaskDatabase
from .models import Priority, Status, Task, TaskFilters
from .vector_store import TaskVectorStore


class IndexSyncError(RuntimeError):
    """SQLite committed, but Chroma could not be fully synchronized."""


class BatchSyncError(IndexSyncError):
    """SQLite batch committed, but Chroma synchronization failed."""

    def __init__(self, results):
        self.results = results
        super().__init__(
            "SQLite batch committed; Chroma synchronization pending"
        )


@dataclass(frozen=True)
class TaskSearchResult:
    task: Task
    distance: float


class TaskService:
    def __init__(
        self,
        database: TaskDatabase,
        vector_store: TaskVectorStore,
    ):
        self.database = database
        self.vector_store = vector_store

    def sync_pending(self) -> int:
        completed = 0

        for change in (
            self.database.pending_index_changes()
        ):
            try:
                task = self.database.get_task(
                    change.task_id
                )

                if (
                    change.operation == "delete"
                    or task is None
                ):
                    self.vector_store.remove_task(
                        change.task_id
                    )
                else:
                    self.vector_store.upsert_task(task)

                self.database.acknowledge_index_change(
                    change
                )

                completed += 1

            except Exception as exc:
                raise IndexSyncError(
                    "SQLite changes are committed; "
                    "Chroma sync is pending for "
                    f"{change.task_id}. "
                    "Run service.reconcile_index() "
                    "to repair the index."
                ) from exc

        return completed

    def reconcile_index(self) -> tuple[int, int]:
        """Repair missing, stale and orphaned entries."""
        snapshot = (
            self.database.pending_index_changes()
        )

        try:
            changes = self.vector_store.reconcile(
                self.database.all_tasks()
            )

            for entry in snapshot:
                self.database.acknowledge_index_change(
                    entry
                )

            return changes

        except Exception as exc:
            raise IndexSyncError(
                "Index repair failed; SQLite "
                "remains authoritative."
            ) from exc

    def create_task(self, **fields) -> Task:
        task = self.database.create_task(**fields)
        self.sync_pending()
        return task

    def get_task(
        self, task_id: str
    ) -> Task | None:
        return self.database.get_task(task_id)

    def list_tasks(
        self,
        status=None,
        priority=None,
    ) -> list[Task]:
        return self.database.list_tasks(
            status=status,
            priority=priority,
        )

    def count_tasks(
        self,
        status=None,
        priority=None,
    ) -> int:
        return self.database.count_tasks(
            status=status,
            priority=priority,
        )

    def all_tasks(self) -> list[Task]:
        return self.database.all_tasks()

    def update_task(
        self,
        task_id: str,
        **fields,
    ) -> Task:
        task = self.database.update_task(
            task_id, **fields
        )
        self.sync_pending()
        return task

    def change_status(
        self,
        task_id: str,
        status: Status | str,
    ) -> Task:
        return self.update_task(
            task_id,
            status=status,
        )

    def complete_task(
        self, task_id: str
    ) -> Task:
        return self.change_status(
            task_id,
            Status.DONE,
        )

    def delete_task(
        self, task_id: str
    ) -> Task:
        task = self.database.delete_task(task_id)
        self.sync_pending()
        return task

    def search_tasks(
        self,
        query: str,
        limit: int = 5,
        status: Status | str | None = None,
        priority: Priority | str | None = None,
    ) -> list[TaskSearchResult]:
        filters = TaskFilters(
            status=status,
            priority=priority,
        )

        self.sync_pending()

        candidates = self.vector_store.search(
            query,
            limit=limit,
            status=filters.status,
            priority=filters.priority,
        )

        results = []

        for candidate in candidates:
            task = self.database.get_task(
                candidate.task_id
            )

            if task is None:
                continue

            if (
                filters.status is not None
                and task.status != filters.status
            ):
                continue

            if (
                filters.priority is not None
                and task.priority != filters.priority
            ):
                continue

            results.append(
                TaskSearchResult(
                    task=task,
                    distance=candidate.distance,
                )
            )

        return results

    def apply_mutations(self, mutations: list[dict]) -> list[dict]:
        """Commit all SQLite mutations, then synchronize Chroma."""

        results = self.database.apply_mutations(mutations)

        try:
            self.sync_pending()
        except IndexSyncError as exc:
            raise BatchSyncError(results) from exc

        return results
