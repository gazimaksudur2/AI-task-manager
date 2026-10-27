
"""ChromaDB is a disposable index; SQLite task records are authoritative."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from .models import Priority, Status, Task, TaskFilters


@dataclass(frozen=True)
class SearchCandidate:
    task_id: str
    distance: float


class TaskVectorStore:
    def __init__(
        self,
        path: str | Path,
        embedding_function=None,
        max_distance: float | None = None,
    ):
        if max_distance is not None and max_distance < 0:
            raise ValueError("max_distance must not be negative")
        self.max_distance = max_distance
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(path)
        )

        kwargs = (
            {"embedding_function": embedding_function}
            if embedding_function is not None
            else {}
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="task_search",
                **kwargs,
            )
        )

    @staticmethod
    def task_document(task: Task) -> str:
        return (
            f"{task.title}\n"
            f"{task.description or ''}"
        ).strip()

    @staticmethod
    def task_metadata(task: Task) -> dict[str, Any]:
        metadata = {
            "task_id": task.id,
            "status": task.status.value,
            "priority": task.priority.value,
            "revision": task.revision,
        }

        if task.due_date is not None:
            metadata["due_date"] = (
                task.due_date.isoformat()
            )

        return metadata

    def upsert_task(self, task: Task) -> None:
        """Insert or update a task without retaining obsolete metadata."""

        new_metadata = self.task_metadata(task)

        existing = self.collection.get(
            ids=[task.id],
            include=["metadatas"],
        )

        if existing["ids"]:
            old_metadata = existing["metadatas"][0] or {}

            # A regular upsert may retain metadata keys
            # that are missing from the new dictionary.
            removed_keys = (
                set(old_metadata) - set(new_metadata)
            )

            if removed_keys:
                self.collection.delete(ids=[task.id])

        self.collection.upsert(
            ids=[task.id],
            documents=[self.task_document(task)],
            metadatas=[new_metadata],
        )

    def remove_task(self, task_id: str) -> None:
        self.collection.delete(ids=[task_id])

    def search(
        self,
        query: str,
        limit: int = 5,
        status: Status | str | None = None,
        priority: Priority | str | None = None,
    ) -> list[SearchCandidate]:
        if (
            not isinstance(query, str)
            or not query.strip()
        ):
            raise ValueError(
                "Search query must not be blank"
            )

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
        ):
            raise ValueError(
                "Search limit must be a positive integer"
            )

        filters = TaskFilters(
            status=status,
            priority=priority,
        )

        conditions = []

        if filters.status is not None:
            conditions.append({
                "status": filters.status.value
            })

        if filters.priority is not None:
            conditions.append({
                "priority": filters.priority.value
            })

        where = (
            conditions[0]
            if len(conditions) == 1
            else (
                {"$and": conditions}
                if conditions else None
            )
        )

        count = self.collection.count()

        if count == 0:
            return []

        result = self.collection.query(
            query_texts=[query.strip()],
            n_results=min(limit, count),
            where=where,
            include=["distances"],
        )

        ids = result["ids"][0]
        distances = result["distances"][0]

        return [
            SearchCandidate(
                task_id=task_id,
                distance=float(distance),
            )
            for task_id, distance
            in zip(ids, distances)
            if self.max_distance is None or distance <= self.max_distance
        ]

    def reconcile(
        self,
        tasks: list[Task],
    ) -> tuple[int, int]:
        """Repair missing/outdated documents and delete orphaned entries."""

        records = self.collection.get(
            include=["documents", "metadatas"]
        )

        existing = {
            task_id: (document, metadata)
            for task_id, document, metadata in zip(
                records["ids"],
                records["documents"],
                records["metadatas"],
            )
        }

        authoritative = {
            task.id: task
            for task in tasks
        }

        upserts = 0

        for task in tasks:
            expected = (
                self.task_document(task),
                self.task_metadata(task),
            )

            if existing.get(task.id) != expected:
                self.upsert_task(task)
                upserts += 1

        deleted = 0

        for old_id in (
            existing.keys() - authoritative.keys()
        ):
            self.remove_task(old_id)
            deleted += 1

        return upserts, deleted
