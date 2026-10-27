
"""Run the Stage 2 data layer without an LLM."""

from task_manager.config import get_settings
from task_manager.database import TaskDatabase
from task_manager.task_service import TaskService
from task_manager.vector_store import TaskVectorStore


def main():
    settings = get_settings()

    database = TaskDatabase(settings.sqlite_path)

    vector_store = TaskVectorStore(
        settings.chroma_path
    )

    service = TaskService(
        database,
        vector_store,
    )

    # Repair the index from existing SQLite records.
    service.reconcile_index()

    print("\n1. Creating tasks")

    task1 = service.create_task(
        title="Prepare Python assignment",
        description="Finish the AI task manager data layer",
        priority="High",
        due_date="2026-11-15",
    )

    task2 = service.create_task(
        title="Pay electricity bill",
        description="Complete monthly utility payment",
        priority="Medium",
    )

    print("Created:", task1.id, task1.title)
    print("Created:", task2.id, task2.title)

    print("\n2. Listing all tasks")

    for task in service.list_tasks():
        print(task.id, task.title, task.status.value)

    print("\n3. Updating a task")

    updated = service.update_task(
        task1.id,
        status="In progress",
        priority="High",
    )

    print(
        updated.id,
        updated.title,
        updated.status.value,
    )

    print("\n4. Semantic search")

    results = service.search_tasks(
        "utility invoice payment"
    )

    for result in results:
        print(
            result.task.id,
            result.task.title,
            "distance:",
            round(result.distance, 4),
        )

    print("\n5. Completing a task")

    completed = service.complete_task(task1.id)

    print(completed.id, completed.status.value)

    print("\n6. Clearing a due date")

    cleared = service.update_task(
        task1.id,
        due_date=None,
    )

    print(cleared.id, "due date:", cleared.due_date)

    print("\n7. Total task count")

    print(service.count_tasks())

    print("\n8. Pending index operations")

    print(database.pending_index_changes())

    print("\nDemo completed.")


if __name__ == "__main__":
    main()
