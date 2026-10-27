
"""Stage 3 CLI: interact with the AI task agent."""

from uuid import uuid4

from task_manager.config import get_settings
from task_manager.database import TaskDatabase
from task_manager.vector_store import TaskVectorStore
from task_manager.task_service import TaskService
from task_manager.graph import build_agent_graph


def main():
    settings = get_settings()

    db = TaskDatabase(settings.sqlite_path)
    index = TaskVectorStore(settings.chroma_path)
    service = TaskService(db, index)

    # Repair stale/missing index records at application startup.
    service.reconcile_index()

    graph = build_agent_graph(service)

    # Keep the same thread across messages, so clarification
    # replies refer to the previous unresolved request.
    thread_id = str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    print("\nAI Task Manager — Stage 3")
    print("Provider:", settings.llm_provider)
    print("Model:", settings.llm_model)
    print("Type 'exit' to quit.\n")

    while True:
        try:
            message = input("You: ").strip()

            if message.lower() in {"exit", "quit"}:
                break

            if not message:
                continue

            result = graph.invoke(
                {"user_request": message},
                config=config,
            )

            print("\nAssistant:")
            print(result["response"])
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        except Exception as exc:
            print(
                "\nApplication error:",
                str(exc),
            )
            print(
                "Check the database and pending index queue "
                "before retrying a mutation.\n"
            )


if __name__ == "__main__":
    main()
