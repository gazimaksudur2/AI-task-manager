
"""Strict LLM-proposed actions. These schemas never execute an action."""

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import Priority, Status


class Operation(str, Enum):
    CREATE = "CREATE"
    LIST = "LIST"
    SEARCH = "SEARCH"
    UPDATE = "UPDATE"
    CHANGE_STATUS = "CHANGE_STATUS"
    COMPLETE = "COMPLETE"
    DELETE = "DELETE"
    HELP = "HELP"


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Operation

    # The task being modified. Distinct from its new title.
    target_id: str | None = None
    target_title: str | None = None
    target_description: str | None = None

    # New task values, or updated values.
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    status: Status | None = None
    priority: Priority | None = None

    # Keep dates as text until Python resolves relative expressions.
    due_date: str | None = None

    # Explicit clear operations; a missing/null field alone
    # must never accidentally delete existing information.
    clear_description: bool = False
    clear_due_date: bool = False

    # Query and read filters.
    query: str | None = None
    unfinished: bool = False

    @model_validator(mode="after")
    def validate_operation(self) -> Self:
        op = self.operation

        values = {
            "target_id": self.target_id,
            "target_title": self.target_title,
            "target_description": self.target_description,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "due_date": self.due_date,
            "query": self.query,
        }

        active = {
            key for key, value in values.items()
            if value is not None
        }

        if self.clear_description:
            active.add("clear_description")
        if self.clear_due_date:
            active.add("clear_due_date")
        if self.unfinished:
            active.add("unfinished")

        targets = {
            "target_id", "target_title", "target_description"
        }

        allowed = {
            Operation.CREATE: {
                "title", "description", "status",
                "priority", "due_date",
            },
            Operation.LIST: {
                "status", "priority", "unfinished",
            },
            Operation.SEARCH: {
                "query", "status", "priority", "unfinished",
            },
            Operation.UPDATE: targets | {
                "title", "description", "status",
                "priority", "due_date",
                "clear_description", "clear_due_date",
            },
            Operation.CHANGE_STATUS: targets | {"status"},
            Operation.COMPLETE: targets,
            Operation.DELETE: targets,
            Operation.HELP: set(),
        }

        extra = active - allowed[op]
        if extra:
            raise ValueError(
                f"{op.value} does not accept: {sorted(extra)}"
            )

        if op == Operation.CREATE:
            if not self.title or not self.title.strip():
                raise ValueError("CREATE requires a title")

        if op in {
            Operation.UPDATE,
            Operation.CHANGE_STATUS,
            Operation.COMPLETE,
            Operation.DELETE,
        }:
            if not (active & targets):
                raise ValueError(f"{op.value} requires a target")

        if op == Operation.UPDATE:
            editable = {
                "title", "description", "status",
                "priority", "due_date",
                "clear_description", "clear_due_date",
            }

            if not (active & editable):
                raise ValueError("UPDATE needs a field to change")

        if op == Operation.CHANGE_STATUS:
            if self.status is None:
                raise ValueError(
                    "CHANGE_STATUS requires a valid status"
                )

        if op == Operation.SEARCH:
            if not self.query or not self.query.strip():
                raise ValueError("SEARCH requires a query")

        for name in (
            "target_id", "target_title", "target_description",
            "query",
        ):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} cannot be blank")

        if self.title is not None and not self.title.strip():
            raise ValueError("Title cannot be blank")

        if self.due_date is not None and not self.due_date.strip():
            raise ValueError("Due date cannot be blank")

        if (
            op == Operation.UPDATE
            and self.description is not None
            and not self.description.strip()
        ):
            raise ValueError(
                "Use clear_description=true to clear a description"
            )

        if self.clear_description and self.description is not None:
            raise ValueError(
                "Cannot set and clear description together"
            )

        if self.clear_due_date and self.due_date is not None:
            raise ValueError(
                "Cannot set and clear due date together"
            )

        if self.unfinished and self.status == Status.DONE:
            raise ValueError(
                "Done and unfinished filters conflict"
            )

        return self


class ActionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[Action] = Field(
        min_length=1, max_length=12
    )
