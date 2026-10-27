
"""Validated task values and public data shapes."""

import re
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Status(str, Enum):
    TODO = "Todo"
    IN_PROGRESS = "In progress"
    DONE = "Done"


class Priority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


def validate_due_date(value):
    if value is None or (
        isinstance(value, date) and not isinstance(value, datetime)
    ):
        return value

    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", value
    ):
        raise ValueError("Due date must be YYYY-MM-DD or None")

    return date.fromisoformat(value)


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    status: Status = Status.TODO
    priority: Priority = Priority.MEDIUM
    due_date: date | None = None

    @field_validator("title")
    @classmethod
    def clean_title(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Title cannot be blank")
        return value

    @field_validator("description")
    @classmethod
    def clean_description(cls, value):
        return value.strip() or None if value is not None else None

    @field_validator("due_date", mode="before")
    @classmethod
    def check_date(cls, value):
        return validate_due_date(value)


class TaskUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    status: Status | None = None
    priority: Priority | None = None
    due_date: date | None = None

    @field_validator("title")
    @classmethod
    def clean_title(cls, value):
        if value is None:
            return value

        value = value.strip()
        if not value:
            raise ValueError("Title cannot be blank")
        return value

    @field_validator("description")
    @classmethod
    def clean_description(cls, value):
        return value.strip() or None if value is not None else None

    @field_validator("due_date", mode="before")
    @classmethod
    def check_date(cls, value):
        return validate_due_date(value)


class TaskFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Status | None = None
    priority: Priority | None = None


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str | None
    status: Status
    priority: Priority
    due_date: date | None
    created_at: datetime
    updated_at: datetime
    revision: int


def parse_task_id(task_id: str) -> int:
    if not isinstance(task_id, str) or not re.fullmatch(
        r"TSK-\d{6,}", task_id
    ):
        raise ValueError("Task ID must look like TSK-000001")

    number = int(task_id[4:])
    if number < 1:
        raise ValueError("Task ID must be positive")

    return number


def format_task_id(number: int) -> str:
    return f"TSK-{number:06d}"
