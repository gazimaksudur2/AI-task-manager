
"""LangGraph task agent: plan, validate, clarify, execute, respond."""

import re
from copy import deepcopy
from datetime import date, datetime, timedelta
from typing import Callable, TypedDict
from zoneinfo import ZoneInfo

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .agent import LLMPlanner
from .agent_models import ActionPlan, Operation
from .models import Status
from .intent_resolution import (
    is_cancel_request, is_new_instruction, is_simple_list_request,
    is_unqualified_delete_request, partial_title_matches,
)
from .task_service import BatchSyncError, TaskService

import logging

logger = logging.getLogger(__name__)

MUTATIONS = {
    Operation.CREATE,
    Operation.UPDATE,
    Operation.CHANGE_STATUS,
    Operation.COMPLETE,
    Operation.DELETE,
}

TARGET_MUTATIONS = MUTATIONS - {Operation.CREATE}


class AgentState(TypedDict, total=False):
    user_request: str
    submission_id: str
    plan: dict | None
    resolved_targets: dict[str, str]
    target_versions: dict[str, int]
    validation_errors: list[str]
    pending_clarification: dict | None
    execution_results: list[dict]
    index_pending: bool
    response: str
    route: str


class DateResolutionError(ValueError):
    pass


def dhaka_today() -> date:
    return datetime.now(
        ZoneInfo("Asia/Dhaka")
    ).date()


def resolve_due_date(
    expression: str,
    today: date | None = None,
) -> str:
    """Resolve supported dates using Python, never the model's clock."""
    if today is None:
        today = dhaka_today()

    text = expression.strip().lower()

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise DateResolutionError(
                "Invalid calendar date"
            ) from exc

    offsets = {
        "today": 0,
        "tomorrow": 1,
        "day after tomorrow": 2,
    }

    if text in offsets:
        return (
            today + timedelta(days=offsets[text])
        ).isoformat()

    match = re.fullmatch(r"in (\d{1,4}) days?", text)
    if match:
        days = int(match.group(1))
        return (
            today + timedelta(days=days)
        ).isoformat()

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    match = re.fullmatch(
        r"(this|next) (monday|tuesday|wednesday|"
        r"thursday|friday|saturday|sunday)",
        text,
    )

    if match:
        qualifier, weekday = match.groups()
        target = weekdays[weekday]

        days_ahead = (
            target - today.weekday()
        ) % 7

        if qualifier == "next":
            days_ahead = 7 - today.weekday() + target

        return (
            today + timedelta(days=days_ahead)
        ).isoformat()

    raise DateResolutionError(
        f"Cannot interpret '{expression}' confidently. "
        "Please give an exact YYYY-MM-DD date."
    )


def split_due_date_time(expression: str) -> tuple[str, str | None]:
    """Separate a simple clock time from a supported due-date expression.

    The database deliberately has no time column: a clock time is saved
    as human-readable description text, not silently discarded.
    """
    text = expression.strip()
    match = re.fullmatch(
        r"(?P<day>.+?)\s+(?:at\s+)?"
        r"(?P<clock>(?:0?[1-9]|1[0-2])(?::[0-5]\d)?\s*[ap]\.?m\.?"
        r"|(?:[01]?\d|2[0-3]):[0-5]\d|noon|midnight)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return text, None

    raw_time = match.group("clock").lower().strip()
    if raw_time in {"noon", "midnight"}:
        hour = 12 if raw_time == "noon" else 0
        minute = 0
    else:
        twelve_hour = re.fullmatch(
            r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?",
            raw_time,
        )
        if twelve_hour:
            hour = int(twelve_hour.group(1)) % 12
            if twelve_hour.group(3) == "p":
                hour += 12
            minute = int(twelve_hour.group(2) or 0)
        else:
            hour_text, minute_text = raw_time.split(":")
            hour, minute = int(hour_text), int(minute_text)

    period = "AM" if hour < 12 else "PM"
    display_hour = (hour % 12) or 12
    return match.group("day"), f"{display_hour}:{minute:02d} {period}"


def append_meeting_time(description: str | None, clock: str) -> str:
    """Keep existing notes and avoid adding an already-mentioned time."""
    current = (description or "").strip()
    hh, rest = clock.split(":", 1)
    mm, meridiem = rest.split(" ", 1)
    # Accept both '4 pm' and '4:00 PM' as existing descriptions.
    time_pattern = (
        rf"(?<!\d){int(hh)}(?::{mm})?\s*"
        rf"{meridiem[0]}\.?m\.?(?![a-z])"
    )
    if re.search(time_pattern, current, flags=re.IGNORECASE):
        return current
    note = f"Time: {clock}"
    return f"{current}\n{note}" if current else note


ORDINALS = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
}


def _task_options(
    candidates: list[dict],
    question: str = "Which task do you mean?",
) -> str:
    if not candidates:
        return (
            "No matching task was found. "
            "Reply with an existing task ID or exact title."
        )

    lines = [
        f"{i}. {item['id']} — {item['title']}"
        for i, item in enumerate(candidates, start=1)
    ]

    return (
        question + "\n"
        + "\n".join(lines)
        + "\n\n\n Reply with its ID or option number."
    )


def _clarification_selection(
    text: str,
    pending: dict,
    service: TaskService,
) -> str | None:
    """Interpret clarification deterministically, without the LLM."""

    value = text.strip()
    candidates = pending.get("candidates", [])

    if re.fullmatch(r"TSK-\d{6,}", value, re.I):
        task_id = value.upper()
        return (
            task_id
            if service.get_task(task_id) is not None
            else None
        )

    match = re.fullmatch(
        r"(?:the\s+)?(first|second|third|fourth|fifth)"
        r"(?:\s+one)?",
        value,
        re.I,
    )

    if match:
        position = ORDINALS[match.group(1).lower()]
        if position < len(candidates):
            return candidates[position]["id"]

    match = re.fullmatch(
        r"(?:option\s+)?(\d+)",
        value,
        re.I,
    )

    if match:
        position = int(match.group(1)) - 1
        if 0 <= position < len(candidates):
            return candidates[position]["id"]

    # A manually provided exact title is also acceptable.
    matches = [
        task for task in service.all_tasks()
        if task.title.strip().casefold() == value.casefold()
    ]

    if len(matches) == 1:
        return matches[0].id

    return None


def _mutation_fields(action) -> dict:
    """Convert a validated action into allowlisted service fields."""
    fields = {}

    for name in (
        "title", "description", "status",
        "priority", "due_date",
    ):
        value = getattr(action, name)
        if value is not None:
            fields[name] = (
                value.value if hasattr(value, "value") else value
            )

    if action.clear_description:
        fields["description"] = None

    if action.clear_due_date:
        fields["due_date"] = None

    return fields


def build_agent_graph(
    service: TaskService,
    planner: Callable[[str], ActionPlan] | None = None,
    checkpointer=None,
):
    """Build a testable graph. Inject a fake planner for unit tests."""
    if planner is None:
        planner = LLMPlanner()

    if checkpointer is None:
        checkpointer = InMemorySaver()

    def understand(state: AgentState) -> dict:
        request = state["user_request"].strip()

        if not request:
            return {
                "validation_errors": ["Request cannot be blank."],
                "route": "respond",
                "response": "",
            }

        pending = state.get("pending_clarification")

        # The user may cancel or issue an entirely new instruction instead of
        # answering the previous question. Do not trap the chat in clarify.
        if pending and is_cancel_request(request):
            return {
                "plan": None,
                "pending_clarification": None,
                "resolved_targets": {},
                "target_versions": {},
                "validation_errors": [],
                "execution_results": [{
                    "kind": "notice",
                    "message": "Canceled the previous request. No task changes were made.",
                }],
                "route": "respond",
                "response": "",
            }

        if pending and is_new_instruction(request):
            # The following fresh plan replaces all old targets and state.
            pending = None

        if pending:
            if pending["kind"] == "delete_selection":
                selected = _clarification_selection(
                    request, pending, service
                )

                candidate = next(
                    (
                        item for item in pending["candidates"]
                        if item["id"] == selected
                    ),
                    None,
                )

                if candidate is None:
                    return {
                        "route": "clarify",
                        "response": "",
                    }

                return {
                    "plan": {
                        "actions": [{
                            "operation": "DELETE",
                            "target_id": selected,
                        }],
                    },
                    "resolved_targets": {"0": selected},
                    "target_versions": {
                        "0": candidate["revision"],
                    },
                    "pending_clarification": None,
                    "validation_errors": [],
                    "execution_results": [],
                    "route": "validate",
                    "response": "",
                }

            plan = deepcopy(state["plan"])
            index = pending["index"]

            if pending["kind"] == "date":
                try:
                    resolved = resolve_due_date(request)
                except DateResolutionError:
                    return {
                        "route": "clarify",
                        "response": "",
                    }

                plan["actions"][index]["due_date"] = resolved

                return {
                    "plan": plan,
                    "pending_clarification": None,
                    "route": "validate",
                    "response": "",
                }

            selected = _clarification_selection(
                request, pending, service
            )

            if selected is None:
                return {
                    "route": "clarify",
                    "response": "",
                }

            task = service.get_task(selected)

            targets = dict(
                state.get("resolved_targets") or {}
            )
            versions = dict(
                state.get("target_versions") or {}
            )

            targets[str(index)] = task.id
            versions[str(index)] = task.revision

            return {
                "resolved_targets": targets,
                "target_versions": versions,
                "pending_clarification": None,
                "route": "validate",
                "response": "",
            }
        
        try:
            # Unfiltered task-list commands are deterministic and do not
            # require an LLM call, even after a pending clarification.
            if is_simple_list_request(request):
                plan = ActionPlan.model_validate({
                    "actions": [{"operation": "LIST"}],
                })
            elif is_unqualified_delete_request(request):
                tasks = service.all_tasks()
                if not tasks:
                    return {
                        "plan": None,
                        "pending_clarification": None,
                        "resolved_targets": {},
                        "target_versions": {},
                        "validation_errors": [],
                        "execution_results": [{
                            "kind": "notice",
                            "message": "There are no tasks to delete.",
                        }],
                        "route": "respond",
                        "response": "",
                    }

                candidates = [
                    {
                        "id": task.id,
                        "title": task.title,
                        "revision": task.revision,
                    }
                    for task in tasks
                ]
                pending = {
                    "kind": "delete_selection",
                    "index": 0,
                    "candidates": candidates,
                    "prompt": _task_options(
                        candidates,
                        question="Which task should I delete?",
                    ),
                }
                return {
                    "plan": None,
                    "resolved_targets": {},
                    "target_versions": {},
                    "pending_clarification": pending,
                    "validation_errors": [],
                    "execution_results": [],
                    "route": "clarify",
                    "response": "",
                }
            else:
                proposed = planner(request)
                plan = ActionPlan.model_validate(proposed)

        except Exception as exc:
            logger.exception(
                "LLM planning failed: %s",
                type(exc).__name__,
            )

            return {
                "plan": None,
                "resolved_targets": {},
                "target_versions": {},
                "pending_clarification": None,
                "execution_results": [],
                "validation_errors": [
                    "Could not produce a valid action plan. "
                    "No changes were made. "
                    "Check the terminal error details."
                ],
                "route": "respond",
                "response": "",
            }

        # IMPORTANT:
        # This return must be OUTSIDE the except block.
        return {
            "plan": plan.model_dump(mode="json"),
            "resolved_targets": {},
            "target_versions": {},
            "validation_errors": [],
            "pending_clarification": None,
            "execution_results": [],
            "index_pending": False,
            "route": "validate",
            "response": "",
        }

    def validate_and_resolve(state: AgentState) -> dict:
        try:
            plan_data = deepcopy(state["plan"])
            plan = ActionPlan.model_validate(plan_data)

            targets = dict(
                state.get("resolved_targets") or {}
            )
            versions = dict(
                state.get("target_versions") or {}
            )

            for index, action in enumerate(plan.actions):
                key = str(index)

                if action.operation not in TARGET_MUTATIONS:
                    continue

                if key in targets:
                    task = service.get_task(targets[key])

                    if (
                        task is None
                        or task.revision != versions[key]
                    ):
                        raise ValueError(
                            "A previously selected task was "
                            "deleted or changed. Start the "
                            "request again."
                        )
                    continue

                candidates = []
                task = None
                reason = ""

                if action.target_id:
                    # IDs from the model are verified against SQLite.
                    task = service.get_task(action.target_id)

                    if task is None:
                        reason = (
                            f"Task {action.target_id} does not exist."
                        )

                elif action.target_title:
                    exact = [
                        task for task in service.all_tasks()
                        if task.title.strip().casefold()
                        == action.target_title.strip().casefold()
                    ]

                    if len(exact) == 1:
                        task = exact[0]
                    elif len(exact) > 1:
                        candidates = [
                            {"id": t.id, "title": t.title}
                            for t in exact
                        ]
                        reason = "Multiple tasks have this title."
                    else:
                        # A literal, unique title fragment is safe to resolve
                        # automatically; ambiguous fragments never mutate.
                        fragments = partial_title_matches(
                            action.target_title, service.all_tasks()
                        )
                        if len(fragments) == 1:
                            task = fragments[0]
                        elif len(fragments) > 1:
                            candidates = [
                                {"id": t.id, "title": t.title}
                                for t in fragments[:5]
                            ]
                            reason = "Several task titles match that phrase."
                        else:
                            reason = (
                                f"No task title matches '{action.target_title}'. "
                                "The following semantic suggestions may be unrelated."
                            )

                if task is None and not candidates:
                    # No semantic fallback for a supplied nonexistent ID.
                    if not action.target_id:
                        phrase = (
                            action.target_title
                            or action.target_description
                        )

                        matches = service.search_tasks(
                            phrase, limit=5
                        )

                        candidates = [
                            {
                                "id": result.task.id,
                                "title": result.task.title,
                            }
                            for result in matches
                        ]

                        # Never automatically mutate a single
                        # semantic-search candidate.
                        if candidates and not reason:
                            reason = (
                                "Please confirm the semantic match."
                            )

                if task is not None:
                    targets[key] = task.id
                    versions[key] = task.revision
                    continue

                pending = {
                    "kind": "target",
                    "index": index,
                    "candidates": candidates,
                    "prompt": (
                        f"Action {index + 1}: {reason}\n"
                        + _task_options(candidates)
                    ),
                }

                return {
                    "plan": plan_data,
                    "resolved_targets": targets,
                    "target_versions": versions,
                    "pending_clarification": pending,
                    "route": "clarify",
                }

            # Only after all mutation targets are resolved do we normalize
            # dates and preserve optional clock times. This avoids losing
            # time details while waiting for an ambiguous-target reply.
            for index, action in enumerate(plan.actions):
                if action.due_date is None:
                    continue

                date_text, clock = split_due_date_time(action.due_date)
                try:
                    resolved_date = resolve_due_date(date_text)
                except DateResolutionError:
                    pending = {
                        "kind": "date",
                        "index": index,
                        "candidates": [],
                        "prompt": (
                            f"Action {index + 1}: "
                            f"I cannot confidently interpret "
                            f"'{action.due_date}'. "
                            "Please provide YYYY-MM-DD."
                        ),
                    }
                    return {
                        "pending_clarification": pending,
                        "plan": plan_data,
                        "resolved_targets": targets,
                        "target_versions": versions,
                        "route": "clarify",
                    }

                plan_data["actions"][index]["due_date"] = resolved_date

                if clock is not None:
                    if action.clear_description:
                        raise ValueError(
                            "Cannot clear the description and save a meeting "
                            "time in it in the same action."
                        )
                    original_description = action.description
                    if (action.operation == Operation.UPDATE
                            and original_description is None):
                        target = service.get_task(targets[str(index)])
                        original_description = target.description
                    plan_data["actions"][index]["description"] = (
                        append_meeting_time(original_description, clock)
                    )

            return {
                "plan": plan_data,
                "resolved_targets": targets,
                "target_versions": versions,
                "pending_clarification": None,
                "validation_errors": [],
                "route": "execute",
            }

        except Exception as exc:
            return {
                "validation_errors": [
                    f"Validation or target resolution failed: {exc}"
                ],
                "pending_clarification": None,
                "route": "respond",
            }

    def clarify(state: AgentState) -> dict:
        pending = state["pending_clarification"]
        return {
            "response": pending["prompt"],
        }

    def execute(state: AgentState) -> dict:
        plan = ActionPlan.model_validate(state["plan"])
        targets = state.get("resolved_targets") or {}
        versions = state.get("target_versions") or {}

        mutations = []
        mutation_indices = []

        for index, action in enumerate(plan.actions):
            if action.operation not in MUTATIONS:
                continue

            item = {
                "operation": action.operation.value,
                "fields": _mutation_fields(action),
            }

            if action.operation != Operation.CREATE:
                key = str(index)
                item["task_id"] = targets[key]
                item["expected_revision"] = versions[key]

            # The specialized operations require only their
            # narrowly allowed fields.
            if action.operation == Operation.CHANGE_STATUS:
                item["fields"] = {
                    "status": action.status.value
                }
            elif action.operation in {
                Operation.COMPLETE, Operation.DELETE
            }:
                item["fields"] = {}

            mutations.append(item)
            mutation_indices.append(index)

        results_by_index = {}
        index_pending = False

        if mutations:
            try:
                saved = service.apply_mutations(
                    mutations,
                    request_id=state.get("submission_id"),
                )

            except BatchSyncError as exc:
                # The SQLite commit was confirmed, but the
                # Chroma index still needs synchronization.
                saved = exc.results
                index_pending = True

            except Exception:
                logger.exception("Task mutation batch failed")
                return {
                    "execution_results": [],
                    "validation_errors": [
                        "Batch execution failed without a confirmed result. "
                        "Check SQLite before retrying; no success was confirmed."
                    ],
                    "route": "respond",
                }

            for index, result in zip(
                mutation_indices, saved
            ):
                results_by_index[index] = {
                    "kind": "mutation",
                    "operation": result["operation"],
                    "task": result["task"].model_dump(
                        mode="json"
                    ),
                }

        # Reads are deliberately performed after the mutation batch.
        # SQLite mutations are atomic; reads do not alter the database.
        for index, action in enumerate(plan.actions):
            if action.operation in MUTATIONS:
                continue

            try:
                if action.operation == Operation.HELP:
                    results_by_index[index] = {
                        "kind": "help",
                    }
                    continue

                if action.operation == Operation.LIST:
                    rows = service.list_tasks(
                        status=action.status,
                        priority=action.priority,
                    )

                    if action.unfinished:
                        rows = [
                            row for row in rows
                            if row.status != Status.DONE
                        ]

                    results_by_index[index] = {
                        "kind": "list",
                        "tasks": [
                            row.model_dump(mode="json")
                            for row in rows
                        ],
                    }
                    continue

                if index_pending:
                    raise RuntimeError(
                        "Search index synchronization is pending"
                    )

                # Retrieve from Chroma and fetch authoritative
                # records through the existing task service.
                # Fetch all candidates before applying 'unfinished'
                # so a small top-k set cannot hide Todo tasks.
                limit = max(
                    1, service.count_tasks()
                )

                matches = service.search_tasks(
                    action.query,
                    limit=limit,
                    status=action.status,
                    priority=action.priority,
                )

                if action.unfinished:
                    matches = [
                        result for result in matches
                        if result.task.status != Status.DONE
                    ]

                results_by_index[index] = {
                    "kind": "search",
                    "tasks": [
                        result.task.model_dump(mode="json")
                        for result in matches
                    ],
                }

            except Exception as exc:
                results_by_index[index] = {
                    "kind": "read_error",
                    "message": str(exc),
                }

        return {
            "execution_results": [
                results_by_index[i]
                for i in sorted(results_by_index)
            ],
            "index_pending": index_pending,
            "validation_errors": [],
            "route": "respond",
        }

    def respond(state: AgentState) -> dict:
        errors = state.get("validation_errors") or []

        if errors:
            return {"response": "\n".join(errors)}

        messages = []

        for result in state.get("execution_results") or []:
            kind = result["kind"]

            if kind == "notice":
                messages.append(result["message"])

            elif kind == "mutation":
                task = result["task"]
                operation = result["operation"]

                verbs = {
                    "CREATE": "Created",
                    "UPDATE": "Updated",
                    "CHANGE_STATUS": "Changed status",
                    "COMPLETE": "Completed",
                    "DELETE": "Deleted",
                }

                messages.append(
                    f"{verbs[operation]} {task['id']} — "
                    f"{task['title']} "
                    f"({task['status']}, {task['priority']})."
                )

            elif kind in {"list", "search"}:
                tasks = result["tasks"]

                if not tasks:
                    label = (
                        "semantic candidates"
                        if kind == "search"
                        else "tasks"
                    )
                    messages.append(
                        f"No {label} found for the requested filters."
                    )
                    continue

                heading = (
                    "Ranked semantic candidates"
                    if kind == "search"
                    else "Task list"
                )

                lines = [
                    f"- {task['id']} — {task['title']} "
                    f"[{task['status']}, {task['priority']}]"
                    for task in tasks
                ]

                messages.append(
                    f"{heading} ({len(tasks)}):\n"
                    + "\n".join(lines)
                )

            elif kind == "help":
                messages.append(
                    "You can create, list, search, update, "
                    "complete and delete tasks; change their "
                    "status or priority; and set or clear dates."
                )

            elif kind == "read_error":
                messages.append(
                    "A read operation failed: "
                    + result["message"]
                )

        if state.get("index_pending"):
            messages.append(
                "SQLite changes were committed, but Chroma "
                "synchronization is pending. Run "
                "service.reconcile_index() to repair search."
            )

        return {
            "response": "\n\n".join(messages)
            if messages else "No operations were executed."
        }

    graph = StateGraph(AgentState)

    graph.add_node("understand", understand)
    graph.add_node(
        "validate_and_resolve", validate_and_resolve
    )
    graph.add_node("clarify", clarify)
    graph.add_node("execute", execute)
    graph.add_node("respond", respond)

    graph.add_edge(START, "understand")

    graph.add_conditional_edges(
        "understand",
        lambda state: state["route"],
        {
            "validate": "validate_and_resolve",
            "clarify": "clarify",
            "respond": "respond",
        },
    )

    graph.add_conditional_edges(
        "validate_and_resolve",
        lambda state: state["route"],
        {
            "execute": "execute",
            "clarify": "clarify",
            "respond": "respond",
        },
    )

    graph.add_edge("execute", "respond")
    graph.add_edge("clarify", END)
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=checkpointer)
