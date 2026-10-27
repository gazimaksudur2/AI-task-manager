"""Deterministic clarification controls and conservative title-fragment matching.

No LangGraph, LLM, database, or embedding dependencies; safe to test offline.
"""

import re
from collections.abc import Iterable
from typing import TypeVar

TaskT = TypeVar("TaskT")

# A new explicit instruction supersedes an unresolved clarification. Freeform
# replies that do NOT look like commands still get the existing clarification.
_NEW_COMMAND = re.compile(
    r"^(?:(?:please|can you|could you|would you|i want to|i need to)\s+)*"
    r"(?:show|find|list|search|display|create|add|make|schedule|"
    r"move|mark|complete|finish|delete|remove|update|edit|rename|"
    r"change|set|clear|help|what|which|how many|remind)\b",
    re.IGNORECASE,
)
_CANCEL = re.compile(
    r"^(?:cancel|cancel request|cancel that|never\s?mind|"
    r"forget it|stop|start over|skip|abort)(?:[.!])?$",
    re.IGNORECASE,
)
_SIMPLE_LIST = re.compile(
    r"^(?:(?:please|can you|could you)\s+)?"
    r"(?:show|find|list|display)\s+"
    r"(?:(?:me|the|all|my|tasks?)\s*)+[.!?]?$",
    re.IGNORECASE,
)
# Includes short commands like 'find all' and 'show all'. Do not intercept
# filtered queries ('show high priority tasks', 'find documentation').
_GENERIC_TITLE_WORDS = frozenset({
    "a", "an", "the", "task", "tasks", "that", "which", "have",
    "has", "having", "in", "its", "on", "of", "with", "called",
    "named", "title", "my", "one", "to", "for", "this", "please",
    "about", "referring", "it", "is", "there", "by",
})


def is_cancel_request(message: str) -> bool:
    return bool(_CANCEL.fullmatch(message.strip()))


def is_new_instruction(message: str) -> bool:
    text = message.strip()
    # Users also naturally express task creation without an imperative verb.
    return bool(
        _NEW_COMMAND.match(text)
        or re.match(
            r"^(?:i\s+have\s+(?:a|an|some|to)\s+|"
            r"i\s+(?:need|want)\s+to\s+|"
            r"there(?:'s| is)\s+a\s+)",
            text,
            re.IGNORECASE,
        )
    )


def is_simple_list_request(message: str) -> bool:
    text = message.strip()
    return bool(_SIMPLE_LIST.fullmatch(text)) and bool(re.search(r"\b(?:all|tasks?)\b", text, re.I))


def partial_title_matches(fragment: str, tasks: Iterable[TaskT]) -> list[TaskT]:
    """Return *all* literal-word title matches, never a speculative best match.

    Example: 'the demo task' matches 'Prepare my demo'. If two titles
    contain 'demo', BOTH are returned so the graph must ask which task.
    Only title words are considered; semantic matches still require consent.
    """
    query_words = {
        word for word in re.findall(r"[\w]+", fragment.casefold())
        if word not in _GENERIC_TITLE_WORDS and len(word) >= 3
    }
    if not query_words:
        return []
    return [
        task for task in tasks
        if query_words.issubset(set(re.findall(r"[\w]+", task.title.casefold())))
    ]
