
"""LLM adapter: natural language -> validated action plan only."""

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_models import ActionPlan
from .config import get_settings


SYSTEM_PROMPT = """
You are a task-management intent parser, not a task executor.

Return a structured ActionPlan containing every requested operation.
You may propose only:
CREATE, LIST, SEARCH, UPDATE, CHANGE_STATUS, COMPLETE,
DELETE, HELP.

Rules:
- Never execute tools, SQL or code.
- Never invent task IDs. Include target_id only when the
  user explicitly supplies an ID.
- Use target_title for a task's existing name or distinctive title words.
  Example: "demo task" or "task with demo in its title" can be
  target_title="demo". Do not invent an exact title not given by the user.
- Use target_description for a vague semantic reference with no title words.
- Never invent a matching task or silently drop an action if a task may
  not exist. The resolver will ask for clarification.
- For UPDATE, title means the NEW title.
- CREATE requires title.
- DELETE and other mutations require a target.
- "Move task to In progress" means CHANGE_STATUS.
- "Mark task done" means COMPLETE.
- "Show tasks related to X" means SEARCH.
- "Show all tasks" means LIST.
- "Unfinished" means unfinished=true.
- Priority must be Low, Medium or High.
- Status must be Todo, In progress or Done.
- due_date accepts ONLY a calendar date or a supported date phrase,
  such as "2026-10-28", "tomorrow", or "next Friday".
  Python will resolve relative dates. NEVER put a clock time in due_date.
- If the user specifies a clock time, keep it in description (e.g.
  "Time: 4:00 PM") while keeping due_date limited to the DATE.
  Preserve other description details; never discard the meeting time.
- Do not invent a timezone or guess an unsupported date.
- Clear a description only with clear_description=true.
- Clear a due date only with clear_due_date=true.
- Split multiple user requests into multiple actions.
- If instructions are not safely expressible as actions,
  produce a validation error rather than guessing.

The current Asia/Dhaka date is {current_date}.

OUTPUT FORMAT:
Return only a valid JSON object with a top-level "actions" array.
Do not include Markdown or explanations.
Use operation names exactly as defined.
Omit optional fields that do not apply.

Example response for creating a task:
{
  "actions": [
    {
      "operation": "CREATE",
      "title": "Finish research proposal",
      "priority": "High",
      "due_date": "tomorrow"
    }
  ]
}

Example: "I have a meeting tomorrow at 4 pm with the CEO of SpaceX":
{
  "actions": [
    {
      "operation": "CREATE",
      "title": "Meeting with SpaceX CEO",
      "due_date": "tomorrow",
      "description": "Time: 4:00 PM"
    }
  ]
}

"""


def build_chat_model():
    """Create only the selected provider's client."""

    settings = get_settings()

    if not settings.provider_key_configured:
        raise ValueError(
            f"No API key configured for {settings.llm_provider}"
        )

    model = settings.llm_model

    if settings.llm_provider == "groq":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key.get_secret_value(),
            temperature=0,
            max_retries=1,
        )

    if settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            api_key=settings.google_api_key.get_secret_value(),
            temperature=0,
            max_retries=1,
        )

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=settings.claude_api_key.get_secret_value(),
            temperature=0,
            max_retries=1,
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key.get_secret_value(),
            temperature=0,
            max_retries=1,
        )

    raise ValueError("Unsupported LLM provider")


class LLMPlanner:
    def __init__(self):
        model = build_chat_model()
        settings = get_settings()
        self.timezone = ZoneInfo(settings.app_timezone)

        # JSON-schema output is available for the selected
        # recommended models. Pydantic still validates locally.
        if settings.llm_provider == "groq":
            self.structured_model = model.with_structured_output(
                ActionPlan,
                method="json_mode",
            )

        elif settings.llm_provider == "anthropic":
            self.structured_model = model.with_structured_output(
                ActionPlan,
                method="json_schema",
            )
        elif settings.llm_provider == "google":
            self.structured_model = model.with_structured_output(
                ActionPlan,
                method="json_schema",
            )

        elif settings.llm_provider == "openai":
            self.structured_model = model.with_structured_output(
                ActionPlan,
                method="function_calling",
            )

    def __call__(self, request: str) -> ActionPlan:
        current_date = datetime.now(
            self.timezone
        ).date().isoformat()

        messages = [
            SystemMessage(
                content=SYSTEM_PROMPT.replace(
                    "{current_date}", current_date
                )
            ),
            HumanMessage(content=request),
        ]

        raw = self.structured_model.invoke(messages)

        # Never trust provider-level validation alone, including target IDs.
        plan = ActionPlan.model_validate(raw)
        for action in plan.actions:
            if action.target_id is not None:
                supplied = re.search(
                    rf"(?<![A-Za-z0-9_-]){re.escape(action.target_id)}"
                    rf"(?![A-Za-z0-9_-])",
                    request,
                    flags=re.IGNORECASE,
                )
                if supplied is None:
                    raise ValueError(
                        "The model proposed a task ID not supplied by the user. "
                        "Please identify the task by its ID or title."
                    )
        return plan
