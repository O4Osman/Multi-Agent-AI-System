"""Langfuse observability setup for the Learning Accelerator."""

import os

from langfuse import get_client


def _langfuse_configured() -> bool:
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    )


def get_langfuse_handler(session_id: str, user_id: str = "local"):
    """Create a Langfuse callback handler."""
    if not _langfuse_configured():
        return None

    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()

    except Exception as exc:
        print(f"[Observability] Failed to create handler: {exc}")
        return None


def get_langfuse_config(
        session_id: str,
        user_id: str = "local",
        extra_config: dict | None = None,
) -> dict:
    """Build LangGraph configuration with Langfuse tracing."""

    config = {
        "configurable": {
            "thread_id": session_id,
        },
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_user_id": user_id,
            "langfuse_tags": [
                "learning-accelerator",
                "local-inference",
            ],
            "langfuse_trace_name": "learning-accelerator",
        },
    }

    if extra_config:
        config.update(extra_config)

    handler = get_langfuse_handler(
        session_id=session_id,
        user_id=user_id,
    )

    if handler:
        config["callbacks"] = [handler]
        print(
            f"[Observability] Langfuse tracing enabled: "
            f"{os.getenv('LANGFUSE_BASE_URL')}"
        )
    else:
        print("[Observability] Running without Langfuse tracing.")

    return config


def flush_langfuse() -> None:
    """Flush pending Langfuse events."""

    if not _langfuse_configured():
        return

    try:
        get_client().flush()
        print("[Observability] Langfuse events flushed.")
    except Exception as exc:
        print(f"[Observability] Flush failed: {exc}")
