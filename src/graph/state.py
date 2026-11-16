from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class Topic:
    """A single topic within the study roadmap."""
    title: str
    description: str
    estimated_minutes: int
    prerequisites: list[str] = field(default_factory=list)
    # pending -> in_progress -> completed | needs_review
    status: str = 'pending'

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Topic":
        return cls(
            title=data["title"],
            description=data["description"],
            estimated_minutes=data["estimated_minutes"],
            prerequisites=data.get("prerequisites", []),
            status=data.get("status", "pending"),
        )


@dataclass
class StudyRoadmap:
    """The full study plan produced by the Curriculum Planner."""
    goal: str
    total_weeks: int
    topics: list[Topic]
    weekly_hours: int = 5

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "goal": self.goal,
            "total_weeks": self.total_weeks,
            "weekly_hours": self.weekly_hours,
            "topics": [t.to_dict() for t in self.topics],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StudyRoadmap":
        """Reconstruct from a plain dict."""
        return cls(
            goal=data["goal"],
            total_weeks=data["total_weeks"],
            weekly_hours=data.get("weekly_hours", 5),
            topics=[Topic.from_dict(t) for t in data.get("topics", [])],
        )

    def completed_count(self) -> int:
        """How many topics have been completed."""
        return sum(1 for t in self.topics if t.status == "completed")

    def is_complete(self) -> bool:
        return all(t.status in ("completed", "needs_review") for t in self.topics)


@dataclass
class QuizQuestion:
    """
    One question within a quiz, with the user's answer and grading.

    The Quiz Generator creates these.
    The Progress Coach reads them to identify weak areas.
    """

    question: str
    expected_answer: str

    # Filled in after the user answers
    user_answer: str = ""
    correct: bool = False
    feedback: str = ""  # One sentence of specific feedback from grader
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QuizResult:
    """
    The complete result of one quiz session on a single topic.

    Stored in AgentState.quiz_results, one entry per topic per session.
    The Progress Coach reads this to decide what to do next.
    """
    topic: str
    questions: list[QuizQuestion]
    score: float  # Average score across all questions (0.0 - 1.0)
    weak_areas: list[str]  # Concepts the student got wrong missed
    timestamp: str = ""  # ISO format UTC timestamp

    @classmethod
    def from_dict(cls, data: dict) -> "QuizResult":
        """
        Reconstruct from a plain dict.

        Called when LangGraph deserializes quiz_results from a SQLite
        checkpoint as raw dicts (msgpack round-trip). This happens when resuming a crashed or interrupted session.
        """
        return cls(
            topic=data.get("topic", ""),
            questions=[],  # Questions not needed for coaching logic
            score=float(data.get("score", 0.0)),
            weak_areas=data.get("weak_areas", []),
            timestamp=data.get("timestamp", ""),
        )

    def passed(self) -> bool:
        """ A score of 0.5 or above is considered as pass."""
        return self.score >= 0.5

    def strong_pass(self) -> bool:
        """A score of 0.75 or above, ready to move to next topic."""
        return self.score >= 0.75


# ─────────────────────────────────────────────────────────────────────────────
# The main state class
#
# AgentState is the TypedDict that LangGraph manages.
# Every node in the graph receives the full state and returns
# a partial update (only the keys it changed).
#
# Why TypedDict and not a regular class?
#   LangGraph requires dict-compatible objects. TypedDict gives us
#   type safety while staying dict-compatible.
#
# Why inherit from TypedDict directly instead of using a subclass trick?
#   LangGraph 1.x works cleanly with TypedDict. No need for workarounds.
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The shared state for the Learning Accelerator graph.

    Partial updates: when a node returns {"approved": True}, LangGraph
    merges that into the existing state. It does NOT replace the whole dict.
    Nodes only return the keys they changed.

    The one exception is `messages`: it uses the add_messages reducer,
    which appends to the list instead of replacing it.
    """

    # ---Conversation History-----
    # 'add_messages' is the reduces, new messages are appended,
    # not written over. This preserves the full conversation history
    # across all agent calls within a session.
    messages: Annotated[list[BaseMessage], add_messages]

    # ---Session Identity------
    # Unique ID for this study session. Used as the LangGraph thread_id
    # (for checkpointing) and as the key for MCP memory storage.
    session_id: str

    # ---User's learning goal-----
    # Set at session start. Never changes during a session.
    # e.g. "Learn Python closures and decorators from scratch"
    goal: str

    # ---Curriculum Planner output-----
    # Populated by curriculum_planner node.
    # Read by every subsequent agent.
    # None until the planner has run.
    roadmap: StudyRoadmap | None

    # ---Human approval-----
    # Set to True when the user approves the roadmap via interrupt().
    # The graph will not proceed past human_approval until this is True.
    approved: bool

    # ---Study loop position-----
    # Which topic are we currently on?
    # 0 = first topic, len(roadmpa.topics) = all done.
    # The progress_coach node increments this after each quiz.
    current_topic_index: int

    # ---Quiz history-----
    # One QuizResult per topic studied. Grows as the session progresses.
    quiz_results: list[QuizResult]

    # ---Accumulated weak areas-----
    # Duplicated list of concepts the student has struggled with.
    # Build p across all quizzes. The progress_coach uses this to
    # schedule review session.
    weak_areas: list[str]

    # ---File system path-----
    # Where to find the student's study notes.
    # Passed to MCP filesystem tools.
    study_materials_path: str

    # --Error handling -----
    # If a node fails, it writes the rror message here instead of
    # raising an exception. The graph routes to an error handler node.
    # None during normal operation
    error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# State factory function
#
# Always use this to create a new session state.
# Never construct AgentState manually, the factory ensures all fields
# have sensible defaults and no required fields are accidentally omitted.
# ─────────────────────────────────────────────────────────────────────────────

def initial_state(goal: str, session_id: str, study_materials_path: str = 'study_materials/sample_notes') -> dict:
    """
    Create the initial state for a new study session.

    Returns a plain dict (not an AgentState instance) because
    LangGraph accepts plain dicts as initial state, it validates
    the keys against the TypedDict schema internally.

    Args:
        goal:   When use want to learn. e.g. "Learn Python decorrators from scrathch"
        session_id: Unique ID for this session. Used as the LangGraph thread_id.
                    Format: short UUID or human-readable string.
        study_materials_path: Path to directory containing .md notes. Defaults to the sample notes we created.\

    Returns:
         A dict matching the AgentState schema with all fields initialized.
    """

    return {
        'messages': [],  # No messages ye
        'session_id': session_id,
        'goal': goal,
        'roadmap': None,  # Planner hasn't run yet
        'approved': False,  # User hasn't approved yet
        'current_topic_index': 0,  # Start at topic 0
        'quiz_results': [],  # No quizzes yet
        'weak_areas': [],  # No weak areas identified yet
        'study_materials_path': study_materials_path,
        'error': None,  # No errors
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utility: safe state accessors
#
# These helpers make it safe to read from state in agent nodes
# without crashing on missing or None values.
# ─────────────────────────────────────────────────────────────────────────────

def get_current_topic(state: dict) -> Topic | None:
    """
    Get the topic currently being studied, or None if done.

    Usage in an agent node:
        topic = get_current_topic(state)
        if topic is None:
            return {"error": "No current topic"}

    Handles dict or dataclass roadmap (msgpack deserialization returns dicts).
    """
    roadmap = state.get('roadmap')
    if roadmap is None:
        return None

    # After checkpoint deserialization, roadmap may come back as a dict
    if isinstance(roadmap, dict):
        topics_raw = roadmap.get('topics', [])
    else:
        topics_raw = roadmap.topics

    idx = state.get('current_topic_index', 0)
    if idx >= len(roadmap.topics):
        return None

    t = topics_raw[idx]

    # Individual topics may also come back as dicts
    if isinstance(t, dict):
        return Topic.from_dict(t)

    return t


def get_latest_quiz_result(state: dict) -> QuizResult | None:
    """
    Get the most recent quiz result, or None if no quizzes have run.

    Handles dict or dataclass quiz results, after a checkpoint resume,
    LangGraph may deserialize quiz_results as a list of plain dicts.

    Usage in the progress_coach node to analyze the just-completed quiz.
    """
    results = state.get('quiz_results', [])
    if not results:
        return None

    latest = results[-1]

    # After msgpack checkpoint deserialization, quiz results may come
    # back as plain dicts. Reconstruct them using from_dict().

    if isinstance(latest, dict):
        return QuizResult.from_dict(latest)

    return latest


def session_is_complete(state: dict) -> bool:
    """
    True when all topics have been studied.

    Used by the conditional edge function to decide whether to loop
    back to the explainer or end the session.

    Handles dict or dataclass roadmap (msgpack deserialization returns dicts).
    """

    roadmap = state.get('roadmap')
    if roadmap is None:
        return True

    topics = roadmap.get('topics', []) if isinstance(roadmap, dict) else roadmap.topics
    idx = state.get('current_topic_index', 0)
    return idx >= len(roadmap.topics)
