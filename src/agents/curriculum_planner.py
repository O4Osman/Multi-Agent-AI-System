import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import StudyRoadmap, Topic

MODEL_NAME = os.getenv("LLM_MODEL", "qwen2.5:7b")

PLANNER_SYSTEM_PROMPT = """You are an expert curriculum designer. your job is to create
a structed study roadmap when given a learning goal.

Return ONLY valid JSON with no prose, no markdown code fences, no explanation.
The JSON must match this exact schema:

{
    "goal": "the original learning goal exactly as given",
    "total_weeks": <integer between 1 and 12>,
    "weekly_hours": <integer between 3 and 10>,
    "topics": [
     {
       "title": "Short topic name (3-6 words)",
       "description": "One clear sentence explaining what this topic covers",
       "estimated_minutes": <integer between 30 and 120>,
       "prerequisites": ["title of earlier topic if required, else empty list"],
       "status": "pending" 
     }
   ]
}

Rules:
- Order topics from foundational to advanced
- prerequisites must reference earlier topic titles exactly as written
- Aim for 2 to 3 topics
- status must always be "pending"
"""


def parse_roadmap_json(json_string: str) -> StudyRoadmap:
    """Parse the LLM's JSON output into a StudyRoadmap dataclass."""
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON.\n"
            f"Error: {e}\n"
            f"Raw output (first 300 chars): {json_string[:300]}"
        )

    required = ['goal', 'total_weeks', 'topics']
    for field in required:
        if field not in data:
            raise ValueError(f"LLM JSON missing required field: {field}")

    if not isinstance(data['topics'], list) or len(data['topics']) == 0:
        raise ValueError("LLM JSON 'topics' must be non-empty list")

    topics = []
    for i, t in enumerate(data['topics']):
        for field in ['title', 'description', 'estimated_minutes']:
            if field not in t:
                raise ValueError(f"Topic {i} missing required field: {field}")
        topics.append(Topic(
            title=t['title'],
            description=t['description'],
            estimated_minutes=int(t['estimated_minutes']),
            prerequisites=t.get('prerequisites', []),
            status=t.get('status', 'pending')
        ))

    return StudyRoadmap(
        goal=data['goal'],
        total_weeks=int(data['total_weeks']),
        weekly_hours=int(data.get('weekly_hours', 5)),
        topics=topics
    )


def roadmap_from_dict(data: dict) -> StudyRoadmap:
    """Validate the LLM's structured output and convert it into a StudyRoadmap dataclass."""

    required = ["goal", "total_weeks", "topics"]

    for field in required:
        if field not in data:
            raise ValueError(f"LLM response missing required field: {field}")

    if not isinstance(data["topics"], list) or len(data["topics"]) == 0:
        raise ValueError("LLM response 'topics' must be a non-empty list")

    topics = []

    for i, t in enumerate(data["topics"]):
        for field in ["title", "description", "estimated_minutes"]:
            if field not in t:
                raise ValueError(f"Topic {i} missing required field: {field}")

        topics.append(
            Topic(
                title=t["title"],
                description=t["description"],
                estimated_minutes=int(t["estimated_minutes"]),
                prerequisites=t.get("prerequisites", []),
                status=t.get("status", "pending"),
            )
        )

    roadmap = StudyRoadmap(
        goal=data["goal"],
        total_weeks=int(data["total_weeks"]),
        weekly_hours=int(data.get("weekly_hours", 5)),
        topics=topics,
    )

    return roadmap


def build_planner_llm() -> ChatOpenAI:
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.1,
    )

    return llm.with_structured_output(StudyRoadmap)


def curriculum_planner_node(state: dict) -> dict:
    """
    LangGraph node: Curriculum Planner
    Reads: state['goal']
    Writes: state['roadmap'], state['messages'], state['error']
    """

    goal = state.get('goal', '').strip()
    if not goal:
        return {'error': 'No learning goal provided.'}

    print(f"\n[Curriculum Planner] Building roadmap for: {goal}")

    llm = build_planner_llm()

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Create a study roadmap for: {goal}"),
    ]

    print(f"[Curriculum Planner] Calling {MODEL_NAME}")

    response = llm.invoke(messages)

    try:
        roadmap = roadmap_from_dict(response)
    except ValueError as e:
        print(f'[Curriculum Planner] Parse error: {e}')
        return {
            'error': str(e),
            'messages': messages + [response],
        }

    print(f"[Curriculum Planner] Created {len(roadmap.topics)} topics."
          f"{roadmap.total_weeks} weeks")

    for i, topic in enumerate(roadmap.topics, 1):
        prereqs = f" (needs: {', '.join(topic.prerequisites)})" if topic.prerequisites else ""
        print(f"   {i}. {topic.title}{prereqs}, {topic.estimated_minutes} min")

    return {
        'roadmap': roadmap,
        'messages': messages,
        'error': None
    }
