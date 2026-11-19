# Learning Accelerator

A multi-agent study assistant. Give it a learning goal and it builds a study
roadmap, asks for your approval, explains each topic using your own notes,
quizzes you, and coaches you on weak areas.

Built with **LangGraph**, **MCP**, **A2A**, **CrewAI**, **Langfuse**, and a
**Streamlit** UI. LLM calls go through the **OpenRouter** API (no local model
required).

## How it works

```
curriculum_planner -> human_approval -> explainer -> quiz_generator -> progress_coach
                           ^ (rejected)                                  |    |
                           +---------------------------------------------+    +-> next topic / END
```

| Agent | Role |
|---|---|
| Curriculum Planner | Turns your goal into a roadmap of topics |
| Human Approval | Pauses the graph until you accept or reject the plan |
| Explainer | Reads your notes through MCP tools and explains the topic |
| Quiz Generator | Writes questions and grades your answers |
| Progress Coach | Gives feedback, tracks weak areas, picks the next step |

Supporting services:

- **MCP servers** (`src/mcp_servers`): filesystem (study notes) and memory tools, imported in-process.
- **A2A Quiz Service** (port 9011): the coach delegates quiz grading to it, and falls back to local grading if it is down.
- **CrewAI Study Buddy** (port 9002): offers another angle on a topic when a score is low. Optional.
- **Langfuse**: optional tracing. If it is not configured or unreachable, tracing is skipped.

More detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project layout

```
main.py                  Terminal interface
streamlit_app.py         Web interface (also the deployment entry point)
src/agents/              Planner, explainer, quiz generator, coach, approval
src/graph/               LangGraph workflow and state
src/mcp_servers/         Filesystem and memory MCP tools
src/a2a_services/        A2A quiz service and client
src/crewai_agent/        CrewAI Study Buddy
src/observability/       Langfuse setup
study_materials/         Your notes (markdown) that the Explainer reads
tests/                   Test suite
docker-compose.yml       Local Langfuse stack
```

## Requirements

- Python 3.13
- An [OpenRouter](https://openrouter.ai) API key
- Docker (only if you want local Langfuse tracing)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For tests and evals use `pip install -r requirements-dev.txt` instead.
The repo also has a `uv.lock` if you prefer `uv sync`.

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=provider/model-id        # an OpenRouter model ID

# Optional: Langfuse tracing (local Docker)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

`.env` is git-ignored. Never commit it.

## Run

### Web UI

```powershell
streamlit run streamlit_app.py
```

To launch the quiz service and Study Buddy automatically from the app, set
`START_SIDECARS=true`. Otherwise start them yourself in two terminals:

```powershell
python src/a2a_services/quiz_service.py
python src/crewai_agent/study_buddy.py
```

### Terminal

```powershell
python main.py "Learn Python closures from scratch"
python main.py --resume <session-id>
```

### Langfuse (optional)

```powershell
docker compose up -d
```

Open http://localhost:3000, create a project, and copy its keys into `.env`.
`docker-compose.yml` uses placeholder credentials and is for local use only.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | LLM access | required |
| `LLM_MODEL` | OpenRouter model ID | `qwen2.5:7b` (not valid on OpenRouter, set it) |
| `USE_A2A_QUIZ` | Delegate grading to the A2A quiz service | `true` |
| `USE_STUDY_BUDDY` | Use the CrewAI Study Buddy | `true` |
| `START_SIDECARS` | Streamlit app starts both services itself | `false` |
| `QUIZ_SERVICE_URL` / `STUDY_BUDDY_URL` | Service addresses | localhost defaults |
| `NOTES_PATH` | Folder of study notes | `study_materials/sample_notes` |
| `CHECKPOINTS_DB` | SQLite checkpoint file (terminal app) | `data/checkpoints.db` |
| `LANGFUSE_*` | Tracing keys and URL | unset (tracing off) |

## Add your own notes

Drop markdown files into `study_materials/sample_notes/` (or point `NOTES_PATH`
elsewhere). The Explainer lists, searches, and reads them while explaining.

## Tests

```powershell
pytest
```

Slow evaluation tests are marked `eval` and skipped by default.

## Deploy to Streamlit Cloud

1. Push the repo to GitHub. Make sure `.env` is not tracked.
2. On [share.streamlit.io](https://share.streamlit.io) create an app from the repo, branch `master`, main file `streamlit_app.py`.
3. In **Advanced settings** choose **Python 3.13** and paste your secrets (template in [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example)):

   ```toml
   OPENROUTER_API_KEY = "sk-or-..."
   LLM_MODEL = "provider/model-id"
   START_SIDECARS = "true"
   USE_A2A_QUIZ = "true"
   USE_STUDY_BUDDY = "true"
   ```

4. Leave `LANGFUSE_*` unset. Your local Docker Langfuse is not reachable from the cloud, and the app skips tracing automatically.
5. Deploy. Pushes to `master` redeploy automatically.

Notes:

- Progress is stored in a SQLite file that resets on restart or redeploy.
- The app is public by default. Set a spending limit on your OpenRouter key, or restrict viewers in the app's sharing settings.
- If the app runs out of memory, set `START_SIDECARS=false` and `USE_STUDY_BUDDY=false`. Grading falls back to local, and Study Buddy feedback is disabled.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Langfuse unreachable, tracing disabled` | Docker stack is not running. Start it, or ignore. |
| `port 9011 already in use` | A service is already running. The app reuses it. |
| `ModuleNotFoundError: langchain_openai` | Reinstall from `requirements.txt`. |
| 401 errors from the LLM | Check `OPENROUTER_API_KEY`. |
| Model not found | Set `LLM_MODEL` to a valid OpenRouter ID. |
