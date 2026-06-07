# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangChain/LangGraph AI agents project demonstrating various agentic patterns and workflows. The project contains multiple example implementations including RAG systems, supervisor patterns, routing agents, and workflow orchestrations.

## Environment Setup

- Python 3.13 required (>=3.13,<3.14)
- Uses `uv` for dependency management
- All dependencies pinned to exact versions in `pyproject.toml` — use `uv add pkg==version` to maintain pinning
- Environment variables in `.env` file (OpenAI API keys, database URLs)
- PostgreSQL with pgvector extension for vector storage
- MCP Database Toolbox (GenAI Toolbox) for spending assistant - standalone tool, not a Python package

## Key Commands

### Development
```bash
# Start PostgreSQL database
docker compose up -d

# Install dependencies
uv sync

# Add a new dependency
uv add package-name

# Run a specific agent module directly
uv run python -m src.flights.flight_assistant
uv run python -m src.workflows.supervisor
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_flight_assistant.py

# Run with verbose output
uv run pytest -v
```

### LangGraph Deployment
```bash
# Run LangGraph server locally (development mode)
uv run langgraph dev

# Build Docker image for deployment
docker build -t ai-agents .

# Deploy with Docker Compose (persistent mode)
docker compose -f compose-deploy.yaml up
```

## Architecture

### Project Structure
- `src/`: All agent implementations organized by domain
- `tests/`: Integration and regression tests using agentevals
- `langgraph.json`: Defines all available LangGraph agents/graphs
- `Dockerfile`: Production deployment configuration

### Agent Patterns Implemented

**Simple Agents**
- `simple-chat/`: Basic chat agent
- `coooking/`: Cooking assistant with tools (directory name intentionally spelled with three 'o's)
- `flights/`: Flight booking system with error handling middleware
  - `flights/flight_service.py`: Shared flight data and business logic used by all entrypoints
  - `flights/a2a/`: A2A protocol server exposing the flight agent (run: `uv run python -m src.flights.a2a`)
  - `flights/mcp/`: FastMCP server exposing flight tools (run: `uv run python -m src.flights.mcp`)
- `transactions/`: Spending assistant (requires MCP Database Toolbox, Brave Search API, PostgreSQL with transaction data)
- `native-llm/`: Native LLM API integration

**RAG Implementations**
- `conventions/`: Basic RAG using Chroma vector store
- `conventions_agentic/`: Advanced RAG with query rewriting, document grading, and relevance checks

**Workflow Patterns**
- `workflows/prompt_chaining.py`: Sequential prompt chaining
- `workflows/parallelization.py`: Parallel execution of tasks
- `workflows/routing.py`: Conditional routing based on state (e.g., CV review score routing to interview/rejection)
- `workflows/supervisor.py`: Supervisor pattern with specialized sub-agents (DBA + QA engineer)
- `workflows/evaluator_optimizer.py`: Evaluation and optimization workflows

### Key Architectural Concepts

**LangGraph State Management**
- All agents use `StateGraph` with typed state classes
- State extends `MessagesState` or custom `TypedDict`
- Conditional edges route execution based on state
- `START` and `END` constants define graph boundaries

**Tool Integration**
- Tools defined with `@tool` decorator and Pydantic schemas for args
- `ToolNode` wraps tools for graph integration
- `tools_condition` helper for conditional tool routing
- Custom middleware (e.g., `ToolErrorHandlerMiddleware`) for error handling

**Agent Creation**
- Use `create_agent()` helper from langchain.agents
- Accepts model, system_prompt, tools, middleware, and checkpointer
- Returns compiled StateGraph

**Multi-Agent Patterns**
- Supervisor delegates to specialized agents (see supervisor.py)
- Each sub-agent has focused tools and system prompt
- Routing agents use conditional edges to direct flow

### Testing Strategy

Uses `agentevals` library for trajectory-based testing:
- `create_trajectory_match_evaluator()` validates agent behavior
- Define expected message sequences (HumanMessage → AIMessage → ToolMessage)
- Supports strict or flexible matching modes
- Integration tests validate full agent flows
- Regression tests ensure behavior consistency

### Special Requirements

**Spending Assistant**
- Requires MCP Database Toolbox: `toolbox --tools-file src/transactions/spending_db_tools.yaml`
- Needs PostgreSQL with transactions table (schema and data in `src/transactions/data/`)
- Requires Brave Search API key for MCC code lookups
- Config file: `src/transactions/spending_db_tools.yaml`

**RAG Agents (Conventions)**
- Require document indexing: `uv run python src/conventions/conventions_indexer.py`
- Source documents in `src/conventions/sources/` (MD and PDF)
- Creates Chroma vector store in `src/conventions/chroma_db/`

### Deployment

**Local Development**
- `uv run langgraph dev` starts development server
- Agents accessible via LangGraph Studio

**Production**
- Base image: `langchain/langgraph-api:3.13`
- All graphs registered in `LANGSERVE_GRAPHS` env var
- Persists state with PostgreSQL checkpointer
- Uses in-memory storage for testing

## Model Configuration

- Primary model: GPT-4o-mini (most implementations)
- Reasoning model: GPT-5-mini with reasoning_effort (supervisor, routing patterns)
- Embeddings: text-embedding-3-large (OpenAI)

## Important Conventions

**Error Handling**
- Use `ToolException` for tool errors
- Implement middleware to convert exceptions to ToolMessages
- Return human-friendly error messages to users

**Graph Construction**
- Add nodes before edges
- Use `add_conditional_edges()` for routing logic
- Always compile graph with `workflow.compile()`
- Optional: pass checkpointer for persistence

**State Updates**
- Nodes return dict updates to merge into state
- Messages append to existing message list
- Other fields overwrite existing values

**Vector Store Usage**
- Chroma for local vector storage
- Persist directory pattern: `./src/{module}/chroma_db`
- Use `as_retriever()` with k parameter for similarity search
