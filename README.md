# AI Agents with LangChain & LangGraph

A collection of AI agent implementations demonstrating various agentic patterns including RAG systems, supervisor patterns, routing agents, and workflow orchestrations using LangChain and LangGraph.

## Quick Start

```bash
# 1. Clone and navigate to the project
cd ai-agents

# 2. Create virtual environment and install dependencies
uv sync

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 4. Start PostgreSQL database
docker compose up -d

# 5. Start MCP Toolbox for Databases
toolbox --tools-file src/transactions/spending_db_tools.yaml

# 6. Run LangGraph development server
uv run langgraph dev
```

Access LangGraph Studio at `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024` (or the port shown in terminal output).

## Prerequisites

- **Python 3.13** (required)
- **UV** (recommended package manager)
- **Docker** and **Docker Compose** (for PostgreSQL database)
- **OpenAI API Key** (required for all agents)
- **LangSmith API Key** (optional, for tracing)
- **Brave Search API Key** (required for spending assistant)

## Detailed Setup

### 1. Install UV Package Manager

UV is a package manager for Python projects that makes it easy to install, update, and remove dependencies. 

Follow [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/) or install UV quickly from the command line:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### 2. Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - LangSmith for tracing and monitoring
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true

# Required for spending assistant
BRAVE_SEARCH_API_KEY=your_brave_search_api_key_here

# Required for supervisor and spending agents (PostgreSQL database)
POSTGRES_DB_URL=postgresql://root:123qwe@localhost:5432/ai

# Required for spending assistant (MCP Toolbox for Databases)
MCP_DB_TOOLBOX_URL=http://localhost:5000/mcp
```

**Where to get API keys:**
- OpenAI: https://platform.openai.com/api-keys
- LangSmith: https://smith.langchain.com/settings
- Brave Search: https://brave.com/search/api/

### 3. Install Dependencies

This project uses UV for dependency management:

```bash
# Create virtual environment and install all dependencies from pyproject.toml
uv sync
```

### 4. Start Required Services

#### PostgreSQL Database

Start PostgreSQL database with Docker Compose:

```bash
docker compose up -d
```
Verify database is running:

```bash
docker compose ps
```

This starts a PostgreSQL instance with pgvector extension on `localhost:5432`.

#### MCP Toolbox for Databases

Start the MCP Toolbox for Databases server with the spending agent's configuration:

```bash
# From the project root
toolbox --tools-file src/transactions/spending_db_tools.yaml
```

This starts the MCP server on `http://localhost:5000/mcp`.

### 5. Run LangGraph Development Server

Start the LangGraph development server:

```bash
uv run langgraph dev
```

This will:
- Start the LangGraph API server
- Launch LangGraph Studio in your browser
- Hot-reload on code changes
- Make all agents available for testing

## Available Agents

The following agents are available in LangGraph Studio:

### Simple Agents
- **chat** - Basic conversational agent
- **chef** - Cooking assistant with recipe tools
- **flights** - Flight booking system with search, book, and cancel capabilities

### RAG Agents (Retrieval-Augmented Generation)
- **conventions** - Basic RAG for API conventions lookup
- **conventions_agentic** - Advanced RAG with query rewriting and document grading

### Database Agents
- **spending** - Transaction spending analysis assistant (requires special setup)
- **supervisor** - Multi-agent supervisor with DBA and QA engineers

### Workflow Patterns
- **prompt_chaining** - Sequential prompt execution
- **parallel** - Parallel task execution
- **routing** - Conditional routing (CV review example)
- **evaluator_optimizer** - Evaluation and optimization workflows

## Agent-Specific Setup

### RAG Agents (conventions, conventions_agentic)

These agents require vector database indexing before first use:

```bash
# Index the conventions documents
uv run src/conventions/conventions_indexer.py
```

This will:
- Load markdown and PDF files from `src/conventions/sources/`
- Split documents into chunks
- Create vector embeddings using OpenAI
- Store in local Chroma database at `src/conventions/chroma_db/`

**Source documents included:**
- API conventions (REST API, API types)
- Kafka messaging conventions

### Spending Assistant

The spending assistant is the most complex agent and requires additional setup:

#### 1. Install MCP Toolbox for Databases

The spending assistant uses MCP Toolbox for Databases to interact with PostgreSQL.

Follow the [installation instructions](https://googleapis.github.io/genai-toolbox/getting-started/mcp_quickstart/) for your platform from the official documentation.

#### 2. Start MCP server

Start the MCP Toolbox for Databases server with the spending agent's configuration:

```bash
# From the project root
toolbox --tools-file src/transactions/spending_db_tools.yaml
```

This starts the MCP server on `http://localhost:5000/mcp` which provides the `account-total-spend` tool to the agent.

**Keep this server running in a separate terminal.**

#### 3. Create Database Schema

Create the transactions table in PostgreSQL:

```bash
# Connect to PostgreSQL
psql postgresql://root:123qwe@localhost:5432/ai

# Run the schema creation script
\i src/transactions/data/create_db.sql

# Or directly from command line
psql postgresql://root:123qwe@localhost:5432/ai -f src/transactions/data/create_db.sql
```

#### 4. Load Transaction Data

Load sample transaction data:

```bash
# From psql
\i src/transactions/data/transactions_data.sql

# Or from command line
psql postgresql://root:123qwe@localhost:5432/ai -f src/transactions/data/transactions_data.sql
```

This loads sample transactions for three accounts (1001, 2002, 3003) with various merchants and MCC codes.

#### 5. Verify Setup

```bash
# Verify MCP server is running
curl http://localhost:5000/mcp

# Verify database has data
psql postgresql://root:123qwe@localhost:5432/ai -c "SELECT COUNT(*) FROM transactions;"
```
Install and run [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) to [verify MCP server is working correctly](https://googleapis.github.io/genai-toolbox/getting-started/mcp_quickstart/#step-3-connect-to-mcp-inspector).


#### 6. Environment Variables Required

Ensure these are set in your `.env` file:

```bash
BRAVE_SEARCH_API_KEY=your_brave_search_api_key_here
POSTGRES_DB_URL=postgresql://root:123qwe@localhost:5432/ai
MCP_DB_TOOLBOX_URL=http://localhost:5000/mcp
```

#### How the Spending Assistant Works

The agent:
1. Uses **Brave Search** to find MCC (Merchant Category Codes) for spending categories
2. Calls the **MCP Database Toolbox** `account-total-spend` tool to query transaction totals
3. Filters by account_id, MCC codes, and date range
4. Returns spending analysis to the user

Example queries:
- "How much did account 1001 spend on restaurants in October 2025?"
- "What's my total spending on groceries last month?"

### Supervisor Agent

Requires PostgreSQL database connection. Ensure the database is running:

```bash
# Check database status
docker compose ps
```

The supervisor agent demonstrates a multi-agent pattern with:
- **DBA Agent**: Generates SQL queries
- **QA Agent**: Tests SQL queries
- **Supervisor**: Coordinates between agents

## Running Individual Agents

You can also run agents directly as Python scripts:

```bash
# Flight booking assistant
uv run src/flights/flight_assistant.py

# Supervisor pattern
uv run src/workflows/supervisor.py

# Routing pattern
uv run src/workflows/routing.py

# RAG with conventions
uv run src/conventions/conventions.py
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_flight_assistant.py

# Run with verbose output
uv run pytest -v

# Run integration tests only
uv run pytest tests/test_flight_assistant.py::test_flight_search

# Run regression tests
uv run pytest tests/test_flight_assistant_regression.py
```

Tests use the `agentevals` library for trajectory-based agent evaluation.

## Development Workflow

### Using LangGraph Studio

1. Start the development server: `uv run langgraph dev`
2. Open LangGraph Studio (automatically opens in browser)
3. Select an agent from the dropdown
4. Test interactions in the chat interface
5. View state, messages, and execution graph
6. Make code changes (hot-reloads automatically)

### Adding a New Agent

1. Create your agent file in `src/your_module/your_agent.py`
2. Export your compiled graph (e.g., `my_agent = workflow.compile()`)
3. Register in `langgraph.json`:
```json
{
  "graphs": {
    "my_agent": "./src/your_module/your_agent.py:my_agent"
  }
}
```
4. Restart `uv run langgraph dev` to load the new agent

### Adding Dependencies

To add new Python packages to the project:

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name
```

### Debugging

Enable LangSmith tracing for detailed execution logs:

```bash
# In .env file
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
```

View traces at https://smith.langchain.com

## Production Deployment

### Build Docker Image

```bash
docker build -t ai-agents:0.1.0 .
```

### Deploy with Docker Compose

```bash
# Update image tag in compose-deploy.yaml if needed
docker compose -f compose-deploy.yaml up -d
```

This starts:
- PostgreSQL with persistence
- Redis for caching
- LangGraph API server

Access API at `http://localhost:8123`

### Environment Variables for Production

Update `.env` for production:
- Use secure database credentials
- Configure Redis URI
- Set production API keys
- Adjust `MCP_DB_TOOLBOX_URL` if using external services
