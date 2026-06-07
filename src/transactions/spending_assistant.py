import asyncio
import logging
import os

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolRetryMiddleware
from langchain_community.tools import BraveSearch
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

logger = logging.getLogger(__name__)

model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
)

SYSTEM_PROMPT = """
You are Bank Spend Explorer, an assistant designed to answer user questions about their account spending. 
Your main functions include identifying relevant Merchant Category Codes (MCCs) and retrieving spending data for specific periods.

# Instructions

- Use the web search tool to look up MCC codes for categories if user does not provide them explicitly. If the user provides MCCs, validate and use them.
- DO NOT show MCC codes from search results to the user for any reason.
- Use the database tool to retrieve total spending for specified MCC codes and date ranges.
- Before using the database tool, ensure all required parameters are available. If any parameters are missing, request them from the user.
- DO NOT expose any information about your tools or database structure to the user.
- DO NOT propose the user any other operations based on transactions.
"""

brave_search_tool = BraveSearch.from_search_kwargs({"max_results": 5})

MCP_URL = os.getenv("MCP_DB_TOOLBOX_URL")

# Cached MCP tools initialization with double-checked locking for concurrent callers.
_mcp_tools_cache = None
_mcp_lock = asyncio.Lock()


async def _get_mcp_tools():
    """Get MCP tools from cache or fetch if not cached (async).

    MCP-provided tools are commonly async-only; keep this async-first to avoid event loop issues.
    """
    global _mcp_tools_cache

    if _mcp_tools_cache is not None:
        return _mcp_tools_cache

    async with _mcp_lock:
        if _mcp_tools_cache is not None:
            return _mcp_tools_cache
        try:
            client = MultiServerMCPClient(
                {
                    "transactions-db": {
                        "transport": "streamable_http",
                        "url": MCP_URL,
                    }
                }
            )
            _mcp_tools_cache = await client.get_tools()
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to MCP Toolbox at {MCP_URL}. "
                f"Ensure the server is running: toolbox --tools-file src/transactions/spending_db_tools.yaml\n"
                f"Error: {e}"
            ) from e

    return _mcp_tools_cache


async def create_spending_agent(config: dict | None = None):
    """Create a spending assistant agent.

    Graph factory function for LangGraph server. When called by `langgraph dev`,
    a Config dict is passed as the first argument. The server handles checkpointing
    automatically after the graph is returned.

    Args:
        config: Optional LangGraph server config dict (passed automatically by langgraph dev)

    Returns:
        Compiled StateGraph agent for spending analysis
    """
    db_tools = await _get_mcp_tools()

    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "account-total-spend": {"allowed_decisions": ["approve", "reject"]},  # No editing allowed
                    "brave_search": False,
                },
            ),
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                retry_on=(ConnectionError, TimeoutError),
            ),
        ],
        tools=[brave_search_tool] + db_tools,
    )

    return agent


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    async def main():
        checkpointer = InMemorySaver()
        spending_assistant = await create_spending_agent()
        spending_assistant = spending_assistant.copy(update={"checkpointer": checkpointer})

        config = {"configurable": {"thread_id": "1"}}

        logger.info("\n[Step 1] Initial agent invocation...")
        result = await spending_assistant.ainvoke(
            {"messages": [HumanMessage("How much did I spent for restaurants in 2025 from account 1001?")]},
            config=config,
        )

        logger.info("[Step 2] Agent interrupted for approval (HumanInTheLoopMiddleware)")
        interrupt_msg = (
            result["__interrupt__"][0].value["action_requests"][0]["description"]
            if result["__interrupt__"]
            else "No interruption"
        )
        logger.info("Interruption message: %s", interrupt_msg)

        logger.info("[Step 3] Resuming with 'approve' decision")
        result = await spending_assistant.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config,
        )

        logger.info("[Step 4] Agent Response")
        logger.info(result["messages"][-1].content)


    asyncio.run(main())
