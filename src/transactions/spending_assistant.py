import asyncio
import os

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.tools import BraveSearch
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Checkpointer, Command

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

# Cached MCP tools initialization
_mcp_tools_cache = None


async def _get_mcp_tools():
    """Get MCP tools from cache or fetch if not cached (async).

    MCP-provided tools are commonly async-only; keep this async-first to avoid event loop issues.
    """
    global _mcp_tools_cache

    if _mcp_tools_cache is None:
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


async def create_spending_agent(checkpointer: Checkpointer = None):
    """Create a spending assistant agent.

    Args:
        checkpointer: Optional checkpointer for conversation state persistence

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
        ],
        tools=[brave_search_tool] + db_tools,
        checkpointer=checkpointer,
    )

    return agent


if __name__ == "__main__":
    async def main():
        checkpointer = InMemorySaver()
        spending_assistant = await create_spending_agent(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "1"}}

        print("\n[Step 1] Initial agent invocation...")
        result = await spending_assistant.ainvoke(
            {"messages": [HumanMessage("How much did I spent for restaurants in 2025 from account 1001?")]},
            config=config,
        )

        print("[Step 2] Agent interrupted for approval (HumanInTheLoopMiddleware)")
        print(f"Interruption message: {result["__interrupt__"][0].value["action_requests"][0]["description"] if result["__interrupt__"] else 'No interruption'}")

        print("[Step 3] Resuming with 'approve' decision")
        result = await spending_assistant.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config,
        )

        print("[Step 4] Agent Response")
        print(result["messages"][-1].content)


    asyncio.run(main())
