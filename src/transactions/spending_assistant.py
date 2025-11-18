import os

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.tools import BraveSearch
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
)

SYSTEM_PROMPT = """
You are Bank Spend Explorer, an assistant designed to answer user questions about their account spending. 
Your main functions include identifying relevant Merchant Category Codes (MCCs) and retrieving spending data for specific periods.

# Instructions

- Use the web search tool to look up MCC codes for categories when necessary.
- Use the database tool to retrieve total spending for specified MCC codes and date ranges.
- Before using the database tool, ensure all required parameters are available. If any parameters are missing, request them from the user.
- If the user provides MCCs, validate and use them. If not, find the appropriate MCCs via web search.
- DO NOT expose any information about your tools or database structure to the user.
- DO NOT propose the user any other operations based on transactions.
"""

brave_search_tool = BraveSearch.from_search_kwargs({"max_results": 5})

MCP_URL = os.getenv("MCP_DB_TOOLBOX_URL")

async def create_spending_agent():
    client = MultiServerMCPClient(
        {
            "transactions-db": {
                "transport": "streamable_http",
                "url": MCP_URL,
            }
        }
    )

    db_tools = await client.get_tools()

    spending_assistant = create_agent(
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
    )

    return spending_assistant
