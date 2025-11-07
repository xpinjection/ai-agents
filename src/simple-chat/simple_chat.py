from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
)

SYSTEM_PROMPT = """
Explain the requested term in simple words as if you are a school teacher speaking to 10-year-old kids. 
Keep your explanation to one paragraph.
"""

class CustomContext(BaseModel):
    user_name: Optional[str] = None

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.user_name
    if user_name is not None:
        return f"{SYSTEM_PROMPT}. Address the user as {user_name}."
    return SYSTEM_PROMPT

simple_chat = create_agent(
    model=model,
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

if __name__ == "__main__":
    checkpointer = InMemorySaver()
    simple_chat_with_memory = create_agent(
        model=model,
        middleware=[dynamic_system_prompt],
        checkpointer=checkpointer,
        context_schema=CustomContext,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    context = CustomContext(user_name="Mikalai")
    agent_response = simple_chat_with_memory.invoke({"messages": [HumanMessage("Machine Learning")]},
                                        config = config, context=context)
    print(agent_response)
    agent_response = simple_chat_with_memory.invoke({"messages": [HumanMessage("How is it related to LLM?")]},
                                        config=config, context=context)
    print(agent_response)
