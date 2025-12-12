"""LangGraph agent wrapper with A2A streaming interface."""

from enum import Enum
from typing import AsyncGenerator

from a2a.types import Message, Role
from a2a.utils.message import get_message_text
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from flights.flight_assistant import create_flight_assistant


class ResponseState(str, Enum):
    """Agent response states for A2A protocol."""
    COMPLETED = "completed"
    ERROR = "error"


class ResponseFormat(BaseModel):
    """Structured response format for A2A protocol.

    Attributes:
        state: Current state of the response
        content: Message content to send to user
        is_task_complete: Whether the task is finished
    """
    state: ResponseState
    content: str
    is_task_complete: bool = False

    @property
    def is_final(self) -> bool:
        """Check if this is a final response state."""
        return self.state in [
            ResponseState.COMPLETED,
            ResponseState.ERROR
        ]


class FlightAgent:
    """LangGraph flight agent wrapper for A2A protocol.

    This class wraps the existing flight booking agent with an A2A-compliant
    streaming interface, enabling multi-turn conversations and task state management.
    """

    def __init__(self):
        """Initialize flight agent with memory checkpointer."""
        # Create agent with MemorySaver for multi-turn conversations
        checkpointer = MemorySaver()
        self.agent = create_flight_assistant(checkpointer=checkpointer)
        print("FlightAgent initialized with MemorySaver checkpointer")

    async def stream(
            self,
            messages: list[Message],
            thread_id: str
    ) -> AsyncGenerator[ResponseFormat, None]:
        """Stream agent responses with incremental updates.

        Args:
            messages: List of A2A messages (conversation history)
            thread_id: Thread identifier for conversation context

        Yields:
            ResponseFormat objects with agent state and content
        """
        try:
            # Convert A2A messages to LangChain format
            lc_messages = self._convert_messages(messages)

            # Configure thread for conversation state
            config = {"configurable": {"thread_id": thread_id}}

            print(f"Streaming agent response for thread {thread_id}")

            # Track if we've seen any agent responses
            has_response = False
            final_content = ""

            # Stream agent execution
            async for chunk in self.agent.astream(
                    {"messages": lc_messages},
                    config=config,
                    stream_mode="values"
            ):
                # Process the chunk to extract the latest message
                if "messages" in chunk:
                    messages_list = chunk["messages"]
                    if messages_list:
                        last_message = messages_list[-1]

                        # Check if it's an AI message (agent response)
                        if isinstance(last_message, AIMessage):
                            has_response = True
                            final_content = last_message.content
                            print("Agent response received")

            # Generate final response
            if has_response and final_content:
                yield ResponseFormat(
                    state=ResponseState.COMPLETED,
                    content=final_content,
                    is_task_complete=True
                )
            else:
                # No response generated
                yield ResponseFormat(
                    state=ResponseState.ERROR,
                    content="No response generated from agent",
                    is_task_complete=False
                )

        except Exception as e:
            print(f"Error streaming agent response: {e}")
            yield ResponseFormat(
                state=ResponseState.ERROR,
                content=f"Error processing request: {str(e)}",
                is_task_complete=False
            )

    def _convert_messages(self, a2a_messages: list[Message]) -> list:
        """Convert A2A messages to LangChain format.

        Args:
            a2a_messages: List of A2A Message objects

        Returns:
            List of LangChain message objects (HumanMessage, AIMessage)
        """
        lc_messages = []

        for msg in a2a_messages:
            # Extract text from message
            text = get_message_text(msg)

            # Convert based on role
            if msg.role == Role.user:
                lc_messages.append(HumanMessage(content=text))
            elif msg.role == Role.agent:
                lc_messages.append(AIMessage(content=text))
            else:
                # Default to human message for unknown roles
                print(f"Warning: Unknown role {msg.role}, treating as user message")
                lc_messages.append(HumanMessage(content=text))

        print(f"Converted {len(a2a_messages)} A2A messages to LangChain format")
        return lc_messages
