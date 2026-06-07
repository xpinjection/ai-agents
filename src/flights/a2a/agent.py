"""LangGraph agent wrapper with A2A streaming interface."""

import logging
from enum import Enum
from typing import AsyncGenerator

from a2a.helpers import get_message_text
from a2a.types import Message, Role
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from flights.flight_assistant import create_flight_assistant

logger = logging.getLogger(__name__)


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
        logger.info("FlightAgent initialized with MemorySaver checkpointer")

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

            logger.info("Streaming agent response for thread %s", thread_id)

            # Track AI messages so we yield each new one as an interim update
            # and only mark the final emission as is_task_complete=True.
            last_emitted_content: str | None = None
            last_emitted_id: str | None = None

            async for chunk in self.agent.astream(
                    {"messages": lc_messages},
                    config=config,
                    stream_mode="values"
            ):
                if "messages" not in chunk:
                    continue
                messages_list = chunk["messages"]
                if not messages_list:
                    continue
                last_message = messages_list[-1]
                if not isinstance(last_message, AIMessage):
                    continue
                if not last_message.content:
                    # Skip tool-call-only chunks with no user-visible content.
                    continue

                message_id = getattr(last_message, "id", None)
                if message_id == last_emitted_id and last_message.content == last_emitted_content:
                    continue

                last_emitted_id = message_id
                last_emitted_content = last_message.content
                logger.info("Agent interim response received")
                yield ResponseFormat(
                    state=ResponseState.COMPLETED,
                    content=last_message.content,
                    is_task_complete=False,
                )

            if last_emitted_content is not None:
                # Final marker so the A2A client knows the task is done.
                yield ResponseFormat(
                    state=ResponseState.COMPLETED,
                    content=last_emitted_content,
                    is_task_complete=True,
                )
            else:
                yield ResponseFormat(
                    state=ResponseState.ERROR,
                    content="No response generated from agent",
                    is_task_complete=False,
                )

        except Exception as e:
            logger.exception("Error streaming agent response")
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
            if msg.role == Role.ROLE_USER:
                lc_messages.append(HumanMessage(content=text))
            elif msg.role == Role.ROLE_AGENT:
                lc_messages.append(AIMessage(content=text))
            else:
                # Default to human message for unknown roles
                logger.warning("Unknown role %s, treating as user message", msg.role)
                lc_messages.append(HumanMessage(content=text))

        logger.info("Converted %d A2A messages to LangChain format", len(a2a_messages))
        return lc_messages
