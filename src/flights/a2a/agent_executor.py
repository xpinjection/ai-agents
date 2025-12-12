"""A2A protocol executor for Flight Booking Agent."""

import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Task, TaskState, TaskStatus,
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent,
    Message,
    InternalError, UnsupportedOperationError
)
from a2a.utils.helpers import build_text_artifact
from a2a.utils.message import new_agent_text_message
from a2a.utils.task import new_task

from flights.a2a.agent import FlightAgent, ResponseState


class FlightAgentExecutor(AgentExecutor):
    """A2A executor for flight booking agent.

    This class implements the AgentExecutor interface to handle A2A protocol
    requests, manage task lifecycle, and stream agent responses.
    """

    def __init__(self, task_store):
        """Initialize the flight agent executor.

        Args:
            task_store: Task store for managing task state (InMemoryTaskStore or DatabaseTaskStore)
        """
        self.agent = FlightAgent()
        self.task_store = task_store
        print("FlightAgentExecutor initialized")

    async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue
    ) -> None:
        """Execute flight agent task with streaming.

        This method handles the complete request lifecycle:
        1. Validate request
        2. Retrieve user input
        3. Create or retrieve task
        4. Stream agent responses
        5. Handle outcomes (working/input_required/completed/error)

        Args:
            context: Request context with user input and metadata
            event_queue: Queue for sending events back to client

        Raises:
            ServerError: If execution fails
        """
        try:
            # Validate request (placeholder for future validation)
            if not self._validate_request(context):
                print("Warning: Invalid request received, proceeding anyway")

            # Get user input text from context
            user_input_text = context.get_user_input()
            if not user_input_text:
                raise ValueError("No user input provided in request")

            print(f"Processing message: {user_input_text[:100]}...")  # Log first 100 chars

            # Get the actual Message object from the request
            user_message = context.message

            # Retrieve or create task
            task = await self._get_or_create_task(context, user_message)
            print(f"Task ID: {task.id}, Context ID: {task.context_id}")

            # Build conversation history from task (already includes all messages)
            messages = task.history

            # Stream agent responses
            async for response in self.agent.stream(
                    messages=messages,
                    thread_id=task.id
            ):
                print(f"Agent response state: {response.state}")

                if response.state == ResponseState.COMPLETED:
                    # Task complete - create artifact and finish
                    print("Agent task completed successfully")

                    # Create artifact with response content using SDK utility
                    artifact = build_text_artifact(
                        artifact_id=str(uuid.uuid4()),
                        text=response.content
                    )

                    # Create completion message
                    completion_message = new_agent_text_message(
                        response.content,
                        context_id=task.context_id,
                        task_id=task.id
                    )

                    # Update task status
                    task.status = TaskStatus(
                        state=TaskState.completed,
                        message=completion_message
                    )
                    task.artifacts = [artifact]
                    task.history.append(completion_message)

                    # Save task with error handling
                    try:
                        await self.task_store.save(task)
                    except Exception as e:
                        print(f"Error: Failed to save task {task.id}: {e}")
                        raise InternalError(f"Failed to persist task state: {str(e)}")

                    # Send artifact update
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            task_id=task.id,
                            context_id=task.context_id,
                            artifact=artifact
                        )
                    )
                    print("Artifact update sent")

                    # Send completion status
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=task.id,
                            context_id=task.context_id,
                            status=task.status,
                            final=True  # Task completed
                        )
                    )
                    print(f"Task {task.id} completed")
                    return

                elif response.state == ResponseState.ERROR:
                    # Handle error state
                    print(f"Error: Agent error: {response.content}")

                    # Create error message
                    error_message = new_agent_text_message(
                        f"Error: {response.content}",
                        context_id=task.context_id,
                        task_id=task.id
                    )

                    # Update task status to failed
                    task.status = TaskStatus(
                        state=TaskState.failed,
                        message=error_message
                    )
                    task.history.append(error_message)

                    # Save task with error handling
                    try:
                        await self.task_store.save(task)
                    except Exception as e:
                        print(f"Error: Failed to save failed task {task.id}: {e}")
                        # Continue to send error event even if save fails

                    # Send error status update
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=task.id,
                            context_id=task.context_id,
                            status=task.status,
                            final=True  # Task failed
                        )
                    )
                    print(f"Task {task.id} failed")
                    return

        except Exception as e:
            print(f"Error executing flight agent: {e}")
            # Re-raise the exception, the framework will handle it
            raise

    async def cancel(
            self,
            context: RequestContext,
            event_queue: EventQueue
    ) -> None:
        """Cancel task execution.

        Cancellation is not supported for this agent as flight booking
        operations execute quickly and don't benefit from cancellation.

        Args:
            context: Request context
            event_queue: Event queue

        Raises:
            UnsupportedOperationError: Always raised as cancellation is not supported
        """
        print("Warning: Cancellation requested but not supported")
        raise UnsupportedOperationError(
            "Cancellation is not supported for flight booking agent. "
            "Flight operations execute quickly and complete synchronously."
        )

    async def _get_or_create_task(
            self,
            context: RequestContext,
            user_message: Message
    ) -> Task:
        """Retrieve existing task or create new one.

        Args:
            context: Request context
            user_message: User's message

        Returns:
            Task object (existing or newly created)
        """
        task_id = user_message.task_id

        if task_id:
            # Retrieve existing task for multi-turn conversation
            task = await self.task_store.get(task_id)
            if task:
                print(f"Retrieved existing task: {task_id}")
                # Append new message to history for continuing conversation
                if not task.history:
                    task.history = []
                task.history.append(user_message)
                return task
            else:
                print(f"Warning: Task {task_id} not found in store, creating new task")

        # Create new task (new_task utility already includes user_message in history)
        task = new_task(user_message)
        print(f"Created new task: {task.id}")
        return task

    def _validate_request(self, context: RequestContext) -> bool:
        """Validate incoming request.

        This is a placeholder for future validation logic such as:
        - Schema validation
        - Authentication checks
        - Rate limiting
        - Content filtering

        Args:
            context: Request context

        Returns:
            True if valid, False otherwise
        """
        # Future: Add validation logic here
        return True
