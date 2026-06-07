"""A2A protocol executor for Flight Booking Agent."""

from a2a.helpers import (
    new_task_from_user_message,
    new_text_artifact_update_event,
    new_text_status_update_event,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    InternalError,
    Message,
    Task,
    TaskState,
    UnsupportedOperationError,
)

from flights.a2a.agent import FlightAgent, ResponseState


class FlightAgentExecutor(AgentExecutor):
    """A2A executor for flight booking agent.

    Implements the AgentExecutor interface using the v1 Task-lifecycle
    streaming pattern: enqueue the Task first, then emit artifact/status
    update events until a terminal state.
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
        """Execute flight agent task with streaming."""
        try:
            user_input_text = context.get_user_input()
            if not user_input_text:
                raise ValueError("No user input provided in request")

            print(f"Processing message: {user_input_text[:100]}...")

            user_message = context.message

            # Retrieve or create task. v1 streaming protocol requires the
            # Task object to be enqueued as the first event in a task stream.
            task = await self._get_or_create_task(context, user_message)
            print(f"Task ID: {task.id}, Context ID: {task.context_id}")

            await event_queue.enqueue_event(task)

            # Build conversation history from task (already includes all messages)
            messages = list(task.history)

            async for response in self.agent.stream(
                    messages=messages,
                    thread_id=task.id,
            ):
                print(f"Agent response state: {response.state}")

                if response.state == ResponseState.COMPLETED:
                    print("Agent task completed successfully")

                    artifact_event = new_text_artifact_update_event(
                        task_id=task.id,
                        context_id=task.context_id,
                        name="flight_agent_response",
                        text=response.content,
                        last_chunk=True,
                    )
                    await event_queue.enqueue_event(artifact_event)
                    print("Artifact update sent")

                    status_event = new_text_status_update_event(
                        task_id=task.id,
                        context_id=task.context_id,
                        state=TaskState.TASK_STATE_COMPLETED,
                        text=response.content,
                    )
                    await event_queue.enqueue_event(status_event)

                    try:
                        await self._save_task(context, task)
                    except Exception as e:
                        print(f"Error: Failed to save task {task.id}: {e}")
                        raise InternalError(f"Failed to persist task state: {str(e)}")

                    print(f"Task {task.id} completed")
                    return

                elif response.state == ResponseState.ERROR:
                    print(f"Error: Agent error: {response.content}")

                    status_event = new_text_status_update_event(
                        task_id=task.id,
                        context_id=task.context_id,
                        state=TaskState.TASK_STATE_FAILED,
                        text=f"Error: {response.content}",
                    )
                    await event_queue.enqueue_event(status_event)

                    try:
                        await self._save_task(context, task)
                    except Exception as e:
                        print(f"Error: Failed to save failed task {task.id}: {e}")

                    print(f"Task {task.id} failed")
                    return

        except Exception as e:
            print(f"Error executing flight agent: {e}")
            raise

    async def cancel(
            self,
            context: RequestContext,
            event_queue: EventQueue
    ) -> None:
        """Cancellation is not supported for this agent."""
        print("Warning: Cancellation requested but not supported")
        raise UnsupportedOperationError(
            "Cancellation is not supported for flight booking agent."
        )

    async def _get_or_create_task(
            self,
            context: RequestContext,
            user_message: Message,
    ) -> Task:
        """Retrieve existing task or create new one."""
        task_id = user_message.task_id

        if task_id:
            task = await self.task_store.get(task_id, context.call_context)
            if task:
                print(f"Retrieved existing task: {task_id}")
                task.history.append(user_message)
                return task
            print(f"Warning: Task {task_id} not found in store, creating new task")

        task = new_task_from_user_message(user_message)
        print(f"Created new task: {task.id}")
        return task

    async def _save_task(self, context: RequestContext, task: Task) -> None:
        """Save task to the task store (v1 requires ServerCallContext)."""
        await self.task_store.save(task, context.call_context)
