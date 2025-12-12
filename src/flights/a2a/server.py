"""A2A server application factory for Flight Booking Agent."""

from a2a.server.apps.jsonrpc import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from flights.a2a.agent_card import create_agent_card
from flights.a2a.agent_executor import FlightAgentExecutor


def create_app(host: str = "localhost", port: int = 9999):
    """Create A2A Starlette application for flight agent.

    Args:
        host: Server hostname
        port: Server port number

    Returns:
        Configured Starlette application
    """
    # Create task store for managing task state
    task_store = InMemoryTaskStore()

    # Create agent executor
    executor = FlightAgentExecutor(task_store=task_store)

    # Create request handler
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )

    # Create agent card
    agent_card = create_agent_card(host=host, port=port)

    # Create and configure A2A application
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler
    )

    # Build Starlette app
    starlette_app = app.build()

    print(f"Flight A2A agent initialized at http://{host}:{port}")
    print(f"Agent card available at http://{host}:{port}/.well-known/agent.json")

    return starlette_app
