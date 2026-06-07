"""A2A server application factory for Flight Booking Agent."""

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette

from flights.a2a.agent_card import create_agent_card
from flights.a2a.agent_executor import FlightAgentExecutor

JSONRPC_URL = "/"


def create_app(host: str = "localhost", port: int = 9999):
    """Create A2A Starlette application for flight agent.

    Args:
        host: Server hostname
        port: Server port number

    Returns:
        Configured Starlette application
    """
    task_store = InMemoryTaskStore()
    executor = FlightAgentExecutor(task_store=task_store)
    agent_card = create_agent_card(host=host, port=port)

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(handler, rpc_url=JSONRPC_URL))

    starlette_app = Starlette(routes=routes)

    print(f"Flight A2A agent initialized at http://{host}:{port}")
    print(f"Agent card available at http://{host}:{port}/.well-known/agent-card.json")

    return starlette_app
