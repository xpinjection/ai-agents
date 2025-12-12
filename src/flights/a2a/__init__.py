"""A2A protocol integration for Flight Booking Agent.

This module provides an Agent-to-Agent (A2A) protocol interface for the
flight booking agent, enabling interoperability with other AI agents.

Example:
    Start the A2A server:
        $ uv run python -m src.flights.a2a

    Custom host and port:
        $ uv run python -m src.flights.a2a --host 0.0.0.0 --port 8080
"""

from flights.a2a.agent import FlightAgent, ResponseFormat, ResponseState
from flights.a2a.agent_card import create_agent_card
from flights.a2a.agent_executor import FlightAgentExecutor
from flights.a2a.server import create_app

__version__ = "1.0.0"

__all__ = [
    "FlightAgent",
    "ResponseFormat",
    "ResponseState",
    "FlightAgentExecutor",
    "create_agent_card",
    "create_app",
]
