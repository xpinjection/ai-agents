"""CLI entry point for Flight Booking A2A Agent Server."""

import os

import click
import uvicorn
from dotenv import load_dotenv

from flights.a2a.server import create_app


@click.command()
@click.option(
    '--host',
    default='localhost',
    help='Host to bind the server to'
)
@click.option(
    '--port',
    default=9999,
    type=int,
    help='Port to run the server on'
)
def main(host: str, port: int):
    """Launch Flight Booking A2A Agent Server.

    This server exposes the flight booking agent via the A2A protocol,
    enabling agent-to-agent communication and interoperability.

    Example:
        uv run python -m src.flights.a2a
        uv run python -m src.flights.a2a --host 0.0.0.0 --port 8080
    """
    # Load environment variables from .env file
    load_dotenv()

    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY is required to run the flight agent")

    print(f"Starting Flight A2A Agent on {host}:{port}")
    print("Press CTRL+C to stop the server")

    # Create application
    app = create_app(host=host, port=port)

    # Run with uvicorn
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
