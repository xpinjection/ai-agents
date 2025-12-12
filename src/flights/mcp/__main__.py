"""Entry point for running the Flight Booking MCP server."""
import uvicorn
from starlette.middleware.cors import CORSMiddleware

from flights.mcp.server import mcp

if __name__ == "__main__":
    # Run server with stdio transport (default for MCP)
    mcp.run(transport="streamable-http")
    app = mcp.streamable_http_app()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:6274", "http://127.0.0.1:6274"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"]
    )
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
