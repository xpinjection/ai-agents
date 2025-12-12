"""FastMCP server exposing Flight Booking Assistant as MCP tools."""

import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from flights.flight_assistant import (
    create_flight_assistant,
    flight_service,
    FlightsData,
    FlightData,
    BookingData,
    PassengerInfo,
    FlightNotFoundError,
    BookingNotFoundError,
    InputValidationError,
    OverCapacityError,
)

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("Flight Booking Assistant")

# Initialize LangGraph agent with checkpointer for conversation state
_checkpointer = MemorySaver()
_agent = create_flight_assistant(checkpointer=_checkpointer)


# Utilities

def get_thread_id(thread_id: Optional[str] = None) -> str:
    """Return provided thread_id or generate new UUID."""
    return thread_id if thread_id else str(uuid.uuid4())


# Direct MCP Tools - Direct mapping to flight service operations

@mcp.tool(
    description="Search for available flights between two cities. Returns list of flights with details including departure times, prices, and available capacity.")
async def search_flights(
        departure: str,
        destination: str,
        ctx: Context
) -> dict:
    """Search for available flights between departure and destination cities.

    Args:
        departure: Departure city (full city name, e.g., "WARSAW")
        destination: Destination city (full city name, e.g., "KRAKOW")
        ctx: FastMCP context for logging

    Returns:
        Dict with flight list and success status
    """
    try:
        await ctx.info(f"Searching flights from {departure} to {destination}")
        flights = flight_service.list_flights(departure.upper(), destination.upper())
        flights_data = FlightsData(flights=[FlightData.from_domain(f) for f in flights])

        await ctx.info(f"Found {len(flights_data.flights)} flights")
        return {
            "success": True,
            "flights": [f.model_dump() for f in flights_data.flights]
        }
    except Exception as e:
        await ctx.error(f"Error searching flights: {str(e)}")
        raise ToolError(f"Failed to search flights: {str(e)}")


@mcp.tool(
    description="Get detailed information about a specific flight by flight code. Returns flight details including current available capacity and ticket price.")
async def get_flight_details(
        flight_code: str,
        ctx: Context
) -> dict:
    """Get detailed information about a specific flight.

    Args:
        flight_code: Flight unique code (e.g., "LO123")
        ctx: FastMCP context for logging

    Returns:
        Dict with flight details and success status
    """
    try:
        await ctx.info(f"Getting details for flight {flight_code}")
        flight_info = flight_service.get_flight(flight_code.upper())
        flight_data = FlightData.from_domain(flight_info)

        return {
            "success": True,
            "flight": flight_data.model_dump()
        }
    except FlightNotFoundError as e:
        await ctx.error(f"Flight not found: {flight_code}")
        raise ToolError(f"Flight not found: {flight_code}")
    except Exception as e:
        await ctx.error(f"Error getting flight details: {str(e)}")
        raise ToolError(f"Failed to get flight details: {str(e)}")


@mcp.tool(
    description="Book tickets for a flight with passenger information. Validates passenger details and capacity before confirming. Requires flight code and list of passengers with first_name, last_name, and date_of_birth (YYYY-MM-DD).")
async def book_flight_tickets(
        flight_code: str,
        passengers: List[dict],
        ctx: Context
) -> dict:
    """Book tickets for a flight with passenger information.

    Args:
        flight_code: Flight unique code to book (e.g., "LO123")
        passengers: List of passenger dicts with keys: first_name, last_name, date_of_birth (YYYY-MM-DD)
        ctx: FastMCP context for logging

    Returns:
        Dict with booking confirmation and success status
    """
    try:
        await ctx.info(f"Booking flight {flight_code} for {len(passengers)} passenger(s)")

        # Parse and validate passengers using existing Pydantic model
        passenger_infos = [PassengerInfo(**p) for p in passengers]

        # Book tickets
        booking = flight_service.book_tickets(
            flight_code.upper(),
            [p.to_domain() for p in passenger_infos]
        )

        booking_data = BookingData.from_domain(booking)

        await ctx.info(f"Booking successful: {booking.id}")
        return {
            "success": True,
            "booking": booking_data.model_dump()
        }
    except InputValidationError as e:
        await ctx.error(f"Invalid booking data: {str(e)}")
        raise ToolError(f"Invalid booking data: {str(e)}")
    except OverCapacityError as e:
        await ctx.error(f"Flight over capacity: {str(e)}")
        raise ToolError(f"Flight over capacity: {str(e)}")
    except FlightNotFoundError as e:
        await ctx.error(f"Flight not found: {flight_code}")
        raise ToolError(f"Flight not found: {flight_code}")
    except Exception as e:
        await ctx.error(f"Error booking tickets: {str(e)}")
        raise ToolError(f"Failed to book tickets: {str(e)}")


@mcp.tool(
    description="Retrieve booking information by booking ID. Returns full booking details including status, passengers, and total price.")
async def get_booking_details(
        booking_id: str,
        ctx: Context
) -> dict:
    """Get booking details by booking ID.

    Args:
        booking_id: Booking identifier (UUID)
        ctx: FastMCP context for logging

    Returns:
        Dict with booking details and success status
    """
    try:
        await ctx.info(f"Retrieving booking {booking_id}")
        booking = flight_service.find_booking(booking_id)
        booking_data = BookingData.from_domain(booking)

        return {
            "success": True,
            "booking": booking_data.model_dump()
        }
    except BookingNotFoundError as e:
        await ctx.error(f"Booking not found: {booking_id}")
        raise ToolError(f"Booking not found: {booking_id}")
    except Exception as e:
        await ctx.error(f"Error retrieving booking: {str(e)}")
        raise ToolError(f"Failed to retrieve booking: {str(e)}")


@mcp.tool(
    description="Cancel an existing flight booking by booking ID. Returns the updated booking with status CANCELLED.")
async def cancel_flight_booking(
        booking_id: str,
        ctx: Context
) -> dict:
    """Cancel a flight booking.

    Args:
        booking_id: Booking identifier (UUID)
        ctx: FastMCP context for logging

    Returns:
        Dict with cancellation confirmation and success status
    """
    try:
        await ctx.info(f"Cancelling booking {booking_id}")
        booking = flight_service.cancel_booking(booking_id)
        booking_data = BookingData.from_domain(booking)

        await ctx.info(f"Booking {booking_id} cancelled successfully")
        return {
            "success": True,
            "booking": booking_data.model_dump()
        }
    except BookingNotFoundError as e:
        await ctx.error(f"Booking not found: {booking_id}")
        raise ToolError(f"Booking not found: {booking_id}")
    except Exception as e:
        await ctx.error(f"Error cancelling booking: {str(e)}")
        raise ToolError(f"Failed to cancel booking: {str(e)}")


# Conversational Tool - Natural language interface

@mcp.tool(
    description="Have a natural language conversation with the flight booking assistant. The assistant can help search flights, book tickets, manage bookings, and answer questions. Supports multi-turn conversations using thread_id.")
async def chat_with_agent(
        message: str,
        ctx: Context,
        thread_id: Optional[str] = None
) -> dict:
    """Chat with the flight booking assistant using natural language.

    Args:
        message: Your message or question for the assistant
        ctx: FastMCP context for logging
        thread_id: Optional conversation thread ID for multi-turn conversations (generated if not provided)

    Returns:
        Dict with assistant response and thread_id for continuing the conversation
    """
    # Get or generate thread ID
    current_thread_id = get_thread_id(thread_id)

    try:
        await ctx.info(f"Processing message in thread {current_thread_id}")

        # Configure thread for conversation state
        config = {"configurable": {"thread_id": current_thread_id}}

        # Invoke agent with message
        result = await _agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )

        # Extract the final AI message
        messages = result.get("messages", [])
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]

        if ai_messages:
            response_content = ai_messages[-1].content
            await ctx.info("Agent response generated")
        else:
            response_content = "No response generated"
            await ctx.warning("No AI message in response")

        return {
            "success": True,
            "response": response_content,
            "thread_id": current_thread_id
        }

    except Exception as e:
        await ctx.error(f"Error in chat: {str(e)}")
        raise ToolError(f"Failed to process chat message: {str(e)}")
