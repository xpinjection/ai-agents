from datetime import date
from datetime import datetime
from decimal import Decimal
from typing import List, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, ToolException
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import BaseModel, Field

from flights.flight_service import Flight, FlightInfo, Passenger, Booking, FlightNotFoundError, \
    InputValidationError, OverCapacityError, BookingNotFoundError, FlightBookingService


class ListFlightsInput(BaseModel):
    """Input for list flights tool."""
    departure: str = Field(description="Departure city. Only full city name is allowed.")
    destination: str = Field(
        description="Destination city. Only full city name is allowed. Must be different from departure.")


class GetFlightInput(BaseModel):
    """Input for get flight tool."""
    code: str = Field(description="Flight unique code")


class PassengerInfo(BaseModel):
    """Passenger info for booking."""
    first_name: str = Field(description="Passenger first name")
    last_name: str = Field(description="Passenger last name")
    date_of_birth: date = Field(description="Date of birth in ISO format YYYY-MM-DD")

    def to_domain(self) -> Passenger:
        return Passenger(
            first_name=self.first_name,
            last_name=self.last_name,
            date_of_birth=self.date_of_birth,
        )


class BookTicketsInput(BaseModel):
    """Input for book tickets tool."""
    flight_code: str = Field(description="Flight unique code to book")
    passengers: List[PassengerInfo] = Field(description="List of passengers")


class FindBookingInput(BaseModel):
    """Input for find booking tool."""
    booking_id: str = Field(description="Booking identifier (UUID)")


class CancelBookingInput(BaseModel):
    """Input for cancel booking tool."""
    booking_id: str = Field(description="Booking identifier (UUID)")


class FlightData(BaseModel):
    code: str
    departure: str
    destination: str
    departure_time: datetime
    ticket_price: Decimal
    capacity: int
    available_capacity: int

    @classmethod
    def from_domain(cls, flight_info: FlightInfo):
        flight = flight_info.flight
        return cls(
            code=flight.code,
            departure=flight.departure,
            destination=flight.destination,
            departure_time=flight.departure_time,
            ticket_price=flight.ticket_price,
            capacity=flight.capacity,
            available_capacity=flight_info.available_capacity,
        )


class PassengerData(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: date

    @classmethod
    def from_domain(cls, passenger: Passenger):
        return cls(
            first_name=passenger.first_name,
            last_name=passenger.last_name,
            date_of_birth=passenger.date_of_birth,
        )


class BookingData(BaseModel):
    id: str
    flight_code: str
    status: str
    booked_at: datetime
    total_price: Decimal
    passengers: List[PassengerData]

    @classmethod
    def from_domain(cls, booking: Booking):
        return cls(
            id=booking.id,
            flight_code=booking.flight_code,
            status=booking.status.value,
            booked_at=booking.booked_at,
            total_price=booking.total_price,
            passengers=[
                PassengerData.from_domain(passenger)
                for passenger in booking.passengers
            ],
        )


class FlightsData(BaseModel):
    flights: List[FlightData]


available_flights = [
    Flight(
        code="LO123", departure="WARSAW",
        destination="KRAKOW",
        departure_time=datetime(2026, 1, 5, 9, 30),
        ticket_price=Decimal("199.99"),
        capacity=5,
    ),
    Flight(
        code="LO456",
        departure="WARSAW",
        destination="GDANSK",
        departure_time=datetime(2026, 1, 5, 13, 15),
        ticket_price=Decimal("159.50"),
        capacity=3,
    ),
    Flight(
        code="LO789",
        departure="KRAKOW",
        destination="WARSAW",
        departure_time=datetime(2026, 1, 6, 18, 45),
        ticket_price=Decimal("189.00"),
        capacity=4,
    ),
]

flight_service = FlightBookingService(available_flights)


@tool(
    args_schema=ListFlightsInput,
    description="List flights between departure and destination cities.",
)
def list_flights(departure: str, destination: str) -> FlightsData:
    flights = flight_service.list_flights(departure, destination)
    return FlightsData(flights=[FlightData.from_domain(flight) for flight in flights])


@tool(
    args_schema=GetFlightInput,
    description="Get a single flight details by flight code including current available capacity and ticket price.",
)
def get_flight(code: str) -> FlightInfo:
    try:
        return flight_service.get_flight(code)
    except FlightNotFoundError as e:
        raise ToolException(str(e))


@tool(
    args_schema=BookTicketsInput,
    description=(
            "Book tickets for a flight for list of passengers (name, surname, date of birth). "
            "Validates passengers and capacity. On success returns full information about the booking."
    ),
)
def book_tickets(flight_code: str, passengers: List[PassengerInfo]) -> BookingData:
    try:
        booking = flight_service.book_tickets(flight_code, [p.to_domain() for p in passengers])
        return BookingData.from_domain(booking)
    except InputValidationError as e:
        raise ToolException(str(e))
    except OverCapacityError as e:
        raise ToolException(str(e))
    except FlightNotFoundError as e:
        raise ToolException(str(e))


@tool(
    args_schema=FindBookingInput,
    description="Find tickets booking by id and return full booking info including status.",
)
def find_booking(booking_id: str) -> BookingData:
    try:
        booking = flight_service.find_booking(booking_id)
        return BookingData.from_domain(booking)
    except BookingNotFoundError as e:
        raise ToolException(str(e))


@tool(
    args_schema=CancelBookingInput,
    description="Cancel booking by id. Returns the updated booking with status CANCELLED.",
)
def cancel_booking(booking_id: str) -> BookingData:
    try:
        booking = flight_service.cancel_booking(booking_id)
        return BookingData.from_domain(booking)
    except BookingNotFoundError as e:
        raise ToolException(str(e))


class ToolErrorHandlerMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request: ToolCallRequest,
                       handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        try:
            return handler(request)
        except ToolException as e:
            return self._build_tool_error_message(request, e)

    async def awrap_tool_call(self, request: ToolCallRequest,
                              handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        try:
            return await handler(request)
        except ToolException as e:
            return self._build_tool_error_message(request, e)

    def _build_tool_error_message(self, request: ToolCallRequest, e: ToolException) -> ToolMessage:
        return ToolMessage(
            content=f"ERROR calling tool {request.tool.name}: {e}",
            tool_call_id=request.tool_call["id"],
            name=request.tool.name,
            status="error",
        )


SYSTEM_PROMPT = """
You are a helpful and concise flight booking assistant. 
Use available tools to help users find flights and manage their ticket bookings. 
When you receive structured data from any tool, analyze and summarize the most important points in a concise, 
user-friendly summary.

**Detailed instructions:**
- If user asks for an operation that's unavailable via tools, say so politely. 
- If user requests info that's unavailable via tool, say so politely.
- Pay attention to the tool parameters format and ask user for any missed information before calling the tool.
- Explain all errors in a human friendly way, don't use reference to the system or tools in your responses.
"""

model = ChatOpenAI(
    model="gpt-5-mini",
)

flight_assistant = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[list_flights, get_flight, book_tickets, find_booking, cancel_booking],
    middleware=[ToolErrorHandlerMiddleware()],
)
