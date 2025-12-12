"""Agent card definition for Flight Booking A2A agent."""

from a2a.types import AgentCard, AgentSkill, AgentProvider, AgentCapabilities


def create_agent_card(host: str = "localhost", port: int = 9999) -> AgentCard:
    """Create agent card for flight booking agent.

    Args:
        host: Server hostname
        port: Server port number

    Returns:
        AgentCard with flight booking agent metadata
    """
    base_url = f"http://{host}:{port}"

    return AgentCard(
        name="Flight Booking Agent",
        description=(
            "A helpful flight booking assistant that can search flights, "
            "book tickets, and manage bookings. Completes tasks in a single "
            "interaction with all necessary information provided."
        ),
        version="1.0.0",
        protocol_version="0.3.0",
        url=base_url,

        # Agent provider information
        provider=AgentProvider(
            organization="AI Agents Demo",
            url="https://github.com/yourusername/ai-agents"
        ),

        # Skills definition
        skills=[
            AgentSkill(
                id="flight_search",
                name="Flight Search",
                description=(
                    "Search for available flights between cities. "
                    "Returns flight details including departure times, "
                    "prices, and available capacity."
                ),
                tags=["flights", "search", "travel"],
                examples=[
                    "Find flights from Warsaw to Krakow",
                    "Show me flights to Gdansk",
                    "What flights are available from Krakow to Warsaw?"
                ]
            ),
            AgentSkill(
                id="flight_booking",
                name="Flight Booking",
                description=(
                    "Book tickets for flights with passenger information. "
                    "Validates passenger details and capacity before confirming. "
                    "Requires flight code and passenger details (name, DOB)."
                ),
                tags=["booking", "tickets", "reservation"],
                examples=[
                    "Book flight LO123 for 2 passengers",
                    "I want to book tickets on flight LO456",
                    "Reserve seats for John Smith born 1990-01-15"
                ]
            ),
            AgentSkill(
                id="booking_management",
                name="Booking Management",
                description=(
                    "Find and cancel existing bookings by booking ID. "
                    "Retrieves booking details including status, passengers, "
                    "and total price."
                ),
                tags=["booking", "cancellation", "management"],
                examples=[
                    "Find my booking with ID abc-123",
                    "Show booking details for xyz-456",
                    "Cancel my booking def-789"
                ]
            ),
            AgentSkill(
                id="flight_details",
                name="Flight Details",
                description=(
                    "Get detailed information about a specific flight by code. "
                    "Shows current availability, pricing, and departure time."
                ),
                tags=["flights", "information", "details"],
                examples=[
                    "Tell me about flight LO123",
                    "What's the capacity of flight LO456?",
                    "Show details for flight code LO789"
                ]
            )
        ],

        # Capabilities
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=True
        ),

        # Supported content types
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"]
    )
