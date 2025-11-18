import pytest
from agentevals.trajectory import create_trajectory_match_evaluator
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from flights.flight_assistant import create_flight_assistant

@pytest.fixture
def flight_assistant():
    checkpointer = MemorySaver()
    return create_flight_assistant(checkpointer)


@pytest.fixture
def evaluator():
    return create_trajectory_match_evaluator(
        trajectory_match_mode="strict",
        tool_args_match_mode="exact",
    )

def test_flight_search(flight_assistant, evaluator):
    config = {"configurable": {"thread_id": "1"}}
    result = flight_assistant.invoke(
        {"messages": [HumanMessage("What flights are available from Warsaw to Krakow?")]},
        config=config,
    )

    expected_ai_message = """
    I found one flight:

        Flight code: LO123
        Route: Warsaw → Krakow
        Departure: 2026-01-05 at 09:30
        Price: 199.99
        Seats available: 5

    Would you like to book this flight, see full flight details, or search other dates/routes? 
    If you want to book, please provide each passenger’s first name, last name, and date of birth (YYYY-MM-DD).
    """

    reference_trajectory = [
        HumanMessage(content="What flights are available from Warsaw to Krakow?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "list_flights",
                    "args": {"departure": "Warsaw", "destination": "Krakow"}
                }
            ]
        ),
        ToolMessage(
            content="flights=[FlightData(code='LO123', departure='WARSAW', destination='KRAKOW', departure_time=datetime.datetime(2026, 1, 5, 9, 30), ticket_price=Decimal('199.99'), capacity=5, available_capacity=5)]",
            tool_call_id="call_1"
        ),
        AIMessage(content=expected_ai_message),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory
    )

    assert evaluation["score"] is True
