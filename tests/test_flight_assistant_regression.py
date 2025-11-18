import uuid

import pytest
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langsmith import evaluate
from openevals import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, CONCISENESS_PROMPT

from flights.flight_assistant import create_flight_assistant


@pytest.fixture
def flight_assistant():
    checkpointer = MemorySaver()
    return create_flight_assistant(checkpointer)


@pytest.fixture
def evaluators():
    judge_model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
    )

    system = """
    The task of the model under evaluation is to work as a helpful and concise flight booking assistant.
    It uses available tools to help users find flights and manage their ticket bookings. 
    Model should answer only on questions directly related to flights and tickets booking.
    """

    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        system=system,
        feedback_key="correctness",
        continuous=True,
        judge=judge_model,
    )

    conciseness_evaluator = create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        system=system,
        feedback_key="conciseness",
        continuous=True,
        judge=judge_model,
    )

    return [correctness_evaluator, conciseness_evaluator]


def test_flight_assistant_regression(flight_assistant, evaluators):
    checkpointer = MemorySaver()
    flight_assistant = create_flight_assistant(checkpointer=checkpointer)

    def run_agent(inputs: dict) -> dict:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        return flight_assistant.invoke(inputs, config=config)

    results = evaluate(
        run_agent,
        data="Flight assistant regression",
        evaluators=evaluators,
        experiment_prefix="Flight assistant regression experiment",
        max_concurrency=4,
    )

    assert results is not None
