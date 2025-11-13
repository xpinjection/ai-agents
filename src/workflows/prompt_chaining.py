from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class InputState(TypedDict):
    life_story: str
    instructions: str


class OutputState(TypedDict):
    tailored_cv: str


class State(InputState, OutputState):
    master_cv: str


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)


def generate_cv(state: InputState):
    """Generates a clean CV based on user-provided information"""

    generate_cv_message = """
    Here is information on my life and professional trajectory that you should turn into a clean and complete CV.
    Do not invent facts and do not leave out skills or experiences.
    This CV will later be cleaned up, for now, make sure it is complete.
    Return only the CV, no other comments or explanations.
    My life story: 
    
    ------
    {life_story}
    ------
    """

    response = model.invoke(generate_cv_message.format(life_story=state["life_story"]))
    return {"master_cv": response.content}


def tailor_cv(state: State):
    """Tailors a CV according to specific instructions"""

    tailor_cv_message = """
    Here is a CV that needs tailoring to a specific job description, feedback or other instruction.
    You can make the CV look good to meet the requirements, but don't invent facts.
    You can drop irrelevant things if it makes the CV better suited to the instructions.
    The goal is that the applicant gets an interview and can then live up to the CV.
    Return only the CV, no other comments or explanations.
    The master CV: 
    
    ------
    {master_cv}
    ------
    
    The instructions:
    
    ------
    {instructions}
    ------
    """

    response = model.invoke(tailor_cv_message.format(master_cv=state["master_cv"],
                                                     instructions=state["instructions"]))
    return {"tailored_cv": response.content}


workflow = StateGraph(State, input_schema=InputState, output_schema=OutputState)

workflow.add_node(generate_cv)
workflow.add_node(tailor_cv)

workflow.add_edge(START, "generate_cv")
workflow.add_edge("generate_cv", "tailor_cv")
workflow.add_edge("tailor_cv", END)

prompt_chain_agent = workflow.compile()

if __name__ == '__main__':
    life_story = Path("./docs/user_life_story.txt").read_text(encoding="utf-8")
    job_description = Path("./docs/job_description_backend.txt").read_text(encoding="utf-8")
    instructions = f"Adapt the CV to the job description below: {job_description}"

    agent = workflow.compile(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    result = agent.invoke({
        "life_story": life_story,
        "instructions": instructions,
    }, config=config, durability="sync")
    print(result["tailored_cv"])

    states = list(agent.get_state_history(config))

    for state in states:
        print(state)
        print()
