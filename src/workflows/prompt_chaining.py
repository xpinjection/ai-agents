from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    life_story: str
    instructions: str
    master_cv: str
    tailored_cv: str


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)


def generate_cv(state: State):
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


workflow = StateGraph(State)

workflow.add_node("generate_cv", generate_cv)
workflow.add_node("tailor_cv", tailor_cv)

workflow.add_edge(START, "generate_cv")
workflow.add_edge("generate_cv", "tailor_cv")
workflow.add_edge("tailor_cv", END)

prompt_chain_agent = workflow.compile()

if __name__ == '__main__':
    life_story = Path("./docs/user_life_story.txt").read_text(encoding="utf-8")
    job_description = Path("./docs/job_description_backend.txt").read_text(encoding="utf-8")
    instructions = f"Adapt the CV to the job description below: {job_description}"

    result = prompt_chain_agent.invoke({
        "life_story": life_story,
        "instructions": instructions,
    })
    print(result["tailored_cv"])
