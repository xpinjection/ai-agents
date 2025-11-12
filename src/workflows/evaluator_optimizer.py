from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class CvReview(BaseModel):
    score: float = Field(..., description="Score from 0 to 1 how likely you would invite this candidate to an interview")
    feedback: str = Field(..., description="Feedback on the CV, what is good, what needs improvement, what skills are missing, what red flags, etc.")


class State(TypedDict):
    life_story: str
    job_description: str
    cv: str
    cv_review: CvReview
    tailored_cv: str
    review_cycles: int


class ReviewCyclesLimitExceededError(Exception): pass


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)

review_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
).with_structured_output(CvReview)


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
    return {"cv": response.content}


def cv_review(state: State):
    """Reviews a CV according to specific instructions, gives feedback and a score"""

    review_cv_message = """
    You are the hiring manager for specified job description.
    Your review applicant CVs and need to decide who of the many applicants you invite for an on-site interview.
    You give each CV a score and feedback (both the good and the bad things).
    You can ignore things like missing address and placeholders.       
    IMPORTANT: Return your response as valid JSON only, without any markdown formatting or code blocks.

    Job description:            :            

    ------
    {job_description}
    ------

    Candidate CV:

    ------
    {cv}
    ------
    """

    response = review_model.invoke(review_cv_message.format(job_description=state["job_description"],
                                                            cv=state["cv"]))
    return {"cv_review": response}


def tailor_cv(state: State):
    """Tailors a CV according to specific instructions and feedback"""

    tailor_cv_message = """
    Here is a CV that needs tailoring to a specific job description, feedback or other instruction.
    You can make the CV look good to meet the requirements, but don't invent facts.
    You can drop irrelevant things if it makes the CV better suited to the instructions.
    The goal is that the applicant gets an interview and can then live up to the CV.
    
    Also you have optional feedback for tailoring the CV:
    (Again, do not invent facts that are not part of the original CV. If the applicant is not suitable, 
    highlight his existing features that match most closely, but do not make up facts)
    
    The current CV: 

    ------
    {cv}
    ------

    Review feedback:

    ------
    {cv_feedback}
    ------
    """

    review_cycles = state["review_cycles"]
    if review_cycles >= 5:
        raise ReviewCyclesLimitExceededError(f"The review cycles limit has been exceeded: {review_cycles}")
    response = model.invoke(tailor_cv_message.format(cv=state["cv"], cv_feedback=state["cv_review"].feedback))
    return {"cv": response.content, "review_cycles": review_cycles + 1}


def route_review(state: State):
    """Regenerate until the review score is high enough"""
    if state["cv_review"].score > 0.8:
        return "Accepted"
    return "Regenerate"


workflow = StateGraph(State)

workflow.add_node("generate_cv", generate_cv)
workflow.add_node("tailor_cv", tailor_cv)
workflow.add_node("cv_review", cv_review)

workflow.add_edge(START, "generate_cv")
workflow.add_edge("generate_cv", "cv_review")
workflow.add_conditional_edges(
    "cv_review",
    route_review,
    {
        "Accepted": END,
        "Regenerate": "tailor_cv"
    }
)
workflow.add_edge("tailor_cv", "cv_review")

evaluator_optimizer_agent = workflow.compile()

if __name__ == '__main__':
    life_story = Path("./docs/user_life_story.txt").read_text(encoding="utf-8")
    job_description = Path("./docs/job_description_backend.txt").read_text(encoding="utf-8")

    result = evaluator_optimizer_agent.invoke({
        "life_story": life_story,
        "job_description": job_description,
        "review_cycles": 0,
    })

    print(result["cv_review"].score)
    print(result["cv"])
