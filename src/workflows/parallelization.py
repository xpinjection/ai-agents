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
    cv: str
    job_description: str
    hr_requirements: str
    phone_interview_notes: str
    hr_review: CvReview
    manager_review: CvReview
    team_member_review: CvReview
    summary_review: CvReview


review_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
).with_structured_output(CvReview)


def hr_cv_review(state: State):
    """Reviews a CV to check if a candidate fits HR requirements, gives feedback and a score"""

    review_cv_message = """
    You are working for HR and review CVs to fill a position with predefined requirements.
    You give each CV a score and feedback (both the good and the bad things).
    You can ignore things like missing address and placeholders.            
    IMPORTANT: Return your response as valid JSON only, without any markdown formatting or code blocks.
    
    Requirements:            
             
    ------
    {requirements}
    ------
    
    Candidate CV:
    
    ------
    {cv}
    ------
    
    Phone interview notes:
    
    ------
    {phone_interview_notes}
    ------
    """

    response = review_model.invoke(review_cv_message.format(requirements=state["hr_requirements"],
                                                            cv=state["cv"],
                                                            phone_interview_notes=state["phone_interview_notes"]))
    return {"hr_review": response}


def manager_cv_review(state: State):
    """Reviews a CV based on a job description, gives feedback and a score"""

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
    return {"manager_review": response}


def team_member_cv_review(state: State):
    """Reviews a CV to see if a candidate fits in the team, gives feedback and a score"""

    review_cv_message = """
    You work in a team with motivated, self-driven colleagues and a lot of freedom.
    Your team values collaboration, responsibility and pragmatism.
    Your review applicant CVs and need to decide how well this person will fit in your team.
    You give each CV a score and feedback (both the good and the bad things).
    You can ignore things like missing address and placeholders.       
    IMPORTANT: Return your response as valid JSON only, without any markdown formatting or code blocks.

    Candidate CV:

    ------
    {cv}
    ------
    """

    response = review_model.invoke(review_cv_message.format(cv=state["cv"]))
    return {"team_member_review": response}


def summarize_review(state: State):
    feedback = "\n".join([
        f"HR Review: {state["hr_review"].feedback}",
        f"Manager Review: {state["manager_review"].feedback}",
        f"Team Member Review: {state["team_member_review"].feedback}"
    ])
    avg_score = (state["hr_review"].score + state["manager_review"].score + state["team_member_review"].score) / 3.0
    return {"summary_review": CvReview(score=avg_score, feedback=feedback)}


workflow = StateGraph(State)

workflow.add_node("hr_review", hr_cv_review)
workflow.add_node("manager_review", manager_cv_review)
workflow.add_node("team_member_review", team_member_cv_review)
workflow.add_node("review_summarizer", summarize_review)

workflow.add_edge(START, "hr_review")
workflow.add_edge(START, "manager_review")
workflow.add_edge(START, "team_member_review")
workflow.add_edge("hr_review", "review_summarizer")
workflow.add_edge("manager_review", "review_summarizer")
workflow.add_edge("team_member_review", "review_summarizer")
workflow.add_edge("review_summarizer", END)

parallel_agent = workflow.compile()

if __name__ == '__main__':
    job_description = Path("./docs/job_description_backend.txt").read_text(encoding="utf-8")
    cv = Path("./docs/tailored_cv.txt").read_text(encoding="utf-8")
    hr_requirements = Path("./docs/hr_requirements.txt").read_text(encoding="utf-8")
    phone_interview_notes = Path("./docs/phone_interview_notes.txt").read_text(encoding="utf-8")

    result = parallel_agent.invoke({
        "cv": cv,
        "job_description": job_description,
        "hr_requirements": hr_requirements,
        "phone_interview_notes": phone_interview_notes,
    })

    print(result["summary_review"])
