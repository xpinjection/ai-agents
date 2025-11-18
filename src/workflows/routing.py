from datetime import date
from pathlib import Path
from typing import List, Literal

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class CvReview(TypedDict):
    score: float
    feedback: str


class State(MessagesState):
    cv_review: CvReview
    candidate_contact: str
    job_description: str
    decision: str


@tool
def get_current_date() -> str:
    """
    Returns the current system date in ISO format.

    This tool can be used by the agent whenever it needs to include
    the current date in its reasoning, planning, or communication.

    Returns:
        str: Current date in ISO format (YYYY-MM-DD).
    """
    return date.today().isoformat()


class GetInvolvedEmployeesForInterviewArgs(BaseModel):
    job_description_id: str = Field(..., description="Unique job description ID for which to find employees involved in the onsite interview process.")


@tool(args_schema=GetInvolvedEmployeesForInterviewArgs)
def get_involved_employees_for_interview(job_description_id: str) -> List[str]:
    """
    Finds the email addresses and names of employees who should attend
    the onsite interview for a given job description.

    Returns:
        List[str]: List of employees in the format '<Name>: <email>'.
    """
    return [
        "Anna Bolena: hiring.manager@company.com",
        "Chris Durue: near.colleague@company.com",
        "Esther Finnigan: vp@company.com"
    ]


class CreateCalendarEntryArgs(BaseModel):
    email_addresses: List[str] = Field(..., description="List of employee email addresses for whom to create calendar entries.")
    topic: str = Field(..., description="Meeting topic or subject to appear in the calendar.")
    start_time: str = Field(..., description="Start date and time in format 'YYYY-MM-DD HH:MM'.")
    end_time: str = Field(..., description="End date and time in format 'YYYY-MM-DD HH:MM'.")


@tool(args_schema=CreateCalendarEntryArgs)
def create_calendar_entry(email_addresses: List[str], topic: str, start_time: str, end_time: str) -> None:
    """
    Creates calendar entries for given employees based on their email addresses.

    Returns:
        None: This tool produces a side effect (calendar entry creation).
    """
    print("*** CALENDAR ENTRY CREATED ***")
    print(f"Topic: {topic}")
    print(f"Start: {start_time}")
    print(f"End: {end_time}")
    print(f"Participants: {', '.join(email_addresses)}")


class SendEmailArgs(BaseModel):
    to: List[str] = Field(..., description="List of recipient email addresses.")
    cc: List[str] = Field(default_factory=list, description="List of CC (carbon copy) email addresses. Optional.")
    subject: str = Field(..., description="Subject line of the email.")
    body: str = Field(..., description="Body content of the email message.")


@tool(args_schema=SendEmailArgs)
def send_email(to: List[str], cc: List[str], subject: str, body: str) -> None:
    """
    Sends an email to the specified recipients.

    Returns:
        None: This tool produces a side effect (email sending).
    """
    print("*** EMAIL SENT ***")
    print(f"To: {to}")
    print(f"Cc: {cc}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")


class UpdateApplicationStatusArgs(BaseModel):
    job_description_id: str = Field(..., description="Job description ID to which the candidate application belongs.")
    candidate_name: str = Field(..., description="Full candidate name (first and last).")
    new_status: str = Field(..., description="New status to set for the candidate's application (e.g. 'APPROVED', 'REJECTED', 'PENDING').")


@tool(args_schema=UpdateApplicationStatusArgs)
def update_application_status(job_description_id: str, candidate_name: str, new_status: str) -> None:
    """
    Updates the application status for a given candidate and job description.

    Returns:
        None: This tool produces a side effect (status update).
    """
    print("*** APPLICATION STATUS UPDATED ***")
    print(f"Job Description ID: {job_description_id}")
    print(f"Candidate Name: {candidate_name}")
    print(f"New Status: {new_status}")


tools = [get_current_date, get_involved_employees_for_interview, create_calendar_entry, send_email,
         update_application_status]

model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
).bind_tools(tools)


def organize_interview(state: State):
    """Organizes on-site interviews with applicants"""

    interview_message = """
    You organize on-site meetings by sending a calendar invite to all implied employees 
    for a 3h interview in one week from the current date, in the morning.
    You also invite the candidate with a congratulatory email and interview details.
    Lastly, you update the application status to 'invited on-site'.

    Candidate contact: {candidate_contact}
    
    Job description:            

    ------
    {job_description}
    ------
    """

    system_message = interview_message.format(job_description=state["job_description"],
                                              candidate_contact=state["candidate_contact"])
    response = model.invoke([system_message] + state["messages"])
    return {"messages": [response]}


def reject_candidate(state: State):
    """Sends rejection emails to candidates that didn't pass"""

    interview_message = """
    You send a kind email to application candidates that did not pass the first review round.
    You also update the application status to 'rejected'.

    Candidate contact: 
    
    ------
    {candidate_contact}
    ------

    Job description:            

    ------
    {job_description}
    ------
    """

    system_message = interview_message.format(job_description=state["job_description"],
                                              candidate_contact=state["candidate_contact"])
    response = model.invoke([system_message] + state["messages"])
    return {"messages": [response]}


def decision_router(state: State):
    """Decides whether to reject a candidate or organize an interview"""
    if state["cv_review"]["score"] < 0.8:
        return {"decision": "reject"}
    return {"decision": "interview"}


def route_decision(state: State) -> Literal["reject_candidate", "organize_interview"]:
    if state["decision"] == "reject":
        return "reject_candidate"
    elif state["decision"] == "interview":
        return "organize_interview"


workflow = StateGraph(State)

workflow.add_node(reject_candidate)
workflow.add_node("reject_candidate_tools", ToolNode(tools=tools))
workflow.add_node(organize_interview)
workflow.add_node("organize_interview_tools", ToolNode(tools=tools))
workflow.add_node(decision_router)

workflow.add_edge(START, "decision_router")
workflow.add_conditional_edges(
    "decision_router",
    route_decision,
)
workflow.add_conditional_edges(
    "reject_candidate",
    tools_condition,
    {"tools": "reject_candidate_tools", "__end__": END},
)
workflow.add_edge("reject_candidate_tools", "reject_candidate")
workflow.add_conditional_edges(
    "organize_interview",
    tools_condition,
    {"tools": "organize_interview_tools", "__end__": END},
)
workflow.add_edge("organize_interview_tools", "organize_interview")

routing_agent = workflow.compile()

if __name__ == '__main__':
    cv_review = CvReview(score=0.5, feedback="Good candidate")
    job_description = Path("./docs/job_description_backend.txt").read_text(encoding="utf-8")
    candidate_contact = """
    Candidate Contact Card:
    John Doe
    Rue des Carmes 12, 2000 Antwerp, Belgium
    john.doe.dev@protonmail.com
    +32 495 67 89 23
    """

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    routing_agent.invoke({
        "cv_review": cv_review,
        "job_description": job_description,
        "candidate_contact": candidate_contact
    }, config=config)

    cv_review["score"] = 0.9

    config: RunnableConfig = {"configurable": {"thread_id": "2"}}
    routing_agent.invoke({
        "cv_review": cv_review,
        "job_description": job_description,
        "candidate_contact": candidate_contact
    }, config=config)
