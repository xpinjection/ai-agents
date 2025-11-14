from typing import Literal

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")


embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="conventions",
    embedding_function=embeddings_model,
    persist_directory="./src/conventions/chroma_db",
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

retriever_tool = create_retriever_tool(retriever, "search_conventions",
                                       "Search and return information about existing conventions.")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
).bind_tools([retriever_tool])

query_rewrite_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

grader_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
).with_structured_output(GradeDocuments)

ASSISTANT_MESSAGE = """
You are an assistant that answers questions based on internal API conventions.
You always try to find corresponding information in the API conventions before answering user query.
Answer to user only you have all required information from API conventions in the context.
Use ONLY the information provided in the API conventions to answer. 
If the answer is not contained in the API conventions, reply exactly: 'I don't know based on the existing conventions.'
Avoid statements like "Based on the context..." or "The provided information...".
Try to explain the answer based on the rules described in the API conventions.
"""

ANSWER_MESSAGE = """
You are an assistant that answers questions based on internal API conventions.
Use ONLY the information provided in the API conventions to answer. 
If the answer is not contained in the API conventions, reply exactly: 'I don't know based on the existing conventions.'
Avoid statements like "Based on the context..." or "The provided information...".
Try to explain the answer based on the rules described in the API conventions.
API conventions:
-------
{conventions}
-------
Here is the user question: 
-------
{question}
-------
"""

REWRITE_PROMPT = """
Look at the user query related to API conventions and try improve it to expand the underlying semantic intent / meaning.
Returned ONLY an improved version of the user query without any explanations and comments.
Here is the initial question:
-------
{question}
-------
"""

GRADE_PROMPT = """
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Here is the retrieved document:
------- 
{context}
------- 
Here is the user question: 
-------
{question}
-------
"""


def generate_query_or_respond(state: MessagesState):
    """
    Call the model to generate a response based on the current state.
    Given the question, it will decide to retrieve using the retriever tool or respond to the user.
    """
    response = model.invoke([SystemMessage(ASSISTANT_MESSAGE)] + state["messages"])
    return {"messages": [response]}


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].text
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.invoke([HumanMessage(prompt)])

    if response.binary_score == "yes":
        return "generate_answer"
    return "rewrite_question"


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].text
    response = query_rewrite_model.invoke([HumanMessage(REWRITE_PROMPT.format(question=question))])
    return {"messages": [HumanMessage(content=response.text)]}


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].text
    context = state["messages"][-1].content
    prompt = ANSWER_MESSAGE.format(question=question, conventions=context)
    response = model.invoke([HumanMessage(prompt)])
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

conventions_agentic_assistant = workflow.compile()
