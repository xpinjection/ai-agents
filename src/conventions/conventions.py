from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

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

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)

query_rewrite_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

SYSTEM_MESSAGE = """
You are an assistant that answers questions about internal API conventions.
Use ONLY the information provided in the API conventions to answer. 
If the answer is not contained in the API conventions, reply exactly: 'I don't know based on the existing conventions.'
Avoid statements like "Based on the context..." or "The provided information...".
Try to explain the answer based on the rules described in the API conventions.

API conventions:
-------
{conventions}
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


@dynamic_prompt
async def dynamic_system_prompt(request: ModelRequest) -> str:
    user_query = request.state["messages"][-1].text
    response = await query_rewrite_model.ainvoke(REWRITE_PROMPT.format(question=user_query))
    improved_query = response.content
    print(f"Improved query: {improved_query}")
    retrieved_docs = await retriever.ainvoke(improved_query)
    print(f"Found documents: {retrieved_docs}")
    conventions = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return SYSTEM_MESSAGE.format(conventions=conventions)


conventions_assistant = create_agent(
    model=model,
    middleware=[dynamic_system_prompt],
)
