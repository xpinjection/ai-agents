from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GenerateSqlQueryInput(BaseModel):
    """Input for generate SQL query tool."""
    user_task: str = Field(description="Description of the user task for which SQL query is needed.")


class TestSqlQueryInput(GenerateSqlQueryInput):
    """Input for test SQL query tool."""
    sql_query: str = Field(description="SQL query to be tested.")


db = SQLDatabase.from_uri("postgresql://root:123qwe@localhost:5432/ai")

sql_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

toolkit = SQLDatabaseToolkit(db=db, llm=sql_model)

db_tools = toolkit.get_tools()

get_db_schema_tool = next(tool for tool in db_tools if tool.name == "sql_db_schema")
get_db_tables_tool = next(tool for tool in db_tools if tool.name == "sql_db_list_tables")
sql_query_checker_tool = next(tool for tool in db_tools if tool.name == "sql_db_query_checker")
sql_executor_tool = next(tool for tool in db_tools if tool.name == "sql_db_query")

DBA_SYSTEM_PROMPT = """
Act as an experienced DBA with deep knowledge of {dialect} databases.
Given an input question, create a syntactically correct {dialect} query answer user question.
Unless the user specifies a specific number of results in the query, always limit your query to at most {top_k} results.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

Always verify the query syntax before returning it to the user.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".format(
    dialect=db.dialect,
    top_k=5,
)

dba_agent = create_agent(
    model=sql_model,
    system_prompt=DBA_SYSTEM_PROMPT,
    tools=[get_db_schema_tool, get_db_tables_tool, sql_query_checker_tool],
)


@tool(args_schema=GenerateSqlQueryInput, description="Generate SQL query by the user task. Returns SQL query.")
def generate_sql_query(user_task: str):
    result = dba_agent.invoke({
        "messages": [HumanMessage(f"Generate SQL query for the following user task: {user_task}")]
    })
    return result["messages"][-1].text


qa_model = model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="medium",
)

QA_ENGINEER_SYSTEM_PROMPT = """
Act as an QA Engineer with deep experience in testing SQL queries for {dialect} databases.
Given a user task, you create a set of test cases that can be used to validate the correctness of the SQL query.
For each test case you create a focused dataset to verify this test case on the provided SQL query.
Then you run all generated test cases and report the results.
If any errors found during testing report them to the user.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".format(
    dialect=db.dialect,
)

qa_agent = create_agent(
    model=qa_model,
    system_prompt=QA_ENGINEER_SYSTEM_PROMPT,
    tools=[get_db_tables_tool, sql_executor_tool],
)


@tool(args_schema=TestSqlQueryInput, description="Test SQL query according to the user task. Returns test results.")
def test_sql_query(user_task: str, sql_query: str):
    message = """
    Test the following SQL query for the following user task: {user_task}
    
    SQL Query: 
    ------
    {sql_query}
    ------
    """.format(user_task=user_task, sql_query=sql_query)

    result = qa_agent.invoke({
        "messages": [HumanMessage(message)]
    })
    return result["messages"][-1].text


SUPERVISOR_SYSTEM_PROMPT = """
You are a database assistant helping user to generate reliable and tested SQL queries for specific business needs.
Always use provided tools to generate and test SQL queries.
"""

supervisor_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

supervisor_assistant = create_agent(
    model=supervisor_model,
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    tools=[generate_sql_query, test_sql_query],
)

if __name__ == '__main__':
    for tool in db_tools:
        print(f"{tool.name}: {tool.description}\n")

    agent_response = supervisor_assistant.invoke(
        {"messages": [HumanMessage("Top merchant spending for last 3 months")]})
    print(agent_response)
