from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
    use_responses_api=True,
)

web_tool = {"type": "web_search_preview"}
model_with_web_search = model.bind_tools([web_tool])

mcp_tool = {
    "type": "mcp",
    "server_label": "deepwiki",
    "server_url": "https://mcp.deepwiki.com/mcp",
    "require_approval": "never",
}
model_with_mcp = model.bind_tools([mcp_tool])

coding_tool = {
    "type": "code_interpreter",
    "container": {"type": "auto"},
}
model_with_coding = model.bind_tools([coding_tool])

if __name__ == "__main__":
    print("WEB SEARCH")
    web_response = model_with_web_search.invoke([HumanMessage("What is the latest version of Spring Boot and when it was released?")])
    print(web_response)

    print("MCP USAGE")
    mcp_response = model_with_mcp.invoke([HumanMessage("What transport protocols does the current version of the MCP spec (modelcontextprotocol/modelcontextprotocol) support?")])
    print(mcp_response)

    print("CODING")
    response = model_with_coding.invoke([HumanMessage("How to calculate Fibonacci sequence?")])
    print(response)
    coding_response = model_with_coding.invoke([HumanMessage("Generate code to calculate it for 5")],
                                               previous_response_id=response.id)
    print(coding_response)
