# Standard library imports
import os
import tempfile
import json
from typing import List

# Third-party imports
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from langchain_community.document_loaders import CSVLoader
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from oracle.tools import web_search, search_oracle_customer_references, search_oracle_marketplace, search_oracle_documentation
import uuid
from langchain_ollama import ChatOllama
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langchain_xai import ChatXAI

# Load environment variables
load_dotenv()
THREAD_ID = str(uuid.uuid4())
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
AUTH_TYPE = os.getenv("AUTH_TYPE")
CONFIG_PROFILE = os.getenv("CONFIG_PROFILE")
OCI_INFERENCE_ENDPOINT = os.getenv("OCI_INFERENCE_ENDPOINT")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

app = FastAPI()

tools = [web_search, search_oracle_customer_references, search_oracle_marketplace, search_oracle_documentation]

memory = InMemorySaver()

# Initialize embeddings
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint=OCI_INFERENCE_ENDPOINT,
    truncate="NONE",
    compartment_id=COMPARTMENT_ID,
    auth_type=AUTH_TYPE,
    auth_profile=CONFIG_PROFILE
)

supervisor_agent_llm = init_chat_model("gpt-4.1", max_tokens=4000, model_provider="openai")
discovery_agent_llm = init_chat_model("gpt-4.1", max_tokens=4000, model_provider="openai")
solutioning_agent_llm = init_chat_model("gpt-4.1", max_tokens=4000, model_provider="openai")

llm = init_chat_model("gpt-4.1", max_tokens=4000, model_provider="openai")
# llm = ChatXAI(
#     model="grok-3"
# )

# llm = ChatOllama(
#     model="qwen3:8b"
# )

# llm = ChatOCIGenAI(
#     model_id="cohere.command-a-03-2025",
#     #   model_id="cohere.command-r-08-2024",
#     # model_id="cohere.command-r-plus-08-2024",
#     service_endpoint=OCI_INFERENCE_ENDPOINT,
#     compartment_id=COMPARTMENT_ID,
#     is_stream=True,
#     model_kwargs={
#         "temperature": 1,
#         "max_tokens": 4000,
#         # # "frequency_penalty": 0,
#         # # "presence_penalty": 1,
#         # # "top_p": 0.75,
#         # "top_k": None,
#         # "seed": 1
#     },
#     auth_type=AUTH_TYPE,
#     auth_profile=CONFIG_PROFILE
# )

### AGENTS AND SUPERVISOR
### https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/

discovery_agent_prompt = (
    "You are DiscoveryAgent, an OCI Solutions Architect specialist focused solely on discovery.\n"
    "INSTRUCTIONS:\n"
    "- Apply reasoning to the user’s ask.\n"
    "- Understand the problem we’re solving; consider dependencies and constraints.\n"
    "- Ask up to three clarifying questions if needed.\n"
    "- Once discovery is complete, return control to the supervisor.\n"
)

discovery_architect_agent = create_react_agent(
    model=llm,#"openai:gpt-4.1",
    tools=[web_search, search_oracle_customer_references, search_oracle_marketplace, search_oracle_documentation],
    prompt=discovery_agent_prompt,
    name="discovery_architect_agent",
    checkpointer=memory
)

solutioning_agent_prompt = (
    "You are SolArchAgent, an OCI Solutions Architect.\n"
    "INSTRUCTIONS:\n"
    "- Analyze user and discovery-agent data for business goals, dependencies, and constraints.\n"
    "- Research potential architectures using experience, web search, and these sources:\n"
    "  - Oracle Marketplace (vendor, solution, URL)\n"
    "  - Oracle customer references (name, description, URL)\n"
    "  - Oracle public docs & wider web (name, description, URL)\n"
    "- Evaluate the top 1–3 solutions (list pros and cons).\n"
    "- Recommend one solution (justify your choice).\n"
    "- Propose next steps.\n"
    "- Provide a References section with all URLs.\n"
    "- When finished, return control to the supervisor.\n"
    "- Return only the generated text\n"
    "OUTPUT FORMAT:\n"
    "1. Evaluated Solutions: 1–3 solutions (pros & cons)\n"
    "2. Recommended Solution: justification\n"
    "3. Next Steps\n"
    "4. References\n"
)

solutioning_architect_agent = create_react_agent(
    model=llm,#"openai:gpt-4.1",
    tools=[web_search,search_oracle_customer_references, search_oracle_marketplace, search_oracle_documentation],
    prompt=solutioning_agent_prompt,
    name="solutioning_architect_agent",
    checkpointer=memory
)

supervisor_prompt = (
    "You are the Solutions Architecture Leader acting as a supervisor:\n"
    "INSTRUCTIONS:\n"
    "- You coordinate work by delegating tasks to the appropriate agent\n"
    "- Discovery Architect Agent: focuses on discovery and research\n"
    "- Solutioning Architect Agent: focuses on providing solutions\n"
)

supervisor = create_supervisor(
    model=llm,#init_chat_model("gpt-4.1"),
    agents=[discovery_architect_agent, solutioning_architect_agent],
    prompt=supervisor_prompt,
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile(checkpointer=memory)

def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": THREAD_ID}}
    result = supervisor.invoke(
            {
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    },
    config
    )
    
    return result["messages"][-1].content

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/query")
async def query(question: str):
    response = stream_graph_updates(question)
    # Convert the result to a string and clean up any list formatting

    return JSONResponse(content={
        "response": str(response)
    })

@app.post("/clear-memory")
async def clear_memory():
    try:
        global THREAD_ID
        THREAD_ID = str(uuid.uuid4())
        return JSONResponse(content={"message": "Conversation memory cleared successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app2:app", host="0.0.0.0", port=8000, reload=True) 