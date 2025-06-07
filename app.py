# Standard library imports
import os
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
import uuid
from oracle.utils import LLMFactory, load_hako_agent_prompt
from oracle.tools import search_oracle_customer_references, search_oracle_marketplace, search_oracle_documentation, web_search

# Load environment variables
if os.getenv("APP_ENV", "dev").lower() == "dev":
    load_dotenv()

# Setup basic variables
THREAD_ID = str(uuid.uuid4())

# initialize the FastAPI app
app = FastAPI()

# initialize short term memory
# TODO: use a database for this and persist the memory
memory = InMemorySaver()

# initialize the LLM (we can choose from oci or any other model supported by langchain)
# https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
llm = LLMFactory.create_llm("xai:grok-3")

# initialize the tools
hako_tools = [
    search_oracle_customer_references, 
    search_oracle_marketplace, 
    search_oracle_documentation, 
    web_search
]

# load the hako agent prompt
hako_agent_prompt = load_hako_agent_prompt()

# initialize the hako agent
hako_agent = create_react_agent(
    llm,
    tools=hako_tools,
    prompt=hako_agent_prompt, 
    checkpointer=memory
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/query")
async def query(question: str):
    config = {"configurable": {"thread_id": THREAD_ID}}
    result = hako_agent.invoke({"messages": [("user", question)]}, config)

    response = result["messages"][-1].content
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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 