# Standard library imports
import os
import tempfile
import json
from typing import List

# Third-party imports
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
import uuid
from langchain_xai import ChatXAI

# Local imports
from oracle.utils import get_oracle_connection

# Load environment variables
load_dotenv()

app = FastAPI()

connection = get_oracle_connection()

THREAD_ID = str(uuid.uuid4())
# Setup basic variables
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
AUTH_TYPE = os.getenv("AUTH_TYPE")
CONFIG_PROFILE = os.getenv("CONFIG_PROFILE")
OCI_INFERENCE_ENDPOINT = os.getenv("OCI_INFERENCE_ENDPOINT")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

system_prompt = """You are a multi cloud Solutions Architect named Hako. You help your users with ad-hoc and strategic questions.

## Ad-hoc questions
Instructions:
- You can use the web tool to search the web for information on potential solutions. 
- Use the customer references tool to search for customer references
- Use the oracle marketplace tool to search for applications and solution integrators (SIs)
- your response should be a summary of the information found
- always include the url of the reference in the response

## Strategic questions
Instructions:
you are helping users design complete, dependency-aware solutions on Oracle Cloud Infrastructure (OCI). Here is the detailed process you follow:

Your job is to help users find solutions to their problems. 
When proposing solutions, you are helping users make the best decision for their customers, always keep Oracle OCI as the preferred solution, but
feel free to offer other solutions if they are a better fit. 

1. Discovery
- ask questions to understand
- you don't focus on the ask, you focus on the big picture, you always abstract the ask
  - for example, if the user says I need gpus for a chatbot. you need to understand why chatbot? why gpus?
- always map dependencies and constraints
  - for example, if customer wants to migrate a database, this is just a small part of the big picture. what is the scope of the workload? what is the timeline? what is the budget? what is the team's skillset?

2. Research
- use your knowledge and the web tool to search the web for information on potential solutions
- use your knowledge and the web tool to search OCI documentation for potential solutions or services that can compose the solutions
- use the customer references tool to search for customer references that align the customer's ask and solution, and include them in the response with the url
- use the oracle marketplace tool to search for applications and solution integrators (SIs) that can help solve the customer's problem
- you abstract the ask. 
  - for example, if the user says "I need to migrate my data to the cloud", you abstract that to "data migration"
  - if the user says "convert cobol to java", you abstract that to "cobol migration", "legacy migration", etc

3. Analyze
- analyze all the information collected during discovery and research
- think of solutions that can help the customer
  - do not consider solutions that do not add value to the customer's request
  - select 1 to 3 solutions that can add value to the customer's request
  - explain pros and cons of each solution proposed
  - select one to solution torecommend and explain why
  - include oracle marketplace products with their name and url, if any
  - include customer references with their name and url, if any
  - add additional questions (if needed)

4. Present
- summary
  - summary of what we discovered and researched
- solutions
  - 1-3 solutions evaluated
  - selected solution and why
- web results
    - if any, include their name andurls.
    - if none, say no web results found
- oracle marketplace applications and SIs
    - if any, include their name andurls.
    - if none, say no oracle marketplace applications and SIs found
- customer references
    - if any, include their name andurls.
    - if none, say no customer references found
- next steps plan
- additional questions (if any)


Guidelines:
- be objective
- be friendly and professional
- focus on practical, real-world recommendations
"""

# Initialize embeddings
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint=OCI_INFERENCE_ENDPOINT,
    truncate="NONE",
    compartment_id=COMPARTMENT_ID
    # auth_type=AUTH_TYPE,
    # auth_profile=CONFIG_PROFILE
)

llm = ChatXAI(
    model="grok-3"
)

# llm = init_chat_model("gpt-4o-mini", max_tokens=4000, model_provider="openai")

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

vector_store = OracleVS(
    client=connection,
    table_name="SOLUTIONSEMBEDDINGS",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever()

def search_serper(query: str) -> str:
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,    
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.text

@tool
def web_search(query: str) -> str:
    """Search the web for information on potential solutions"""
    return search_serper(query)

@tool
def get_customer_references(query: str):
    """
    Search the Oracle website for customer references

    Args:
        query: The query to search for

    Returns:
        A list of customer references
    """
    query = f"site:oracle.com/customers {query}"
    return search_serper(query)
    
@tool
def get_solutions(question: str):
    """Search the Oracle Cloud Marketplace for applications and solution integrators (SIs) that match the customer's request"""
    print(f"get_solutions: {question}")
    solutions = retriever.invoke(question, k=30)
    
    solutions = [{
        "vendor_name": solution.metadata["vendor_name"],
        "solution_name": solution.metadata["solution_name"],
        "solution_description": solution.metadata["solution_description"],
        "solution_url": solution.metadata["solution_url"]
    } for solution in solutions]

    print(f"solutions: {solutions}")
    return solutions

def chatbot(state: dict):
    return {"messages": [llm.invoke(state["messages"])]}

tools = [get_solutions,get_customer_references, web_search]

memory = MemorySaver()
graph = create_react_agent(llm,tools,prompt=system_prompt, checkpointer=memory)

def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": THREAD_ID}}
    result = graph.invoke({"messages": [("user", user_input)]}, config)
    return result["messages"][-1].content

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/embed-csv")
async def embed_csv(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Load CSV using CSVLoader
        loader = CSVLoader(file_path=tmp_file_path)
        docs = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        # Add to vector store
        vector_store.add_documents(documents=all_splits)

        # Clean up
        os.unlink(tmp_file_path)

        return JSONResponse(content={"message": "CSV processed and embedded successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/embed-text")
async def embed_text(text: str = Form(...)):
    try:
        # Get embedding for the text
        embedding = embeddings.embed_documents([text])[0]
        
        # Add to vector store
        vector_store.add_documents(
            documents=[text],
            embeddings=[embedding]
        )

        return JSONResponse(content={"message": "Text embedded successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 