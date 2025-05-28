from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent
from datetime import datetime
from langgraph_supervisor import create_supervisor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import uuid
from dotenv import load_dotenv
import os
import requests
import json

### INITIALIZATION ###
# Load environment variables
load_dotenv()

THREAD_ID = str(uuid.uuid4())
# Setup basic variables
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
AUTH_TYPE = os.getenv("AUTH_TYPE")
CONFIG_PROFILE = os.getenv("CONFIG_PROFILE")
OCI_INFERENCE_ENDPOINT = os.getenv("OCI_INFERENCE_ENDPOINT")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

model = ChatOpenAI()

### PROMPTS ###
discovery_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Discovery Agent, an OCI Solutions Architect specialist. "
            "Your job is to front-load questions and gather as much detail about the opportunity as possible before any design talk. "
            "Ask about business goals, urgency drivers, current infrastructure, critical workloads, technical constraints, success criteria, timeline, scope, and stakeholders. "
            "When you have all the facts, return a clear, structured summary and hand control back to the supervisor."
        ),
    ]
).partial(time=datetime.now)

research_agent_prompt = ChatPromptTemplate.from_messages(
[
    (
        "system",
        "You are the Research Agent, an OCI specialist for deep technical and competitive analysis. "
        "Given the opportunity context, query internal and external knowledge bases, APIs, and documentation to uncover best practices, technology trends, competitor comparisons, and integration patterns. "
        "Return concise summaries with citations or links, then hand results back to the supervisor."
    ),
]).partial(time=datetime.now)

references_agent_prompt = ChatPromptTemplate.from_messages(
[
    (
        "system",
        "You are the References Agent, responsible for fetching customer success stories and case studies. "
        "Given the opportunity context, look up relevant examples from your CRM or knowledge base—include customer name, industry, outcomes, metrics, and links. "
        "Present a shortlist of the most relevant references and hand it back to the supervisor."
    ),
]).partial(time=datetime.now)

lead_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Lead Agent, the senior Oracle Cloud Infrastructure Solutions Architect. "
            "You receive the discovery summary, research findings, and customer references from other agents. "
            "Your job is to analyze all inputs and present multiple solution options, each with clear pros and cons. "
            "If there are any relevant customer references or products from Oracle Cloud Market Place, they should be included with their name, description and link. "
            "Then recommend the single best solution and explain why it's preferred. "
            "Also provide a detailed next-steps implementation plan and list any additional discovery questions needed to fill gaps. "
            "Always keep the customer's business objectives and technical constraints front and center."
        ),
    ]).partial(time=datetime.now)

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Supervisor Agent, the orchestrator for the multi-agent OCI Solutions Architect chatbot. "
            "Your role is to receive the user's input, determine which specialist agent to invoke (DiscoveryAgent, ResearchAgent, ReferencesAgent, or LeadAgent), "
            "send the relevant context to that agent, collect its output, and loop until the final recommendation is ready. "
            "Always start by invoking the DiscoveryAgent to gather opportunity details. "
            "After receiving its summary, call the ResearchAgent and ReferencesAgent in parallel to enrich the context. "
            "Once you have all inputs, invoke the LeadAgent to produce the final solution options, recommendation, implementation plan, and any follow-up questions. "
            "After each agent call, check if more information is needed—if so, return to DiscoveryAgent. "
            "When the LeadAgent output is complete, present it to the user as the final response. "
            "Keep context consistent and ensure each agent sees only the information it needs."
        ),
    ]
).partial(time=datetime.now)

### TOOLS ###
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

def web_search(query: str):
    """Helps researcher agent search the web for information regarding the customer's request"""
    return search_serper(query)

### AGENTS ###
discovery_agent = create_react_agent(
    model=model,
    tools=[],
    name="discovery_agent",
    prompt=discovery_agent_prompt
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_agent",
    prompt=research_agent_prompt
)

### SUPERVISOR ###
agents = [discovery_agent, research_agent]
workflow = create_supervisor(
    agents,
    model=model,
    prompt=supervisor_prompt
)
supervisor = workflow.compile()
while True:
    user_input = input("Enter a message: ")
    
    if user_input == "exit":
        break
    
    events = supervisor.stream({"messages": [HumanMessage(content=user_input)]}, stream_mode="values")
    for event in events:
        for message in event["messages"]:
            message.pretty_print()
