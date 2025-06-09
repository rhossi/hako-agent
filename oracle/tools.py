from tavily import TavilyClient
from oracle.utils import get_oracle_connection
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_openai import ChatOpenAI
import os
from langchain_community.embeddings import OCIGenAIEmbeddings
from oracle.utils import LLMFactory

COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
OCI_INFERENCE_ENDPOINT = os.getenv("OCI_INFERENCE_ENDPOINT")
EMBEDDINGS_TABLE_NAME = os.getenv("EMBEDDINGS_TABLE_NAME")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def _query_writer(question: str):
    search_llm = LLMFactory.create_llm("xai:grok-3-mini")
    query_writer_instructions = """
        Your goal is to generate a targeted web search query.

        The query will gather information related to a specific question:
        {question}

        Think of keywords related to the technologies (eg: Java, Microservices) or actions (eg: modernization, migration)
        
        Return a list with the top 5 single keywords ordered by relevance separated by OR
    """
    results = search_llm.invoke(query_writer_instructions.format(question=question))
    query = results.content
    return query

def _tavily_search(query: str, **kwargs):
    client = TavilyClient(TAVILY_API_KEY)
    return client.search(query=query,**kwargs)

def web_search(query: str):
    """search the web"""
    return _tavily_search(query=query)
    
def search_oracle_customer_references(question: str):
    """search for oracle customer references"""
    query = _query_writer(question)
    return _tavily_search(query=query,
                          search_depth="advanced",
                          include_domains=["oracle.com/customers"])

def search_oracle_marketplace(question: str):
    """Search the Oracle Cloud Marketplace for applications and solution integrators (SIs) that match the customer's request"""

    app_env = os.getenv("APP_ENV", "dev").lower()
    auth_type = "API_KEY" if app_env == "dev" else "RESOURCE_PRINCIPAL"

    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-multilingual-v3.0",
        service_endpoint=OCI_INFERENCE_ENDPOINT,
        truncate="NONE",
        compartment_id=COMPARTMENT_ID,
        auth_type=auth_type,
    )

    connection = get_oracle_connection()

    vector_store = OracleVS(
        client=connection,
        table_name=EMBEDDINGS_TABLE_NAME,
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever()
    
    solutions = retriever.invoke(question, k=5)
    
    solutions = [{
        "vendor_name": solution.metadata["vendor_name"],
        "solution_name": solution.metadata["solution_name"],
        "solution_description": solution.metadata["solution_description"],
        "solution_url": solution.metadata["solution_url"]
    } for solution in solutions]

    return solutions

def search_oracle_documentation(question: str):
    """search oracle documentation"""

    query = _query_writer(question)
    return _tavily_search(query=query,search_depth="advanced",include_domains=["docs.oracle.com/en"])