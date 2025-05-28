import oracledb
import os
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model

def get_oracle_connection():
    # Connection parameters
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")

    return get_oracle_connection_with_creds(username, password)

def get_oracle_connection_with_creds(username, password):
    dsn = "(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.us-chicago-1.oraclecloud.com))(connect_data=(service_name=g91896c45600041_solutionschatdb_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))"

    # Connect using wallet configuration
    conn = oracledb.connect(
        user=username,
        password=password,
        dsn=dsn
    )
    
    return conn

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def load_hako_agent_prompt():
    with open("oracle/hako_agent_prompt.md", "r") as f:
        return f.read()

class LLMFactory:
    @staticmethod
    def create_llm(model_type: str, **kwargs):
        if model_type == "oci":
            return ChatOCIGenAI(
                model_id=kwargs.get("model_id", "cohere.command-a-03-2025"),
                service_endpoint=kwargs.get("service_endpoint", os.getenv("OCI_INFERENCE_ENDPOINT")),
                compartment_id=kwargs.get("compartment_id", os.getenv("COMPARTMENT_ID")),
                is_stream=kwargs.get("is_stream", True),
                model_kwargs=kwargs.get("model_kwargs", {
                    "temperature": 1,
                    "max_tokens": 4000
                }),
                auth_type=kwargs.get("auth_type", os.getenv("AUTH_TYPE")),
                auth_profile=kwargs.get("auth_profile", os.getenv("CONFIG_PROFILE"))
            )
        else:
            return init_chat_model(model_type)