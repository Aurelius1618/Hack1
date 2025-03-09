from typing import TypedDict, Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
import logging
from app.utils.model_config import classify_query

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the state for our agents
class AgentState(TypedDict):
    query: str 
    isin: Optional[str]
    results: dict
    routing_data: dict
    agent_results: dict

# Define the main workflow
def create_workflow():
    """
    Create the LangGraph workflow for the Tap Bonds AI Layer
    """
    # Initialize the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each agent
    workflow.add_node("orchestrator", route_query)
    workflow.add_node("directory_agent", handle_directory)
    workflow.add_node("finder_agent", handle_finder)
    workflow.add_node("cashflow_agent", handle_cashflow)
    workflow.add_node("screener_agent", handle_screener)
    
    # Add conditional edges based on the orchestrator's decision
    workflow.add_conditional_edges(
        "orchestrator",
        decide_next_node,
        {
            "directory": "directory_agent",
            "finder": "finder_agent",
            "cashflow": "cashflow_agent", 
            "screener": "screener_agent"
        }
    )
    
    # Add edges from each agent to END
    workflow.add_edge("directory_agent", END)
    workflow.add_edge("finder_agent", END)
    workflow.add_edge("cashflow_agent", END)
    workflow.add_edge("screener_agent", END)
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Compile the workflow
    return workflow.compile()

# Define the routing function
def route_query(state: AgentState) -> AgentState:
    """
    Initial routing function that analyzes the query and adds routing data
    Uses Mistral-7B (4-bit quantized) with LoRA adapters for classification
    """
    query = state["query"]
    
    # Extract ISIN if present
    import re
    isin_match = re.search(r'INE[A-Z0-9]{10}', query)
    if isin_match:
        state["isin"] = isin_match.group(0)
    
    # Use the Mistral-7B model for classification
    routing_data = classify_query(query)
    
    # Update the state
    state["routing_data"] = routing_data
    logger.info(f"Routing query to {routing_data['next_node']} with confidence {routing_data['confidence']}")
    return state

# Define the decision function for conditional edges
def decide_next_node(state: AgentState) -> str:
    """
    Determine which node to route to next based on the routing data
    """
    return state["routing_data"]["next_node"]

# Import agent handlers
from app.agents.directory_agent import handle_directory
from app.agents.finder_agent import handle_finder
from app.agents.cashflow_agent import handle_cashflow
from app.agents.screener_agent import handle_screener

# Create a singleton instance of the workflow
workflow = create_workflow() 