from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import os
from dotenv import load_dotenv

# Import the LangGraph workflow
from app.core.workflow import workflow

# Import performance monitoring
from app.utils.monitoring import performance_monitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Tap Bonds AI Layer",
    description="AI-powered bond analysis system using LangGraph and PineconeDB",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class QueryRequest(BaseModel):
    query: str

class CompareRequest(BaseModel):
    isins: List[str]

class PortfolioRequest(BaseModel):
    isins: List[str]

class QueryResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

# Define API endpoints
@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Welcome to Tap Bonds AI Layer",
        "version": "1.0.0",
        "docs": "/docs",
    }

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a natural language query about bonds
    """
    start_time = performance_monitor.start_query()
    
    try:
        # Log the query
        logger.info(f"Processing query: {request.query}")
        
        # Check for EBIT-specific queries to handle them properly
        if "ebit" in request.query.lower():
            # Extract EBIT-related parameters
            import re
            ebit_match = re.search(r'ebit\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?k?m?b?)', request.query, re.IGNORECASE)
            interest_match = re.search(r'interest\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?k?m?b?)', request.query, re.IGNORECASE)
            tax_match = re.search(r'tax\s*(?:rate|percentage)?\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?)\s*%?', request.query, re.IGNORECASE)
            shares_match = re.search(r'shares\s*(?:outstanding|count|number)?\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?k?m?b?)', request.query, re.IGNORECASE)
            
            # Convert values if found
            def convert_value(match):
                if not match:
                    return None
                val = match.group(1).lower()
                # Handle k, m, b suffixes
                if val.endswith('k'):
                    return float(val[:-1]) * 1_000
                elif val.endswith('m'):
                    return float(val[:-1]) * 1_000_000
                elif val.endswith('b'):
                    return float(val[:-1]) * 1_000_000_000
                return float(val)
            
            # Extract and convert values
            ebit = convert_value(ebit_match)
            interest = convert_value(interest_match)
            tax_rate = tax_match.group(1) if tax_match else None
            tax_rate = float(tax_rate) / 100 if tax_rate else None  # Convert percentage to decimal
            shares = convert_value(shares_match)
            
            # Initialize data processor
            from app.data.data_processor import FinancialDataProcessor
            data_processor = FinancialDataProcessor()
            data_processor.load_data()
            
            # Check if we have enough information for EBIT calculation
            if ebit is None:
                logger.error("EBIT value not provided in query")
                return {
                    "status": "error",
                    "message": "EBIT value must be provided for financial calculations. Please specify EBIT amount.",
                    "data": None
                }
                
            # Validate inputs if they're provided
            validation = data_processor.validate_ebit_inputs(ebit, interest, tax_rate, shares)
            if validation["status"] == "error":
                logger.error(f"EBIT validation error: {validation['message']}")
                return {
                    "status": "error",
                    "message": validation["message"],
                    "data": None
                }
            
            # If we're calculating EPS and have all required inputs
            if all([ebit is not None, interest is not None, tax_rate is not None, shares is not None]):
                try:
                    eps = data_processor.calculate_eps(ebit, interest, tax_rate, shares)
                    return {
                        "status": "success",
                        "message": "EPS calculated successfully",
                        "data": {
                            "eps": eps,
                            "inputs": {
                                "ebit": ebit,
                                "interest": interest,
                                "tax_rate": tax_rate * 100,  # Convert to percentage for display
                                "shares": shares
                            }
                        }
                    }
                except ValueError as ve:
                    logger.error(f"EBIT calculation error: {str(ve)}")
                    return {
                        "status": "error",
                        "message": f"Invalid input: {str(ve)}",
                        "data": None
                    }
        
        # Process the query using the workflow for non-EBIT specific queries
        result = workflow.invoke({"query": request.query, "results": {}, "agent_results": {}})
        
        # Extract the agent that processed the query
        agent = result.get("routing_data", {}).get("next_node", "unknown") + "_agent"
        
        # Extract the results
        agent_results = result.get("agent_results", {})
        
        # Record successful query
        performance_monitor.end_query(start_time, agent, True)
        
        # Return the response
        return agent_results
    except Exception as e:
        # Log the error with more detail
        logger.error(f"Error processing query: {str(e)}")
        logger.exception("Detailed error traceback:")
        
        # Record failed query
        performance_monitor.end_query(start_time, "unknown", False, str(e))
        
        # Provide more detailed error message
        error_message = str(e)
        if "ebit" in error_message.lower():
            error_message = "EBIT calculation error. Please ensure you provided valid numeric values for EBIT and any related parameters (interest, tax rate, shares)."
        
        # Return error response
        return {
            "status": "error",
            "message": f"Error processing query: {error_message}",
            "data": None
        }

@app.get("/api/bond/{isin}", response_model=QueryResponse)
async def get_bond_details(isin: str):
    """
    Get details for a specific bond by ISIN
    """
    start_time = performance_monitor.start_query()
    
    try:
        # Log the query
        logger.info(f"Getting details for bond: {isin}")
        
        # Process the query using the workflow
        result = workflow.invoke({
            "query": f"Get details for bond {isin}",
            "isin": isin,
            "results": {},
            "agent_results": {},
            "routing_data": {"next_node": "directory"}
        })
        
        # Extract the results
        agent_results = result.get("agent_results", {})
        
        # Record successful query
        performance_monitor.end_query(start_time, "directory_agent", True)
        
        # Return the response
        return agent_results
    except Exception as e:
        # Log the error
        logger.error(f"Error getting bond details: {str(e)}")
        
        # Record failed query
        performance_monitor.end_query(start_time, "directory_agent", False, str(e))
        
        # Return error response
        return {
            "status": "error",
            "message": f"Error getting bond details: {str(e)}",
            "data": None
        }

@app.get("/api/cashflow/{isin}", response_model=QueryResponse)
async def get_cashflow_schedule(isin: str):
    """
    Get cashflow schedule for a specific bond by ISIN
    """
    start_time = performance_monitor.start_query()
    
    try:
        # Log the query
        logger.info(f"Getting cashflow schedule for bond: {isin}")
        
        # Process the query using the workflow
        result = workflow.invoke({
            "query": f"Get cashflow schedule for bond {isin}",
            "isin": isin,
            "results": {},
            "agent_results": {},
            "routing_data": {"next_node": "cashflow"}
        })
        
        # Extract the results
        agent_results = result.get("agent_results", {})
        
        # Record successful query
        performance_monitor.end_query(start_time, "cashflow_agent", True)
        
        # Return the response
        return agent_results
    except Exception as e:
        # Log the error
        logger.error(f"Error getting cashflow schedule: {str(e)}")
        
        # Record failed query
        performance_monitor.end_query(start_time, "cashflow_agent", False, str(e))
        
        # Return error response
        return {
            "status": "error",
            "message": f"Error getting cashflow schedule: {str(e)}",
            "data": None
        }

@app.post("/api/compare", response_model=QueryResponse)
async def compare_bonds(request: CompareRequest):
    """
    Compare multiple bonds by ISIN
    """
    start_time = performance_monitor.start_query()
    
    try:
        # Log the query
        logger.info(f"Comparing bonds: {request.isins}")
        
        # Process the query using the workflow
        result = workflow.invoke({
            "query": f"Compare bonds {', '.join(request.isins)}",
            "results": {},
            "agent_results": {},
            "routing_data": {"next_node": "finder"}
        })
        
        # Extract the results
        agent_results = result.get("agent_results", {})
        
        # Record successful query
        performance_monitor.end_query(start_time, "finder_agent", True)
        
        # Return the response
        return agent_results
    except Exception as e:
        # Log the error
        logger.error(f"Error comparing bonds: {str(e)}")
        
        # Record failed query
        performance_monitor.end_query(start_time, "finder_agent", False, str(e))
        
        # Return error response
        return {
            "status": "error",
            "message": f"Error comparing bonds: {str(e)}",
            "data": None
        }

@app.post("/api/portfolio/analyze", response_model=QueryResponse)
async def analyze_portfolio(request: PortfolioRequest):
    """
    Analyze a portfolio of bonds
    """
    start_time = performance_monitor.start_query()
    
    try:
        # Log the query
        logger.info(f"Analyzing portfolio with {len(request.isins)} bonds")
        
        # Process the query using the workflow
        result = workflow.invoke({
            "query": f"Analyze portfolio with bonds {', '.join(request.isins)}",
            "results": {},
            "agent_results": {},
            "routing_data": {"next_node": "screener"}
        })
        
        # Extract the results
        agent_results = result.get("agent_results", {})
        
        # Record successful query
        performance_monitor.end_query(start_time, "screener_agent", True)
        
        # Return the response
        return agent_results
    except Exception as e:
        # Log the error
        logger.error(f"Error analyzing portfolio: {str(e)}")
        
        # Record failed query
        performance_monitor.end_query(start_time, "screener_agent", False, str(e))
        
        # Return error response
        return {
            "status": "error",
            "message": f"Error analyzing portfolio: {str(e)}",
            "data": None
        }

@app.get("/api/screen", response_model=QueryResponse)
async def screen_bonds(
    min_yield: Optional[float] = Query(None, description="Minimum yield"),
    max_yield: Optional[float] = Query(None, description="Maximum yield"),
    rating: Optional[str] = Query(None, description="Bond rating"),
    sector: Optional[str] = Query(None, description="Bond sector"),
    min_maturity: Optional[int] = Query(None, description="Minimum maturity in years"),
    max_maturity: Optional[int] = Query(None, description="Maximum maturity in years"),
    issuer_type: Optional[str] = Query(None, description="Issuer type"),
    financial_health: Optional[str] = Query(None, description="Financial health category"),
    limit: int = Query(10, description="Maximum number of results")
):
    """
    Screen bonds based on criteria
    """
    start_time = performance_monitor.start_query()
    
    try:
        # Build query string from parameters
        query_parts = []
        if min_yield is not None:
            query_parts.append(f"minimum yield of {min_yield}%")
        if max_yield is not None:
            query_parts.append(f"maximum yield of {max_yield}%")
        if rating is not None:
            query_parts.append(f"rating of {rating}")
        if sector is not None:
            query_parts.append(f"in the {sector} sector")
        if min_maturity is not None:
            query_parts.append(f"minimum maturity of {min_maturity} years")
        if max_maturity is not None:
            query_parts.append(f"maximum maturity of {max_maturity} years")
        if issuer_type is not None:
            query_parts.append(f"issuer type of {issuer_type}")
        if financial_health is not None:
            query_parts.append(f"financial health of {financial_health}")
        
        query = f"Find bonds with {', '.join(query_parts)} limited to {limit} results"
        
        # Log the query
        logger.info(f"Screening bonds: {query}")
        
        # Process the query using the workflow
        result = workflow.invoke({
            "query": query,
            "results": {},
            "agent_results": {},
            "routing_data": {"next_node": "screener"}
        })
        
        # Extract the results
        agent_results = result.get("agent_results", {})
        
        # Record successful query
        performance_monitor.end_query(start_time, "screener_agent", True)
        
        # Return the response
        return agent_results
    except Exception as e:
        # Log the error
        logger.error(f"Error screening bonds: {str(e)}")
        
        # Record failed query
        performance_monitor.end_query(start_time, "screener_agent", False, str(e))
        
        # Return error response
        return {
            "status": "error",
            "message": f"Error screening bonds: {str(e)}",
            "data": None
        }

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    # Get performance metrics
    metrics = performance_monitor.get_performance_metrics()
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "query_count": metrics["query_count"],
        "success_rate": metrics["success_rate"] if metrics["success_rate"] is not None else "N/A",
        "avg_response_time": metrics["avg_response_time"] if metrics["avg_response_time"] is not None else "N/A"
    }

@app.get("/api/metrics")
async def get_metrics():
    """
    Get performance metrics
    """
    # Get performance metrics
    metrics = performance_monitor.get_performance_metrics()
    
    return metrics

@app.get("/api/report")
async def get_report():
    """
    Get performance report
    """
    # Generate report
    report = performance_monitor.generate_report()
    
    return {"report": report}

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run the application
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=port, reload=True) 