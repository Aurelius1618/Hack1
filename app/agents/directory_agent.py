import re
import logging
from typing import Dict, Any, List, Optional
from app.core.pinecone_config import hybrid_search, NAMESPACES
from app.data.data_processor import FinancialDataProcessor
from app.utils.optimization import cached_isin_lookup, monitor_query_latency

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectoryAgent:
    """
    Agent for handling bond directory queries
    """
    def __init__(self):
        """
        Initialize the directory agent
        """
        self.data_processor = FinancialDataProcessor()
        
    def initialize(self):
        """
        Initialize the agent by loading data
        """
        self.data_processor.load_data()
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query related to bond directory
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Any]: Response with bond information
        """
        # Extract ISIN from query if present
        isin = self._extract_isin(query)
        
        if isin:
            # Validate ISIN
            if not self.validate_isin(isin):
                return {
                    "status": "error",
                    "message": f"Invalid ISIN: {isin}",
                    "data": None
                }
            
            # Get bond details using cached lookup
            bond_details = cached_isin_lookup(isin)
            
            if bond_details:
                return {
                    "status": "success",
                    "message": f"Found bond details for ISIN: {isin}",
                    "data": bond_details
                }
            else:
                return {
                    "status": "error",
                    "message": f"No bond details found for ISIN: {isin}",
                    "data": None
                }
        else:
            # If no ISIN in query, perform semantic search
            search_results = self._semantic_search(query)
            
            if search_results:
                return {
                    "status": "success",
                    "message": "Found bonds matching your query",
                    "data": search_results
                }
            else:
                return {
                    "status": "error",
                    "message": "No bonds found matching your query",
                    "data": None
                }
    
    def _extract_isin(self, query: str) -> Optional[str]:
        """
        Extract ISIN from query
        
        Args:
            query (str): User query
            
        Returns:
            Optional[str]: Extracted ISIN or None
        """
        # Pattern for Indian ISINs
        pattern = r"INE[A-Z0-9]{10}"
        match = re.search(pattern, query)
        
        if match:
            return match.group(0)
        
        return None
    
    def validate_isin(self, isin: str) -> bool:
        """
        Validate an ISIN code
        
        Args:
            isin (str): ISIN code to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check if ISIN matches the pattern for Indian ISINs
        pattern = r"^INE[A-Z0-9]{10}$"
        if not re.fullmatch(pattern, isin):
            return False
        
        # Additional validation logic could be added here
        # For example, check if the ISIN exists in our database
        if self.data_processor.bonds_df is not None:
            return isin in self.data_processor.bonds_df['isin'].values
        
        return True
    
    @monitor_query_latency
    def _semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform semantic search for bonds using hybrid search (BM25 + SBERT)
        
        Args:
            query (str): User query
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        try:
            # Perform hybrid search with alpha=0.7 (70% dense, 30% sparse)
            results = hybrid_search(
                query=query,
                namespace=NAMESPACES["BOND_METADATA"],
                top_k=5,
                alpha=0.7
            )
            
            # Process results
            processed_results = []
            for match in results.matches:
                # Get bond details from database
                isin = match.metadata.get("isin")
                if isin:
                    bond_details = self.data_processor.get_bond_details(isin)
                    if bond_details:
                        # Add search score
                        bond_details["search_score"] = match.score
                        processed_results.append(bond_details)
            
            return processed_results
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            
            # Fallback to simple search in database
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Fallback search method when semantic search fails
        
        Args:
            query (str): User query
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        if self.data_processor.bonds_df is None:
            return []
        
        # Extract keywords from query
        keywords = query.lower().split()
        
        # Filter bonds based on keywords
        results = []
        for _, bond in self.data_processor.bonds_df.iterrows():
            # Check if any keyword matches bond attributes
            match_score = 0
            for keyword in keywords:
                if (
                    (isinstance(bond.get('issuer'), str) and keyword in bond['issuer'].lower()) or
                    (isinstance(bond.get('bond_type'), str) and keyword in bond['bond_type'].lower()) or
                    (isinstance(bond.get('sector'), str) and keyword in bond['sector'].lower())
                ):
                    match_score += 1
            
            if match_score > 0:
                bond_dict = bond.to_dict()
                bond_dict["search_score"] = match_score / len(keywords)
                results.append(bond_dict)
        
        # Sort by match score
        results.sort(key=lambda x: x["search_score"], reverse=True)
        
        # Return top 5 results
        return results[:5]

# Function to handle directory agent queries in the LangGraph workflow
def handle_directory(state):
    """
    Handler for directory agent in the LangGraph workflow
    
    Args:
        state: The current state
        
    Returns:
        Updated state
    """
    query = state["query"]
    
    # Initialize agent
    agent = DirectoryAgent()
    agent.initialize()
    
    # Process query
    result = agent.process_query(query)
    
    # Update state
    state["agent_results"] = result
    
    return state 