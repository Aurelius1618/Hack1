import re
import logging
from typing import Dict, Any, List, Optional
from app.core.pinecone_config import hybrid_search, NAMESPACES
from app.data.data_processor import FinancialDataProcessor
from app.utils.optimization import cached_isin_lookup, monitor_query_latency
from app.utils.model_config import get_mistral_model, cached_generate, handle_lamini_errors

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
        
        # Define chunking rules for different document types
        self.chunking_rules = {
            "legal_docs": {"size": 1024, "overlap": 256},
            "financial_terms": {"size": 512, "overlap": 128},
            "isin_details": {"size": 256, "overlap": 0}
        }
        
        # Initialize Lamini model
        self.llm = get_mistral_model()
        
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
        
        # Extract claimed issuer if present
        claimed_issuer = self._extract_issuer(query)
        
        if isin:
            # Validate ISIN
            if not self.validate_isin(isin):
                return {
                    "status": "error",
                    "message": f"Invalid ISIN: {isin}",
                    "data": None
                }
            
            # Check for ISIN-Issuer mismatch
            if claimed_issuer:
                mismatch_info = self._check_issuer_mismatch(isin, claimed_issuer)
                if mismatch_info.get("error") == "MISMATCH":
                    return {
                        "status": "warning",
                        "message": f"Issuer mismatch: You mentioned {claimed_issuer}, but the actual issuer for {isin} is {mismatch_info['actual_issuer']}",
                        "data": {
                            "actual_issuer": mismatch_info["actual_issuer"],
                            "similar_issuers": mismatch_info.get("similar_issuers", [])
                        }
                    }
            
            # Get bond details
            bond_details = self.data_processor.get_bond_details(isin)
            
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
    
    def handle_isin_mismatch(self, isin: str, claimed_issuer: str) -> Dict[str, Any]:
        """
        Handle ISIN-issuer mismatch
        
        Args:
            isin (str): ISIN code
            claimed_issuer (str): Claimed issuer name
            
        Returns:
            Dict[str, Any]: Response with mismatch information
        """
        # Get actual issuer from database
        actual_issuer = None
        if self.data_processor.bonds_df is not None:
            bond_data = self.data_processor.bonds_df[self.data_processor.bonds_df['isin'] == isin]
            if not bond_data.empty and 'issuer' in bond_data.columns:
                actual_issuer = bond_data['issuer'].iloc[0]
        
        # Find similar issuers
        similar_issuers = []
        if self.data_processor.bonds_df is not None:
            from rapidfuzz import fuzz
            
            # Get unique issuers
            issuers = self.data_processor.bonds_df['issuer'].unique()
            
            # Find similar issuers
            for issuer in issuers:
                similarity = fuzz.token_sort_ratio(claimed_issuer, issuer)
                if similarity >= 70:  # Threshold for similarity
                    # Get ISIN for this issuer
                    issuer_isin = self.data_processor.bonds_df[self.data_processor.bonds_df['issuer'] == issuer]['isin'].iloc[0]
                    similar_issuers.append({
                        "name": issuer,
                        "isin": issuer_isin,
                        "similarity": similarity
                    })
            
            # Sort by similarity
            similar_issuers = sorted(similar_issuers, key=lambda x: x["similarity"], reverse=True)[:3]
        
        return {
            "status": "error",
            "message": "ISIN-issuer mismatch",
            "data": {
                "isin": isin,
                "claimed_issuer": claimed_issuer,
                "actual_issuer": actual_issuer,
                "similar_issuers": similar_issuers
            }
        }
    
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

    def _extract_issuer(self, query: str) -> Optional[str]:
        """
        Extract issuer name from query
        
        Args:
            query (str): User query
            
        Returns:
            Optional[str]: Issuer name if found
        """
        # Look for patterns like "issuer is X" or "X bond" or "bond from X"
        issuer_patterns = [
            r'(?:issuer|company|organization|firm)\s+(?:is|of|named)\s+([A-Za-z0-9\s&]+)',
            r'([A-Za-z0-9\s&]+)\s+bond',
            r'bond\s+(?:from|by|issued by)\s+([A-Za-z0-9\s&]+)'
        ]
        
        for pattern in issuer_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def _check_issuer_mismatch(self, isin: str, claimed_issuer: Optional[str]) -> Dict[str, Any]:
        """
        Check for ISIN-Issuer mismatch
        
        Args:
            isin (str): ISIN code
            claimed_issuer (Optional[str]): Claimed issuer name
            
        Returns:
            Dict[str, Any]: Mismatch information
        """
        if not claimed_issuer:
            return {"error": None}
        
        return self.data_processor.handle_isin_mismatch(isin, claimed_issuer)

    def get_security_details(self, isin: str) -> Dict[str, Any]:
        """
        Get security details for a bond
        
        Args:
            isin (str): ISIN code
            
        Returns:
            Dict[str, Any]: Security details
        """
        if not self.validate_isin(isin):
            return {
                "status": "error",
                "message": f"Invalid ISIN: {isin}",
                "data": None
            }
        
        # Get bond details
        bond_details = self.data_processor.get_bond_details(isin)
        if not bond_details:
            return {
                "status": "error",
                "message": f"No bond details found for ISIN: {isin}",
                "data": None
            }
        
        # Extract security details
        security_details = {
            "collateral": bond_details.get("collateral", "Not specified"),
            "coverage_ratio": bond_details.get("coverage_ratio", None),
            "liquidation_priority": bond_details.get("liquidation_priority", "Not specified"),
            "security_type": bond_details.get("security_type", "Not specified"),
            "secured": bond_details.get("is_secured", False)
        }
        
        return {
            "status": "success",
            "message": f"Found security details for ISIN: {isin}",
            "data": security_details
        }
    
    @handle_lamini_errors
    def handle_isin_query(self, isin: str) -> Dict[str, Any]:
        """
        Handle a query for a specific ISIN
        
        Args:
            isin (str): ISIN code
            
        Returns:
            Dict[str, Any]: Bond details
        """
        if not self.validate_isin(isin):
            return {
                "status": "error",
                "message": f"Invalid ISIN: {isin}",
                "data": None
            }
        
        # Get bond details
        bond_details = self.data_processor.get_bond_details(isin)
        
        if bond_details:
            # Get document links
            document_links = self.get_document_links(isin)
            
            # Format query for Lamini
            query_template = f"""
            Provide a detailed summary of the bond with ISIN {isin}.
            
            Bond details:
            {bond_details}
            
            Document links:
            {document_links}
            
            Response:
            """
            
            # Generate response using Lamini
            if self.llm:
                try:
                    response = cached_generate(query_template)
                    
                    return {
                        "status": "success",
                        "message": "Bond details retrieved successfully",
                        "data": {
                            "isin": isin,
                            "details": bond_details,
                            "documents": document_links,
                            "summary": response
                        }
                    }
                except Exception as e:
                    logger.error(f"Error generating response with Lamini: {str(e)}")
            
            # Fallback response if Lamini fails
            return {
                "status": "success",
                "message": "Bond details retrieved successfully",
                "data": {
                    "isin": isin,
                    "details": bond_details,
                    "documents": document_links
                }
            }
        else:
            return {
                "status": "error",
                "message": f"No bond found with ISIN: {isin}",
                "data": None
            }
    
    def get_document_links(self, isin: str) -> Dict[str, Any]:
        """
        Get document links for a bond
        
        Args:
            isin (str): ISIN code
            
        Returns:
            Dict[str, Any]: Document links
        """
        # This would typically query a database or API to get document links
        # For now, we'll return a mock response
        document_map = {
            # Mock data - in production this would come from a database
            "INE123456789": {
                "offer_doc": f"https://tapbonds.com/docs/{isin}_offer.pdf",
                "trust_deed": f"https://tapbonds.com/docs/{isin}_deed.pdf"
            }
        }
        
        documents = document_map.get(isin, None)
        
        if documents:
            return {
                "status": "success",
                "message": f"Found document links for ISIN: {isin}",
                "data": documents
            }
        else:
            return {
                "status": "error",
                "message": f"No document links found for ISIN: {isin}",
                "data": None
            }
    
    def chunk_document(self, document_text: str, doc_type: str) -> List[str]:
        """
        Chunk a document based on document type
        
        Args:
            document_text (str): Document text
            doc_type (str): Document type (legal_docs, financial_terms, isin_details)
            
        Returns:
            List[str]: Document chunks
        """
        # Get chunking rules for document type
        rules = self.chunking_rules.get(doc_type, self.chunking_rules["legal_docs"])
        chunk_size = rules["size"]
        overlap = rules["overlap"]
        
        # Split document into chunks
        chunks = []
        for i in range(0, len(document_text), chunk_size - overlap):
            chunk = document_text[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        return chunks

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
    
    # Use actor-critic flow for reasoning
    from app.utils.model_config import actor_critic_flow
    reasoning_result = actor_critic_flow(query)
    
    # Add reasoning to state
    state["reasoning_chain"].append(reasoning_result["reasoning"])
    
    # Process query
    result = agent.process_query(query)
    
    # Update state
    state["agent_results"] = result
    
    # Extract entities from result
    if result["status"] == "success" and result["data"]:
        # Extract bond details
        if isinstance(result["data"], list):
            # Multiple bonds
            for bond in result["data"]:
                for key, value in bond.items():
                    if key not in ["search_score"]:
                        state["parsed_entities"][key] = value
                break  # Just use the first bond for entities
        else:
            # Single bond
            for key, value in result["data"].items():
                if key not in ["search_score"]:
                    state["parsed_entities"][key] = value
    
    # Update financial context
    state["financial_context"]["bond_type"] = state["parsed_entities"].get("bond_type")
    state["financial_context"]["issuer"] = state["parsed_entities"].get("issuer")
    state["financial_context"]["sector"] = state["parsed_entities"].get("sector")
    
    return state