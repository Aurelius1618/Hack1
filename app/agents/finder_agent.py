import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from app.core.pinecone_config import hybrid_search, NAMESPACES
from app.data.data_processor import FinancialDataProcessor
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinderAgent:
    """
    Agent for finding and comparing bonds based on yield and other criteria
    """
    def __init__(self):
        """
        Initialize the finder agent
        """
        self.data_processor = FinancialDataProcessor()
        self.platform_weights = {
            "SMEST": 0.85,
            "FixedIncome": 0.78,
            "Institutional": 0.92,
            "Default": 0.80  # Default weight for unknown platforms
        }
        
    def initialize(self):
        """
        Initialize the agent by loading data
        """
        self.data_processor.load_data()
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query related to finding bonds
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Any]: Response with bond information
        """
        # Extract parameters from query
        params = self._extract_parameters(query)
        
        # Find bonds matching criteria
        matching_bonds = self._find_matching_bonds(params)
        
        if matching_bonds:
            # Normalize yields for comparison
            normalized_bonds = self._normalize_yields(matching_bonds)
            
            # Apply platform reliability scoring
            scored_bonds = self._apply_platform_scoring(normalized_bonds)
            
            # Sort by effective yield
            sorted_bonds = sorted(scored_bonds, key=lambda x: x.get('effective_yield', 0), reverse=True)
            
            return {
                "status": "success",
                "message": f"Found {len(sorted_bonds)} bonds matching your criteria",
                "data": sorted_bonds
            }
        else:
            return {
                "status": "error",
                "message": "No bonds found matching your criteria",
                "data": None
            }
    
    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract search parameters from query
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Any]: Extracted parameters
        """
        params = {}
        
        # Extract minimum yield
        min_yield_match = re.search(r'(?:above|min|minimum|at least)\s+(\d+(?:\.\d+)?)\s*%?\s*(?:yield|return)', query, re.IGNORECASE)
        if min_yield_match:
            params['min_yield'] = float(min_yield_match.group(1))
        
        # Extract maximum yield
        max_yield_match = re.search(r'(?:below|max|maximum|at most)\s+(\d+(?:\.\d+)?)\s*%?\s*(?:yield|return)', query, re.IGNORECASE)
        if max_yield_match:
            params['max_yield'] = float(max_yield_match.group(1))
        
        # Extract maturity
        maturity_match = re.search(r'(?:maturity|term)\s+(?:of\s+)?(\d+)\s*(?:year|yr)', query, re.IGNORECASE)
        if maturity_match:
            params['maturity_years'] = int(maturity_match.group(1))
        
        # Extract rating
        rating_match = re.search(r'(?:rating|rated)\s+([A-D][+-]?)', query, re.IGNORECASE)
        if rating_match:
            params['rating'] = rating_match.group(1).upper()
        
        # Extract sector
        sectors = ['technology', 'healthcare', 'finance', 'energy', 'consumer', 'industrial', 'utilities', 'telecom']
        for sector in sectors:
            if sector in query.lower():
                params['sector'] = sector
                break
        
        # Extract issuer type
        issuer_types = ['corporate', 'government', 'municipal', 'sovereign']
        for issuer_type in issuer_types:
            if issuer_type in query.lower():
                params['issuer_type'] = issuer_type
                break
        
        return params
    
    def _find_matching_bonds(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find bonds matching the given parameters
        
        Args:
            params (Dict[str, Any]): Search parameters
            
        Returns:
            List[Dict[str, Any]]: Matching bonds
        """
        if self.data_processor.bonds_df is None:
            return []
        
        # Start with all bonds
        filtered_df = self.data_processor.bonds_df.copy()
        
        # Apply filters based on parameters
        if 'min_yield' in params:
            filtered_df = filtered_df[filtered_df['yield'] >= params['min_yield']]
        
        if 'max_yield' in params:
            filtered_df = filtered_df[filtered_df['yield'] <= params['max_yield']]
        
        if 'maturity_years' in params:
            # Calculate maturity date
            target_maturity = datetime.now().year + params['maturity_years']
            # Filter bonds with maturity date within 1 year of target
            filtered_df = filtered_df[
                (filtered_df['maturity_date'].dt.year >= target_maturity - 1) &
                (filtered_df['maturity_date'].dt.year <= target_maturity + 1)
            ]
        
        if 'rating' in params:
            filtered_df = filtered_df[filtered_df['rating'] == params['rating']]
        
        if 'sector' in params:
            filtered_df = filtered_df[filtered_df['sector'].str.lower() == params['sector'].lower()]
        
        if 'issuer_type' in params:
            filtered_df = filtered_df[filtered_df['issuer_type'].str.lower() == params['issuer_type'].lower()]
        
        # Convert to list of dictionaries
        result = filtered_df.to_dict('records')
        
        return result
    
    def _normalize_yields(self, bonds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize yields to Bond Equivalent Yield (BEY)
        
        BEY = ((1 + r/2)^2) - 1
        
        Args:
            bonds (List[Dict[str, Any]]): List of bonds
            
        Returns:
            List[Dict[str, Any]]: Bonds with normalized yields
        """
        normalized_bonds = []
        
        for bond in bonds:
            bond_copy = bond.copy()
            
            # Get yield and payment frequency
            raw_yield = bond.get('yield', 0)
            frequency = bond.get('payment_frequency', 2)  # Default to semi-annual
            
            # Convert frequency string to number if needed
            if isinstance(frequency, str):
                frequency_map = {
                    'annual': 1,
                    'semi-annual': 2,
                    'quarterly': 4,
                    'monthly': 12
                }
                frequency = frequency_map.get(frequency.lower(), 2)
            
            # Calculate Bond Equivalent Yield (BEY)
            if frequency == 2:  # Already semi-annual
                bey = raw_yield
            elif frequency == 1:  # Annual
                bey = ((1 + raw_yield) ** 0.5) ** 2 - 1
            elif frequency == 4:  # Quarterly
                bey = ((1 + raw_yield/4) ** 4) ** 0.5 - 1
            elif frequency == 12:  # Monthly
                bey = ((1 + raw_yield/12) ** 12) ** 0.5 - 1
            else:
                bey = raw_yield  # Default fallback
            
            # Add normalized yield to bond
            bond_copy['normalized_yield'] = bey
            normalized_bonds.append(bond_copy)
        
        return normalized_bonds
    
    def _apply_platform_scoring(self, bonds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply platform reliability scoring to adjust yields
        
        Args:
            bonds (List[Dict[str, Any]]): List of bonds
            
        Returns:
            List[Dict[str, Any]]: Bonds with effective yields
        """
        scored_bonds = []
        
        for bond in bonds:
            bond_copy = bond.copy()
            
            # Get platform and normalized yield
            platform = bond.get('platform', 'Default')
            normalized_yield = bond.get('normalized_yield', bond.get('yield', 0))
            
            # Get platform weight
            weight = self.platform_weights.get(platform, self.platform_weights['Default'])
            
            # Calculate effective yield
            effective_yield = normalized_yield * weight
            
            # Add effective yield to bond
            bond_copy['effective_yield'] = effective_yield
            bond_copy['platform_reliability'] = weight
            
            scored_bonds.append(bond_copy)
        
        return scored_bonds
    
    def compare_bonds(self, isins: List[str]) -> Dict[str, Any]:
        """
        Compare specific bonds by ISIN
        
        Args:
            isins (List[str]): List of ISINs to compare
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        if self.data_processor.bonds_df is None:
            return {
                "status": "error",
                "message": "Bond data not loaded",
                "data": None
            }
        
        # Filter bonds by ISIN
        bonds = []
        for isin in isins:
            bond_data = self.data_processor.bonds_df[self.data_processor.bonds_df['isin'] == isin]
            if not bond_data.empty:
                bonds.append(bond_data.iloc[0].to_dict())
        
        if not bonds:
            return {
                "status": "error",
                "message": "No bonds found with the provided ISINs",
                "data": None
            }
        
        # Normalize yields
        normalized_bonds = self._normalize_yields(bonds)
        
        # Apply platform scoring
        scored_bonds = self._apply_platform_scoring(normalized_bonds)
        
        # Sort by effective yield
        sorted_bonds = sorted(scored_bonds, key=lambda x: x.get('effective_yield', 0), reverse=True)
        
        return {
            "status": "success",
            "message": f"Compared {len(sorted_bonds)} bonds",
            "data": sorted_bonds
        }

# Function to handle finder agent queries in the LangGraph workflow
def handle_finder(state):
    """
    Handler for finder agent in the LangGraph workflow
    
    Args:
        state: The current state
        
    Returns:
        Updated state
    """
    query = state["query"]
    
    # Initialize agent
    agent = FinderAgent()
    agent.initialize()
    
    # Check if we're comparing specific bonds
    if state.get("isin") or "compare" in query.lower():
        # Extract ISINs from query
        isins = re.findall(r'INE[A-Z0-9]{10}', query)
        
        # If ISINs found, compare them
        if isins:
            result = agent.compare_bonds(isins)
        else:
            # Process as a regular query
            result = agent.process_query(query)
    else:
        # Process as a regular query
        result = agent.process_query(query)
    
    # Update state
    state["agent_results"] = result
    
    return state 