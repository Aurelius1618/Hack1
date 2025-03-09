import logging
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, date
from app.data.data_processor import FinancialDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YieldCalculatorAgent:
    """
    Agent for calculating bond yields, prices, and related metrics
    """
    def __init__(self):
        """
        Initialize the yield calculator agent
        """
        self.data_processor = FinancialDataProcessor()
        
    def initialize(self):
        """
        Initialize the agent by loading data
        """
        self.data_processor.load_data()
    
    def process_query(self, query: str, isin: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query related to yield calculations
        
        Args:
            query (str): User query
            isin (Optional[str]): ISIN code if available
            
        Returns:
            Dict[str, Any]: Response with yield calculation information
        """
        # Extract ISIN from query if not provided
        if not isin:
            isin_match = re.search(r'INE[A-Z0-9]{10}', query)
            if isin_match:
                isin = isin_match.group(0)
        
        if not isin:
            return {
                "status": "error",
                "message": "No ISIN found in query. Please provide an ISIN to calculate yield.",
                "data": None
            }
        
        # Check if query is about YTM
        if re.search(r'ytm|yield to maturity', query.lower()):
            return self._calculate_ytm(isin, query)
        
        # Check if query is about price calculation
        if re.search(r'price|clean price|dirty price', query.lower()):
            return self._calculate_price(isin, query)
        
        # Check if query is about duration
        if re.search(r'duration|macaulay|modified', query.lower()):
            return self._calculate_duration(isin, query)
        
        # Default to YTM calculation
        return self._calculate_ytm(isin, query)
    
    def _calculate_ytm(self, isin: str, query: str) -> Dict[str, Any]:
        """
        Calculate Yield to Maturity for a bond
        
        Args:
            isin (str): ISIN code
            query (str): User query
            
        Returns:
            Dict[str, Any]: YTM calculation results
        """
        # Get bond details
        bond_details = self.data_processor.get_bond_details(isin)
        if not bond_details:
            return {
                "status": "error",
                "message": f"Bond details not found for ISIN: {isin}",
                "data": None
            }
        
        # Get cashflow schedule
        cashflow_df = self.data_processor.get_cashflow_schedule(isin)
        if cashflow_df is None or cashflow_df.empty:
            return {
                "status": "error",
                "message": f"Cashflow schedule not found for ISIN: {isin}",
                "data": None
            }
        
        # Extract price from query if available
        price_match = re.search(r'price\s+(?:of|is|at)?\s*(\d+(?:\.\d+)?)', query, re.IGNORECASE)
        price = float(price_match.group(1)) if price_match else bond_details.get('current_price', 100.0)
        
        # Prepare cashflows for YTM calculation
        cashflows = []
        for _, row in cashflow_df.iterrows():
            payment_date = pd.to_datetime(row['payment_date']).date()
            amount = row['amount']
            years = (payment_date - datetime.now().date()).days / 365.0
            if years > 0:  # Only consider future cashflows
                cashflows.append({
                    'date': payment_date,
                    'amount': amount,
                    'years': years
                })
        
        # Calculate YTM using Newton-Raphson method
        ytm = self._calculate_ytm_newton(price, cashflows)
        
        return {
            "status": "success",
            "message": f"Yield to Maturity calculated for {bond_details.get('issuer_name', 'Unknown')} bond",
            "data": {
                "isin": isin,
                "issuer": bond_details.get('issuer_name', 'Unknown'),
                "ytm": ytm * 100,  # Convert to percentage
                "price": price,
                "coupon_rate": bond_details.get('coupon_rate', 0) * 100,  # Convert to percentage
                "maturity_date": bond_details.get('maturity_date', 'Unknown'),
                "day_count_convention": bond_details.get('day_count_convention', '30/360')
            }
        }
    
    def _calculate_ytm_newton(self, price: float, cashflows: List[Dict]) -> float:
        """
        Calculate YTM using Newton-Raphson method
        
        Args:
            price (float): Bond price
            cashflows (List[Dict]): List of cashflows
            
        Returns:
            float: Yield to Maturity
        """
        # Initial guess
        ytm = 0.05
        tolerance = 1e-10
        max_iterations = 100
        
        for _ in range(max_iterations):
            # Calculate price and derivative at current YTM
            price_at_ytm = self.calculate_clean_price(ytm, cashflows)
            price_derivative = self._calculate_price_derivative(ytm, cashflows)
            
            # Calculate error
            error = price_at_ytm - price
            
            # Check if error is within tolerance
            if abs(error) < tolerance:
                return ytm
            
            # Update YTM
            ytm = ytm - error / price_derivative
            
            # Ensure YTM is positive
            ytm = max(0.001, ytm)
        
        return ytm
    
    def calculate_clean_price(self, ytm: float, cashflows: List[Dict]) -> float:
        """
        Calculate clean price of a bond given YTM and cashflows
        
        Args:
            ytm (float): Yield to Maturity
            cashflows (List[Dict]): List of cashflows
            
        Returns:
            float: Clean price
        """
        return sum(
            cf['amount'] / (1 + ytm)**(cf['years']) 
            for cf in cashflows
        )
    
    def _calculate_price_derivative(self, ytm: float, cashflows: List[Dict]) -> float:
        """
        Calculate derivative of price with respect to YTM
        
        Args:
            ytm (float): Yield to Maturity
            cashflows (List[Dict]): List of cashflows
            
        Returns:
            float: Price derivative
        """
        return -sum(
            cf['years'] * cf['amount'] / (1 + ytm)**(cf['years'] + 1)
            for cf in cashflows
        )
    
    def _calculate_price(self, isin: str, query: str) -> Dict[str, Any]:
        """
        Calculate clean and dirty prices for a bond
        
        Args:
            isin (str): ISIN code
            query (str): User query
            
        Returns:
            Dict[str, Any]: Price calculation results
        """
        # Get bond details
        bond_details = self.data_processor.get_bond_details(isin)
        if not bond_details:
            return {
                "status": "error",
                "message": f"Bond details not found for ISIN: {isin}",
                "data": None
            }
        
        # Get cashflow schedule
        cashflow_df = self.data_processor.get_cashflow_schedule(isin)
        if cashflow_df is None or cashflow_df.empty:
            return {
                "status": "error",
                "message": f"Cashflow schedule not found for ISIN: {isin}",
                "data": None
            }
        
        # Extract YTM from query if available
        ytm_match = re.search(r'ytm\s+(?:of|is|at)?\s*(\d+(?:\.\d+)?)\s*%?', query, re.IGNORECASE)
        ytm = float(ytm_match.group(1))/100 if ytm_match else bond_details.get('yield', 0.05)
        
        # Prepare cashflows for price calculation
        cashflows = []
        for _, row in cashflow_df.iterrows():
            payment_date = pd.to_datetime(row['payment_date']).date()
            amount = row['amount']
            years = (payment_date - datetime.now().date()).days / 365.0
            if years > 0:  # Only consider future cashflows
                cashflows.append({
                    'date': payment_date,
                    'amount': amount,
                    'years': years
                })
        
        # Calculate clean price
        clean_price = self.calculate_clean_price(ytm, cashflows)
        
        # Calculate accrued interest
        settlement_date = datetime.now().date()
        accrued_interest = self.data_processor.calculate_accrued_interest(isin, settlement_date)
        
        # Calculate dirty price
        dirty_price = clean_price + (accrued_interest if accrued_interest else 0)
        
        return {
            "status": "success",
            "message": f"Bond prices calculated for {bond_details.get('issuer_name', 'Unknown')} bond",
            "data": {
                "isin": isin,
                "issuer": bond_details.get('issuer_name', 'Unknown'),
                "ytm": ytm * 100,  # Convert to percentage
                "clean_price": clean_price,
                "accrued_interest": accrued_interest if accrued_interest else 0,
                "dirty_price": dirty_price,
                "coupon_rate": bond_details.get('coupon_rate', 0) * 100,  # Convert to percentage
                "maturity_date": bond_details.get('maturity_date', 'Unknown'),
                "day_count_convention": bond_details.get('day_count_convention', '30/360')
            }
        }
    
    def _calculate_duration(self, isin: str, query: str) -> Dict[str, Any]:
        """
        Calculate Macaulay and Modified Duration for a bond
        
        Args:
            isin (str): ISIN code
            query (str): User query
            
        Returns:
            Dict[str, Any]: Duration calculation results
        """
        # Get bond details
        bond_details = self.data_processor.get_bond_details(isin)
        if not bond_details:
            return {
                "status": "error",
                "message": f"Bond details not found for ISIN: {isin}",
                "data": None
            }
        
        # Get cashflow schedule
        cashflow_df = self.data_processor.get_cashflow_schedule(isin)
        if cashflow_df is None or cashflow_df.empty:
            return {
                "status": "error",
                "message": f"Cashflow schedule not found for ISIN: {isin}",
                "data": None
            }
        
        # Extract YTM from query if available
        ytm_match = re.search(r'ytm\s+(?:of|is|at)?\s*(\d+(?:\.\d+)?)\s*%?', query, re.IGNORECASE)
        ytm = float(ytm_match.group(1))/100 if ytm_match else bond_details.get('yield', 0.05)
        
        # Prepare cashflows for duration calculation
        cashflows = []
        for _, row in cashflow_df.iterrows():
            payment_date = pd.to_datetime(row['payment_date']).date()
            amount = row['amount']
            years = (payment_date - datetime.now().date()).days / 365.0
            if years > 0:  # Only consider future cashflows
                cashflows.append({
                    'date': payment_date,
                    'amount': amount,
                    'years': years
                })
        
        # Calculate clean price
        price = self.calculate_clean_price(ytm, cashflows)
        
        # Calculate Macaulay Duration
        macaulay_duration = sum(
            cf['years'] * cf['amount'] / (1 + ytm)**(cf['years'])
            for cf in cashflows
        ) / price
        
        # Calculate Modified Duration
        modified_duration = macaulay_duration / (1 + ytm)
        
        return {
            "status": "success",
            "message": f"Duration calculated for {bond_details.get('issuer_name', 'Unknown')} bond",
            "data": {
                "isin": isin,
                "issuer": bond_details.get('issuer_name', 'Unknown'),
                "ytm": ytm * 100,  # Convert to percentage
                "price": price,
                "macaulay_duration": macaulay_duration,
                "modified_duration": modified_duration,
                "coupon_rate": bond_details.get('coupon_rate', 0) * 100,  # Convert to percentage
                "maturity_date": bond_details.get('maturity_date', 'Unknown')
            }
        }

def handle_yield_calculator(state):
    """
    Handle yield calculator agent state
    
    Args:
        state: Current state
        
    Returns:
        Updated state
    """
    query = state["query"]
    isin = state.get("isin")
    
    # Initialize agent
    agent = YieldCalculatorAgent()
    agent.initialize()
    
    # Process query
    result = agent.process_query(query, isin)
    
    # Update state
    state["results"] = result
    state["agent_results"] = result.get("data", {})
    
    # Add to reasoning chain
    state["reasoning_chain"].append(f"Processed yield calculation for {isin if isin else 'bond'}")
    
    return state 