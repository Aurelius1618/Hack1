import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from app.data.data_processor import FinancialDataProcessor
from app.utils.optimization import parallelize_cashflow_calculations, monitor_query_latency

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CashFlowAgent:
    """
    Agent for handling cash flow related queries
    """
    def __init__(self):
        """
        Initialize the cash flow agent
        """
        self.data_processor = FinancialDataProcessor()
        
    def initialize(self):
        """
        Initialize the agent by loading data
        """
        self.data_processor.load_data()
    
    def process_query(self, query: str, isin: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query related to cash flows
        
        Args:
            query (str): User query
            isin (Optional[str]): ISIN code if available
            
        Returns:
            Dict[str, Any]: Response with cash flow information
        """
        # Extract ISIN from query
        isin_match = re.search(r'INE[A-Z0-9]{10}', query)
        
        if not isin_match:
            return {
                "status": "error",
                "message": "No ISIN found in query. Please provide an ISIN to get cash flow information.",
                "data": None
            }
        
        isin = isin_match.group(0)
        
        # Check if query is about accrued interest
        if "accrued" in query.lower() or "interest" in query.lower():
            return self._get_accrued_interest(isin, query)
        
        # Otherwise, get cash flow schedule
        return self._get_cashflow_schedule(isin)
    
    def process_multiple_isins(self, isins: List[str]) -> Dict[str, Any]:
        """
        Process cash flow information for multiple ISINs in parallel
        
        Args:
            isins (List[str]): List of ISIN codes
            
        Returns:
            Dict[str, Any]: Cash flow information for each ISIN
        """
        # Use parallel processing for cash flow calculations
        results = parallelize_cashflow_calculations(
            self.data_processor.get_cashflow_schedule,
            isins
        )
        
        # Calculate WAL for each ISIN in parallel
        wal_results = parallelize_cashflow_calculations(
            self.data_processor.calculate_wal,
            isins
        )
        
        # Combine results
        combined_results = {}
        for isin in isins:
            if isin in results and results[isin] is not None:
                combined_results[isin] = {
                    "cashflow_schedule": results[isin],
                    "wal": wal_results.get(isin)
                }
        
        return {
            "status": "success",
            "message": f"Cash flow information for {len(combined_results)} ISINs",
            "data": combined_results
        }
    
    @monitor_query_latency
    def _get_cashflow_schedule(self, isin: str) -> Dict[str, Any]:
        """
        Get cash flow schedule for a bond
        
        Args:
            isin (str): ISIN of the bond
            
        Returns:
            Dict[str, Any]: Cash flow schedule
        """
        # Get bond details
        bond_details = self.data_processor.get_bond_details(isin)
        if not bond_details:
            return {
                "status": "error",
                "message": f"No bond found with ISIN: {isin}",
                "data": None
            }
        
        # Get cash flow schedule
        cashflow_df = self.data_processor.get_cashflow_schedule(isin)
        if cashflow_df is None or cashflow_df.empty:
            return {
                "status": "error",
                "message": f"No cash flow data found for ISIN: {isin}",
                "data": None
            }
        
        # Get day count convention from bond details
        day_count_convention = bond_details.get("day_count_convention", "30/360")
        
        # Process cash flows with day count convention
        processed_cashflows = self._process_cashflows(cashflow_df, day_count_convention)
        
        # Calculate summary statistics
        total_principal = sum(cf.get("principal_amount", 0) for cf in processed_cashflows)
        total_interest = sum(cf.get("interest_amount", 0) for cf in processed_cashflows)
        total_payments = len(processed_cashflows)
        
        # Detect anomalies
        anomalies = self.data_processor.detect_cashflow_anomalies(isin)
        
        return {
            "status": "success",
            "message": f"Cash flow schedule for ISIN: {isin}",
            "data": {
                "bond_details": bond_details,
                "cashflows": processed_cashflows,
                "summary": {
                    "total_principal": total_principal,
                    "total_interest": total_interest,
                    "total_payments": total_payments,
                    "day_count_convention": day_count_convention
                },
                "anomalies": anomalies
            }
        }
    
    def _process_cashflows(self, cashflow_df: pd.DataFrame, day_count_convention: str) -> List[Dict[str, Any]]:
        """
        Process cash flows with day count convention
        
        Args:
            cashflow_df (pd.DataFrame): Cash flow data
            day_count_convention (str): Day count convention
            
        Returns:
            List[Dict[str, Any]]: Processed cash flows
        """
        processed_cashflows = []
        
        # Sort by payment date
        cashflow_df = cashflow_df.sort_values("payment_date")
        
        # Get previous payment date (or issue date)
        prev_date = None
        
        for _, row in cashflow_df.iterrows():
            payment_date = row["payment_date"]
            
            # If first payment, use issue date as previous date
            if prev_date is None:
                # Try to get issue date from bond details
                bond_details = self.data_processor.get_bond_details(row["isin"])
                if bond_details and "issue_date" in bond_details:
                    prev_date = bond_details["issue_date"]
                else:
                    # Estimate issue date as 6 months before first payment
                    prev_date = payment_date - timedelta(days=180)
            
            # Calculate days between payments based on day count convention
            days = self.calculate_days(day_count_convention, prev_date, payment_date)
            
            # Create cash flow entry
            cashflow = row.to_dict()
            cashflow["days_from_previous"] = days
            cashflow["day_count_convention"] = day_count_convention
            
            # Add to processed cash flows
            processed_cashflows.append(cashflow)
            
            # Update previous date
            prev_date = payment_date
        
        return processed_cashflows
    
    def calculate_days(self, day_count_convention: str, start_date: datetime, end_date: datetime) -> int:
        """
        Calculate days between dates based on day count convention
        
        Args:
            day_count_convention (str): Day count convention
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            int: Number of days
        """
        if day_count_convention == "30/360":
            # 30/360 convention
            # Each month is treated as having 30 days
            # Each year is treated as having 360 days
            y1, m1, d1 = start_date.year, start_date.month, min(30, start_date.day)
            y2, m2, d2 = end_date.year, end_date.month, min(30, end_date.day)
            
            return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        
        elif day_count_convention == "ACT/360":
            # ACT/360 convention
            # Actual days between dates, year treated as 360 days
            return (end_date - start_date).days
        
        elif day_count_convention == "ACT/365":
            # ACT/365 convention
            # Actual days between dates, year treated as 365 days
            return (end_date - start_date).days
        
        elif day_count_convention == "ACT/ACT":
            # ACT/ACT convention
            # Actual days between dates, year treated as actual days
            return (end_date - start_date).days
        
        else:
            # Default to actual days
            return (end_date - start_date).days
    
    def _get_accrued_interest(self, isin: str, query: str) -> Dict[str, Any]:
        """
        Get accrued interest for a bond
        
        Args:
            isin (str): ISIN of the bond
            query (str): User query
            
        Returns:
            Dict[str, Any]: Accrued interest information
        """
        # Extract settlement date from query
        from dateutil import parser
        
        # Try to extract date from query
        date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})', query)
        
        if date_match:
            try:
                settlement_date = parser.parse(date_match.group(0))
            except:
                settlement_date = datetime.now()
        else:
            # Use current date if no date specified
            settlement_date = datetime.now()
        
        # Calculate accrued interest
        accrued_interest = self.data_processor.calculate_accrued_interest(isin, settlement_date)
        
        if accrued_interest is None:
            return {
                "status": "error",
                "message": f"Could not calculate accrued interest for ISIN: {isin}",
                "data": None
            }
        
        # Get bond details
        bond_details = self.data_processor.get_bond_details(isin)
        
        return {
            "status": "success",
            "message": f"Accrued interest for ISIN: {isin} as of {settlement_date.strftime('%Y-%m-%d')}",
            "data": {
                "isin": isin,
                "settlement_date": settlement_date.strftime("%Y-%m-%d"),
                "accrued_interest": accrued_interest,
                "bond_details": bond_details
            }
        }

    def accrued_interest(self, settlement: datetime, coupon_dates: List[datetime], 
                       coupon_rate: float, principal: float, dcc: str = "30/360") -> float:
        """
        Calculate accrued interest based on day count convention
        
        Args:
            settlement (datetime): Settlement date
            coupon_dates (List[datetime]): List of coupon payment dates
            coupon_rate (float): Annual coupon rate (as decimal)
            principal (float): Bond principal amount
            dcc (str): Day count convention
            
        Returns:
            float: Accrued interest
        """
        try:
            # Import business date library if available
            try:
                from businessdate import BusinessDate, BusinessRange
                has_business_date = True
            except ImportError:
                has_business_date = False
                logger.warning("BusinessDate library not available, using standard datetime")
            
            # Find the previous and next coupon dates
            prev_date = None
            next_date = None
            
            for date in sorted(coupon_dates):
                if date <= settlement:
                    prev_date = date
                elif date > settlement and next_date is None:
                    next_date = date
            
            if prev_date is None or next_date is None:
                logger.warning("Could not determine coupon period")
                return 0.0
            
            # Calculate days based on day count convention
            if dcc == "30/360":
                # 30/360 convention
                # Each month is treated as having 30 days
                # Each year is treated as having 360 days
                y1, m1, d1 = prev_date.year, prev_date.month, min(30, prev_date.day)
                y2, m2, d2 = settlement.year, settlement.month, min(30, settlement.day)
                
                days_in_period = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
                
                # Calculate total days in coupon period
                y3, m3, d3 = next_date.year, next_date.month, min(30, next_date.day)
                total_days = 360 * (y3 - y1) + 30 * (m3 - m1) + (d3 - d1)
                
            elif dcc == "ACT/ACT":
                # ACT/ACT convention
                if has_business_date:
                    # Use BusinessDate for more accurate calculations
                    bd_prev = BusinessDate(prev_date)
                    bd_settlement = BusinessDate(settlement)
                    bd_next = BusinessDate(next_date)
                    
                    days_in_period = (bd_settlement - bd_prev).days
                    total_days = (bd_next - bd_prev).days
                else:
                    # Fallback to standard datetime
                    days_in_period = (settlement - prev_date).days
                    total_days = (next_date - prev_date).days
            
            elif dcc == "ACT/360":
                # ACT/360 convention
                days_in_period = (settlement - prev_date).days
                total_days = 360
            
            elif dcc == "ACT/365":
                # ACT/365 convention
                days_in_period = (settlement - prev_date).days
                total_days = 365
            
            else:
                # Default to actual days
                days_in_period = (settlement - prev_date).days
                total_days = (next_date - prev_date).days
            
            # Calculate accrued interest
            accrued_interest = (days_in_period / total_days) * coupon_rate * principal
            
            return accrued_interest
        
        except Exception as e:
            logger.error(f"Error calculating accrued interest: {str(e)}")
            return 0.0

# Function to handle cash flow agent queries in the LangGraph workflow
def handle_cashflow(state):
    """
    Handler for cash flow agent in the LangGraph workflow
    
    Args:
        state: The current state
        
    Returns:
        Updated state
    """
    query = state["query"]
    
    # Initialize agent
    agent = CashFlowAgent()
    agent.initialize()
    
    # Process query
    result = agent.process_query(query)
    
    # Update state
    state["agent_results"] = result
    
    return state 