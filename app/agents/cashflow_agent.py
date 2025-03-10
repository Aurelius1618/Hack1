import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, date
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
        
        # Define day count conventions
        self.day_count_conventions = {
            "30/360": self._calculate_30_360_days,
            "ACT/ACT": self._calculate_act_act_days,
            "ACT/360": self._calculate_act_360_days,
            "ACT/365": self._calculate_act_365_days
        }
        
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
        accrued_interest = self.calculate_accrued_interest(isin, settlement_date)
        
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

    def calculate_accrued_interest(self, isin: str, settlement_date: datetime, day_count_convention: str = "30/360") -> float:
        """
        Calculate accrued interest for a bond
        
        Args:
            isin (str): ISIN code
            settlement_date (datetime): Settlement date
            day_count_convention (str): Day count convention (30/360, ACT/ACT, ACT/360, ACT/365)
            
        Returns:
            float: Accrued interest
        """
        # Get bond details
        bond_details = self.data_processor.get_bond_details(isin)
        if not bond_details:
            return 0.0
        
        # Get cashflow schedule
        cashflow_df = self.data_processor.get_cashflow_schedule(isin)
        if cashflow_df is None or cashflow_df.empty:
            return 0.0
        
        # Get coupon rate and payment frequency
        coupon_rate = bond_details.get('coupon_rate', 0.0)
        payment_frequency = bond_details.get('payment_frequency', 2)  # Default to semi-annual
        
        # Find the previous and next coupon dates
        cashflow_dates = pd.to_datetime(cashflow_df['payment_date']).tolist()
        
        # Convert settlement_date to datetime if it's a string
        if isinstance(settlement_date, str):
            settlement_date = pd.to_datetime(settlement_date)
        
        # Find previous and next coupon dates
        previous_date = None
        next_date = None
        
        for date in cashflow_dates:
            if date < settlement_date:
                previous_date = date
            elif date >= settlement_date and next_date is None:
                next_date = date
        
        if previous_date is None or next_date is None:
            return 0.0
        
        # Calculate days based on day count convention
        if day_count_convention == "30/360":
            # 30/360 convention
            days_in_period = 360 * (next_date.year - previous_date.year) + 30 * (next_date.month - previous_date.month) + (min(30, next_date.day) - min(30, previous_date.day))
            days_accrued = 360 * (settlement_date.year - previous_date.year) + 30 * (settlement_date.month - previous_date.month) + (min(30, settlement_date.day) - min(30, previous_date.day))
        elif day_count_convention == "ACT/ACT":
            # ACT/ACT convention
            days_in_period = (next_date - previous_date).days
            days_accrued = (settlement_date - previous_date).days
        elif day_count_convention == "ACT/360":
            # ACT/360 convention
            days_in_period = (next_date - previous_date).days
            days_accrued = (settlement_date - previous_date).days
            days_in_period = 360  # Fixed denominator
        elif day_count_convention == "ACT/365":
            # ACT/365 convention
            days_in_period = (next_date - previous_date).days
            days_accrued = (settlement_date - previous_date).days
            days_in_period = 365  # Fixed denominator
        else:
            # Default to 30/360
            days_in_period = 360 * (next_date.year - previous_date.year) + 30 * (next_date.month - previous_date.month) + (min(30, next_date.day) - min(30, previous_date.day))
            days_accrued = 360 * (settlement_date.year - previous_date.year) + 30 * (settlement_date.month - previous_date.month) + (min(30, settlement_date.day) - min(30, previous_date.day))
        
        # Calculate accrued interest
        coupon_amount = bond_details.get('face_value', 100.0) * coupon_rate
        accrued_interest = coupon_amount * (days_accrued / days_in_period)
        
        return accrued_interest

    def _calculate_30_360_days(self, start_date: date, end_date: date) -> int:
        """
        Calculate days using 30/360 convention
        
        Args:
            start_date (date): Start date
            end_date (date): End date
            
        Returns:
            int: Number of days
        """
        # Adjust day values
        d1 = min(start_date.day, 30)
        d2 = min(end_date.day, 30) if d1 == 30 else end_date.day
        
        # Calculate using 30/360 formula
        return (360 * (end_date.year - start_date.year) + 
                30 * (end_date.month - start_date.month) + 
                (d2 - d1))
    
    def _calculate_act_act_days(self, start_date: date, end_date: date) -> int:
        """
        Calculate days using ACT/ACT convention
        
        Args:
            start_date (date): Start date
            end_date (date): End date
            
        Returns:
            int: Number of days
        """
        return (end_date - start_date).days
    
    def _calculate_act_360_days(self, start_date: date, end_date: date) -> int:
        """
        Calculate days using ACT/360 convention
        
        Args:
            start_date (date): Start date
            end_date (date): End date
            
        Returns:
            int: Number of days
        """
        return (end_date - start_date).days
    
    def _calculate_act_365_days(self, start_date: date, end_date: date) -> int:
        """
        Calculate days using ACT/365 convention
        
        Args:
            start_date (date): Start date
            end_date (date): End date
            
        Returns:
            int: Number of days
        """
        return (end_date - start_date).days
    
    def _is_leap_year(self, year: int) -> bool:
        """
        Check if a year is a leap year
        
        Args:
            year (int): Year to check
            
        Returns:
            bool: True if leap year, False otherwise
        """
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    def calculate_accrued_interest(
        self,
        principal: float, 
        rate: float,
        settlement: date,
        last_coupon: date,
        dcc: str = "30/360"
    ) -> float:
        """
        Calculate accrued interest
        
        Args:
            principal (float): Principal amount
            rate (float): Interest rate (as decimal)
            settlement (date): Settlement date
            last_coupon (date): Last coupon date
            dcc (str): Day count convention
            
        Returns:
            float: Accrued interest
        """
        # Get day count function
        day_count_func = self.day_count_conventions.get(dcc, self.day_count_conventions["30/360"])
        
        # Calculate days
        days = day_count_func(last_coupon, settlement)
        
        # Calculate accrued interest
        if dcc == "30/360":
            return principal * rate * days / 360
        elif dcc == "ACT/360":
            return principal * rate * days / 360
        elif dcc == "ACT/365":
            return principal * rate * days / 365
        elif dcc == "ACT/ACT":
            # For ACT/ACT, we need to handle leap years
            if self._is_leap_year(last_coupon.year) and self._is_leap_year(settlement.year):
                return principal * rate * days / 366
            elif not self._is_leap_year(last_coupon.year) and not self._is_leap_year(settlement.year):
                return principal * rate * days / 365
            else:
                # If dates span different years with different day counts
                days_in_leap_year = 0
                days_in_regular_year = 0
                
                if self._is_leap_year(last_coupon.year):
                    # Count days in leap year
                    year_end = date(last_coupon.year, 12, 31)
                    days_in_leap_year = (min(year_end, settlement) - last_coupon).days
                    days_in_regular_year = days - days_in_leap_year
                else:
                    # Count days in regular year
                    year_end = date(last_coupon.year, 12, 31)
                    days_in_regular_year = (min(year_end, settlement) - last_coupon).days
                    days_in_leap_year = days - days_in_regular_year
                
                return principal * rate * (days_in_leap_year / 366 + days_in_regular_year / 365)
        
        # Default fallback
        return principal * rate * days / 365

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