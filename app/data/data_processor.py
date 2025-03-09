import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from rapidfuzz import fuzz
from sklearn.ensemble import IsolationForest
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialDataProcessor:
    """
    Process financial data for the Tap Bonds application
    """
    def __init__(self, data_dir: str = "data_dump"):
        """
        Initialize the data processor
        
        Args:
            data_dir (str): Directory containing the data files
        """
        self.data_dir = data_dir
        self.bonds_df = None
        self.company_df = None
        self.cashflow_df = None
        
    def load_data(self) -> None:
        """
        Load data from CSV files
        """
        try:
            # Load bonds data
            bonds_file = self._get_latest_file("bonds_details")
            logger.info(f"Loading bonds data from {bonds_file}")
            self.bonds_df = pd.read_csv(os.path.join(self.data_dir, bonds_file))
            
            # Load company data
            company_file = self._get_latest_file("company_insights")
            logger.info(f"Loading company data from {company_file}")
            self.company_df = pd.read_csv(os.path.join(self.data_dir, company_file))
            
            # Load cashflow data
            cashflow_file = self._get_latest_file("cashflows")
            logger.info(f"Loading cashflow data from {cashflow_file}")
            self.cashflow_df = pd.read_csv(os.path.join(self.data_dir, cashflow_file))
            
            # Perform basic data cleaning
            self._clean_data()
            
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _get_latest_file(self, prefix: str) -> str:
        """
        Get the latest file with the given prefix
        
        Args:
            prefix (str): File prefix to search for
            
        Returns:
            str: Filename of the latest file
        """
        files = [f for f in os.listdir(self.data_dir) if f.startswith(prefix)]
        if not files:
            raise FileNotFoundError(f"No files found with prefix {prefix}")
        
        # Sort by date in filename (assuming format prefix_YYYYMMDDHHMM.csv)
        return sorted(files)[-1]
    
    def _clean_data(self) -> None:
        """
        Clean the loaded data
        """
        if self.bonds_df is not None:
            # Basic cleaning for bonds data
            self.bonds_df = self.bonds_df.fillna({
                'yield': 0.0,
                'rating': 'NR',
                'maturity_date': pd.Timestamp.now() + pd.DateOffset(years=5)
            })
        
        if self.company_df is not None:
            # Basic cleaning for company data
            self.company_df = self.company_df.fillna(0)
            
            # Calculate financial ratios
            self._calculate_financial_ratios()
        
        if self.cashflow_df is not None:
            # Basic cleaning for cashflow data
            self.cashflow_df['payment_date'] = pd.to_datetime(self.cashflow_df['payment_date'])
            self.cashflow_df = self.cashflow_df.sort_values(['isin', 'payment_date'])
            
    def _calculate_financial_ratios(self) -> None:
        """
        Calculate critical financial ratios for company insights
        """
        if self.company_df is None:
            logger.warning("Company data not loaded, cannot calculate financial ratios")
            return
            
        # Calculate Interest Coverage Ratio = EBIT / Interest Expense
        self.company_df['interest_coverage_ratio'] = self.company_df['ebit'] / self.company_df['interest_expense'].replace(0, 0.001)
        
        # Calculate Debt Service Coverage Ratio = (Net Income + Depreciation) / Total Debt
        self.company_df['debt_service_coverage_ratio'] = (self.company_df['net_income'] + self.company_df['depreciation']) / self.company_df['total_debt'].replace(0, 0.001)
        
        # Calculate Current Ratio = Current Assets / Current Liabilities
        self.company_df['current_ratio'] = self.company_df['current_assets'] / self.company_df['current_liabilities'].replace(0, 0.001)
        
        # Calculate Debt to Equity Ratio = Total Debt / Shareholders Equity
        self.company_df['debt_to_equity'] = self.company_df['total_debt'] / self.company_df['shareholders_equity'].replace(0, 0.001)
        
        # Calculate Return on Assets = Net Income / Total Assets
        self.company_df['return_on_assets'] = self.company_df['net_income'] / self.company_df['total_assets'].replace(0, 0.001)
        
        # Calculate Return on Equity = Net Income / Shareholders Equity
        self.company_df['return_on_equity'] = self.company_df['net_income'] / self.company_df['shareholders_equity'].replace(0, 0.001)
        
        # Calculate Profit Margin = Net Income / Revenue
        self.company_df['profit_margin'] = self.company_df['net_income'] / self.company_df['revenue'].replace(0, 0.001)
        
        # Calculate Asset Turnover = Revenue / Total Assets
        self.company_df['asset_turnover'] = self.company_df['revenue'] / self.company_df['total_assets'].replace(0, 0.001)
        
        logger.info("Calculated financial ratios for company data")
    
    def entity_resolution(self, threshold: float = 85.0) -> Dict[str, str]:
        """
        Perform entity resolution between bonds and companies
        
        Args:
            threshold (float): Matching threshold for fuzzy matching
            
        Returns:
            Dict[str, str]: Mapping from ISIN to company ID
        """
        if self.bonds_df is None or self.company_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        entity_map = {}
        
        # Get unique issuers from bonds data
        issuers = self.bonds_df['issuer'].unique()
        
        # Get company names from company data
        companies = self.company_df['name'].to_dict()
        
        # Match issuers to companies
        for issuer in issuers:
            best_match = None
            best_score = 0
            
            for company_id, company_name in companies.items():
                match_ratio = fuzz.ratio(issuer, company_name)
                if match_ratio > best_score and match_ratio > threshold:
                    best_score = match_ratio
                    best_match = company_id
            
            if best_match:
                # Map all ISINs for this issuer to the company ID
                for isin in self.bonds_df[self.bonds_df['issuer'] == issuer]['isin']:
                    entity_map[isin] = best_match
        
        logger.info(f"Entity resolution complete. Mapped {len(entity_map)} ISINs to companies.")
        return entity_map
    
    def calculate_weighted_average_life(self, df: pd.DataFrame) -> float:
        """
        Calculate Weighted Average Life (WAL) for a bond
        
        WAL = Σ(t_i × CF_i) / Σ(CF_i)
        
        Args:
            df (pd.DataFrame): Cash flow data with payment_date and amount columns
            
        Returns:
            float: Weighted Average Life in years
        """
        if df is None or df.empty:
            return None
        
        # Ensure payment_date is datetime
        if not pd.api.types.is_datetime64_dtype(df['payment_date']):
            df['payment_date'] = pd.to_datetime(df['payment_date'])
        
        # Calculate days from now to each payment date
        now = datetime.now()
        df['days'] = (df['payment_date'] - now).dt.days
        
        # Convert days to years
        df['years'] = df['days'] / 365.25
        
        # Calculate weighted average
        total_cf = df['amount'].sum()
        if total_cf == 0:
            return None
        
        wal = (df['years'] * df['amount']).sum() / total_cf
        
        return wal
    
    def detect_cashflow_anomalies(self, isin: str) -> List[int]:
        """
        Detect anomalies in cashflow data using Isolation Forest
        
        Args:
            isin (str): ISIN of the bond
            
        Returns:
            List[int]: Indices of anomalous cashflows
        """
        if self.cashflow_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Filter cashflows for the given ISIN
        bond_cashflows = self.cashflow_df[self.cashflow_df['isin'] == isin]
        
        if bond_cashflows.empty or len(bond_cashflows) < 5:
            logger.warning(f"Insufficient cashflow data for anomaly detection for ISIN {isin}")
            return []
        
        # Extract features for anomaly detection
        features = bond_cashflows[['amount']].copy()
        
        # Add time difference between payments as a feature
        payment_dates = bond_cashflows['payment_date'].sort_values()
        time_diffs = payment_dates.diff().dt.days.fillna(0)
        features['time_diff'] = time_diffs.values
        
        # Fit Isolation Forest
        clf = IsolationForest(contamination=0.1, random_state=42)
        predictions = clf.fit_predict(features)
        
        # Get indices of anomalies (-1 indicates an anomaly)
        anomaly_indices = bond_cashflows.index[predictions == -1].tolist()
        
        return anomaly_indices
    
    def calculate_effective_yield(self, base_yield: float, platform: str) -> float:
        """
        Calculate effective yield based on platform reliability
        
        Args:
            base_yield (float): Base yield
            platform (str): Trading platform
            
        Returns:
            float: Effective yield
        """
        platform_weights = {
            "SMEST": 0.85,
            "FixedIncome": 0.78,
            "Institutional": 0.92,
            "Default": 0.80  # Default weight for unknown platforms
        }
        
        weight = platform_weights.get(platform, platform_weights["Default"])
        
        return base_yield * weight
    
    def validate_isin(self, isin: str) -> bool:
        """
        Validate an ISIN code
        
        Args:
            isin (str): ISIN code to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        import re
        
        # Check format (for Indian ISINs)
        pattern = r"^INE[A-Z0-9]{10}$"
        if not re.match(pattern, isin):
            return False
        
        # Check if ISIN exists in our database
        if self.bonds_df is not None:
            return isin in self.bonds_df['isin'].values
        
        return True
    
    def get_bond_details(self, isin: str) -> Optional[Dict]:
        """
        Get details for a bond
        
        Args:
            isin (str): ISIN of the bond
            
        Returns:
            Dict: Bond details
        """
        if self.bonds_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        bond_data = self.bonds_df[self.bonds_df['isin'] == isin]
        
        if bond_data.empty:
            logger.warning(f"No data found for ISIN {isin}")
            return None
        
        # Convert to dictionary
        bond_dict = bond_data.iloc[0].to_dict()
        
        # Add WAL
        bond_dict['wal'] = self.calculate_weighted_average_life(self.cashflow_df[self.cashflow_df['isin'] == isin])
        
        return bond_dict
    
    def get_cashflow_schedule(self, isin: str) -> Optional[pd.DataFrame]:
        """
        Get cashflow schedule for a bond
        
        Args:
            isin (str): ISIN of the bond
            
        Returns:
            pd.DataFrame: Cashflow schedule
        """
        if self.cashflow_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        cashflows = self.cashflow_df[self.cashflow_df['isin'] == isin]
        
        if cashflows.empty:
            logger.warning(f"No cashflow data found for ISIN {isin}")
            return None
        
        # Sort by payment date
        cashflows = cashflows.sort_values('payment_date')
        
        # Detect anomalies
        anomaly_indices = self.detect_cashflow_anomalies(isin)
        cashflows['is_anomaly'] = cashflows.index.isin(anomaly_indices)
        
        return cashflows
    
    def calculate_accrued_interest(self, isin: str, settlement_date: datetime) -> Optional[float]:
        """
        Calculate accrued interest for a bond
        
        Args:
            isin (str): ISIN of the bond
            settlement_date (datetime): Settlement date
            
        Returns:
            float: Accrued interest
        """
        if self.bonds_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        bond_data = self.bonds_df[self.bonds_df['isin'] == isin]
        
        if bond_data.empty:
            logger.warning(f"No data found for ISIN {isin}")
            return None
        
        # Get bond details
        bond = bond_data.iloc[0]
        
        # Get day count convention (default to 30/360)
        dcc = bond.get('day_count', '30/360')
        
        # Get issue date and coupon rate
        issue_date = bond.get('issue_date')
        if issue_date is None:
            logger.warning(f"No issue date found for ISIN {isin}")
            return None
        
        rate = bond.get('coupon_rate', 0) / 100  # Convert percentage to decimal
        principal = bond.get('face_value', 1000)
        
        # Calculate days based on day count convention
        if dcc == '30/360':
            # 30/360 convention
            years = (settlement_date.year - issue_date.year)
            months = (settlement_date.month - issue_date.month)
            days = min(30, settlement_date.day) - min(30, issue_date.day)
            
            total_days = years * 360 + months * 30 + days
            days_in_year = 360
        else:
            # ACT/ACT convention
            total_days = (settlement_date - issue_date).days
            days_in_year = 365
        
        # Calculate accrued interest
        accrued_interest = principal * rate * total_days / days_in_year
        
        return accrued_interest
    
    def calculate_z_score(self, row) -> float:
        """
        Calculate Altman Z-Score for a company
        
        Z-Score = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MVE/TL + 1.0*S/TA
        
        Where:
        - WC = Working Capital
        - TA = Total Assets
        - RE = Retained Earnings
        - EBIT = Earnings Before Interest and Taxes
        - MVE = Market Value of Equity
        - TL = Total Liabilities
        - S = Sales (Revenue)
        
        Args:
            row: Row from company dataframe
            
        Returns:
            float: Z-Score
        """
        try:
            # Calculate ratios
            wc_ta = row['working_capital'] / row['total_assets'] if row['total_assets'] != 0 else 0
            re_ta = row['retained_earnings'] / row['total_assets'] if row['total_assets'] != 0 else 0
            ebit_ta = row['ebit'] / row['total_assets'] if row['total_assets'] != 0 else 0
            
            # For MVE/TL, we'll use a proxy since we might not have market value
            # Using shareholders_equity as a proxy for MVE
            mve_tl = row['shareholders_equity'] / row['total_liabilities'] if row['total_liabilities'] != 0 else 0
            
            # Revenue to Total Assets
            s_ta = row['revenue'] / row['total_assets'] if row['total_assets'] != 0 else 0
            
            # Calculate Z-Score
            z_score = 1.2*wc_ta + 1.4*re_ta + 3.3*ebit_ta + 0.6*mve_tl + 1.0*s_ta
            
            return z_score
        except Exception as e:
            logger.error(f"Error calculating Z-Score: {str(e)}")
            return 0.0 