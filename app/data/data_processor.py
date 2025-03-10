import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from rapidfuzz import fuzz
from sklearn.ensemble import IsolationForest
from datetime import datetime
import logging
import json

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
            # Convert date columns to datetime
            date_columns = ['created_at', 'updated_at', 'allotment_date', 'maturity_date']
            for col in date_columns:
                if col in self.bonds_df.columns:
                    self.bonds_df[col] = pd.to_datetime(self.bonds_df[col], errors='coerce')
            
            # Basic cleaning for bonds data
            self.bonds_df = self.bonds_df.fillna({
                'isin': '',
                'company_name': 'Unknown',
                'issue_size': 0.0,
                'maturity_date': pd.Timestamp.now() + pd.DateOffset(years=5)
            })
        
        if self.company_df is not None:
            # Convert date columns to datetime
            date_columns = ['created_at', 'updated_at']
            for col in date_columns:
                if col in self.company_df.columns:
                    self.company_df[col] = pd.to_datetime(self.company_df[col], errors='coerce')
            
            # Basic cleaning for company data
            self.company_df = self.company_df.fillna({
                'company_name': 'Unknown',
                'company_industry': 'Unknown',
                'description': '',
                'key_metrics': '',
                'pros': '',
                'cons': ''
            })
            
            # Calculate financial ratios
            self._calculate_financial_ratios()
        
        if self.cashflow_df is not None:
            # Convert date columns to datetime
            date_columns = ['created_at', 'updated_at', 'cash_flow_date', 'record_date']
            for col in date_columns:
                if col in self.cashflow_df.columns:
                    self.cashflow_df[col] = pd.to_datetime(self.cashflow_df[col], errors='coerce')
            
            # Convert numeric columns
            numeric_columns = ['cash_flow_amount', 'principal_amount', 'interest_amount', 
                              'tds_amount', 'remaining_principal']
            for col in numeric_columns:
                if col in self.cashflow_df.columns:
                    self.cashflow_df[col] = pd.to_numeric(self.cashflow_df[col], errors='coerce').fillna(0)
            
            # Sort cashflows by ISIN and date
            self.cashflow_df = self.cashflow_df.sort_values(['isin', 'cash_flow_date'])
            
    def _calculate_financial_ratios(self) -> None:
        """
        Calculate critical financial ratios for company insights
        """
        if self.company_df is None:
            logger.warning("Company data not loaded, cannot calculate financial ratios")
            return
        
        try:
            # The financial data is stored in JSON columns, so we need to parse them
            # Add a column to store calculated ratios
            self.company_df['calculated_ratios'] = None
            
            for idx, row in self.company_df.iterrows():
                ratios = {}
                
                # Try to extract financial metrics from the JSON columns
                try:
                    # Parse JSON columns if they're strings
                    key_metrics = row['key_metrics']
                    if isinstance(key_metrics, str) and key_metrics:
                        try:
                            key_metrics = json.loads(key_metrics)
                        except:
                            key_metrics = {}
                    
                    income_statement = row['income_statement']
                    if isinstance(income_statement, str) and income_statement:
                        try:
                            income_statement = json.loads(income_statement)
                        except:
                            income_statement = {}
                    
                    balance_sheet = row['balance_sheet']
                    if isinstance(balance_sheet, str) and balance_sheet:
                        try:
                            balance_sheet = json.loads(balance_sheet)
                        except:
                            balance_sheet = {}
                    
                    cashflow_data = row['cashflow']
                    if isinstance(cashflow_data, str) and cashflow_data:
                        try:
                            cashflow_data = json.loads(cashflow_data)
                        except:
                            cashflow_data = {}
                    
                    # Extract values needed for ratio calculations
                    # These are examples - adjust based on actual JSON structure
                    ebit = self._safe_get(income_statement, 'ebit', 0)
                    interest_expense = self._safe_get(income_statement, 'interest_expense', 0.001)
                    net_income = self._safe_get(income_statement, 'net_income', 0)
                    depreciation = self._safe_get(cashflow_data, 'depreciation', 0)
                    total_debt = self._safe_get(balance_sheet, 'total_debt', 0.001)
                    current_assets = self._safe_get(balance_sheet, 'current_assets', 0)
                    current_liabilities = self._safe_get(balance_sheet, 'current_liabilities', 0.001)
                    shareholders_equity = self._safe_get(balance_sheet, 'shareholders_equity', 0.001)
                    
                    # Calculate ratios
                    ratios['interest_coverage_ratio'] = ebit / interest_expense
                    ratios['debt_service_coverage_ratio'] = (net_income + depreciation) / total_debt
                    ratios['current_ratio'] = current_assets / current_liabilities
                    ratios['debt_to_equity'] = total_debt / shareholders_equity
                    
                    # Store calculated ratios
                    self.company_df.at[idx, 'calculated_ratios'] = json.dumps(ratios)
                    
                except Exception as e:
                    logger.warning(f"Error calculating ratios for company {row.get('company_name', 'Unknown')}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in financial ratio calculations: {str(e)}")
    
    def _safe_get(self, data, key, default=0):
        """
        Safely get a value from a dictionary, handling None values and missing keys
        
        Args:
            data (dict): Dictionary to get value from
            key (str): Key to get
            default: Default value if key is missing or None
            
        Returns:
            Value from dictionary or default
        """
        if data is None or not isinstance(data, dict):
            return default
        
        value = data.get(key, default)
        if value is None or (isinstance(value, (int, float)) and np.isnan(value)):
            return default
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
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
            df (pd.DataFrame): Cash flow data with cash_flow_date and cash_flow_amount columns
            
        Returns:
            float: Weighted Average Life in years
        """
        if df is None or df.empty:
            return None
        
        # Ensure cash_flow_date is datetime
        if not pd.api.types.is_datetime64_dtype(df['cash_flow_date']):
            df['cash_flow_date'] = pd.to_datetime(df['cash_flow_date'])
        
        # Calculate days from now to each payment date
        now = datetime.now()
        df['days'] = (df['cash_flow_date'] - now).dt.days
        
        # Convert days to years
        df['years'] = df['days'] / 365.25
        
        # Calculate weighted average
        total_cf = df['cash_flow_amount'].sum()
        if total_cf == 0:
            return None
        
        wal = (df['years'] * df['cash_flow_amount']).sum() / total_cf
        
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
        features = bond_cashflows[['cash_flow_amount']].copy()
        
        # Add time difference between payments as a feature
        payment_dates = bond_cashflows['cash_flow_date'].sort_values()
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
        
        # Parse JSON fields if they exist
        json_fields = ['issuer_details', 'instrument_details', 'coupon_details', 
                      'redemption_details', 'credit_rating_details', 'listing_details',
                      'key_contacts_details', 'key_documents_details']
        
        for field in json_fields:
            if field in bond_dict and isinstance(bond_dict[field], str):
                try:
                    bond_dict[field] = json.loads(bond_dict[field])
                except:
                    # Keep as string if not valid JSON
                    pass
        
        # Add WAL
        if self.cashflow_df is not None:
            cashflow_data = self.cashflow_df[self.cashflow_df['isin'] == isin]
            if not cashflow_data.empty:
                bond_dict['wal'] = self.calculate_weighted_average_life(cashflow_data)
        
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
        
        # Sort by cash flow date
        cashflows = cashflows.sort_values('cash_flow_date')
        
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
        
        # Parse coupon details if it's a JSON string
        coupon_details = bond.get('coupon_details', {})
        if isinstance(coupon_details, str):
            try:
                coupon_details = json.loads(coupon_details)
            except:
                coupon_details = {}
        
        # Get day count convention (default to 30/360)
        dcc = coupon_details.get('day_count_convention', '30/360')
        
        # Get allotment date and coupon rate
        allotment_date = bond.get('allotment_date')
        if allotment_date is None:
            logger.warning(f"No allotment date found for ISIN {isin}")
            return None
        
        # Ensure allotment_date is a datetime
        if not isinstance(allotment_date, datetime):
            try:
                allotment_date = pd.to_datetime(allotment_date)
            except:
                logger.warning(f"Invalid allotment date for ISIN {isin}")
                return None
        
        # Get coupon rate from coupon details
        coupon_rate = coupon_details.get('coupon_rate')
        if coupon_rate is None:
            logger.warning(f"No coupon rate found for ISIN {isin}")
            return None
        
        # Convert coupon rate to float if it's a string
        if isinstance(coupon_rate, str):
            try:
                coupon_rate = float(coupon_rate.strip('%')) / 100
            except:
                logger.warning(f"Invalid coupon rate for ISIN {isin}")
                return None
        
        # Calculate accrued interest based on day count convention
        if dcc == '30/360':
            # 30/360 convention
            days_in_year = 360
            days_elapsed = self._calculate_30_360_days(allotment_date, settlement_date)
        elif dcc == 'ACT/365':
            # Actual/365 convention
            days_in_year = 365
            days_elapsed = (settlement_date - allotment_date).days
        elif dcc == 'ACT/ACT':
            # Actual/Actual convention
            days_in_year = 366 if self._is_leap_year(settlement_date.year) else 365
            days_elapsed = (settlement_date - allotment_date).days
        else:
            # Default to Actual/360
            days_in_year = 360
            days_elapsed = (settlement_date - allotment_date).days
        
        # Calculate accrued interest
        accrued_interest = (coupon_rate * days_elapsed) / days_in_year
        
        return accrued_interest
    
    def _calculate_30_360_days(self, start_date: datetime, end_date: datetime) -> int:
        """
        Calculate days between two dates using 30/360 convention
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            int: Number of days
        """
        y1, m1, d1 = start_date.year, start_date.month, start_date.day
        y2, m2, d2 = end_date.year, end_date.month, end_date.day
        
        # Adjust for end of month
        if d1 == 31:
            d1 = 30
        if d2 == 31 and d1 >= 30:
            d2 = 30
            
        return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1))
    
    def _is_leap_year(self, year: int) -> bool:
        """
        Check if a year is a leap year
        
        Args:
            year (int): Year to check
            
        Returns:
            bool: True if leap year, False otherwise
        """
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    def calculate_z_score(self, row) -> float:
        """
        Calculate Altman Z-Score for financial health assessment
        
        Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        
        Where:
        - A = Working Capital/Total Assets
        - B = Retained Earnings/Total Assets
        - C = EBIT/Total Assets
        - D = Market Value of Equity/Total Liabilities
        - E = Sales/Total Assets
        
        Args:
            row: DataFrame row with financial data
            
        Returns:
            float: Altman Z-Score
        """
        try:
            # Extract required financial metrics
            working_capital = row.get('current_assets', 0) - row.get('current_liabilities', 0)
            total_assets = row.get('total_assets', 1)  # Avoid division by zero
            retained_earnings = row.get('retained_earnings', 0)
            ebit = row.get('ebit', 0)
            market_value_equity = row.get('market_cap', 0)
            total_liabilities = row.get('total_liabilities', 1)  # Avoid division by zero
            sales = row.get('revenue', 0)
            
            # Calculate components
            A = working_capital / total_assets
            B = retained_earnings / total_assets
            C = ebit / total_assets
            D = market_value_equity / total_liabilities
            E = sales / total_assets
            
            # Calculate Z-Score
            z_score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
            
            return z_score
        except Exception as e:
            logger.error(f"Error calculating Z-Score: {str(e)}")
            return 0.0
    
    def validate_ebit_inputs(self, ebit=None, interest=None, tax_rate=None, shares=None):
        """
        Validate inputs for EBIT-related calculations
        
        Args:
            ebit (float, optional): Earnings Before Interest and Taxes
            interest (float, optional): Interest payments
            tax_rate (float, optional): Tax rate (as decimal)
            shares (int, optional): Number of shares outstanding
            
        Returns:
            Dict[str, Any]: Validation result with status and message
        """
        # Validate EBIT
        if ebit is not None and not isinstance(ebit, (int, float)):
            return {"status": "error", "message": "EBIT must be a numeric value"}
            
        # Validate interest
        if interest is not None:
            if not isinstance(interest, (int, float)):
                return {"status": "error", "message": "Interest must be a numeric value"}
                
        # Validate tax rate
        if tax_rate is not None:
            if not isinstance(tax_rate, (int, float)):
                return {"status": "error", "message": "Tax rate must be a numeric value"}
            if tax_rate < 0 or tax_rate > 1:
                return {"status": "error", "message": "Tax rate must be between 0 and 1"}
                
        # Validate shares
        if shares is not None:
            if not isinstance(shares, (int, float)):
                return {"status": "error", "message": "Shares must be a numeric value"}
            if shares <= 0:
                return {"status": "error", "message": "Shares outstanding must be greater than zero"}
                
        # All validations passed
        return {"status": "success", "message": "All inputs validated"}
        
    def calculate_eps(self, ebit, interest, tax_rate, shares):
        """
        Calculate Earnings Per Share (EPS) using EBIT
        Formula: EPS = (EBIT - Interest) * (1 - Tax Rate) / Shares Outstanding
        
        Args:
            ebit (float): Earnings Before Interest and Taxes
            interest (float): Interest payments
            tax_rate (float): Tax rate (as decimal)
            shares (float): Number of shares outstanding
            
        Returns:
            float: Earnings Per Share
        """
        # Validate inputs
        validation = self.validate_ebit_inputs(ebit, interest, tax_rate, shares)
        if validation["status"] == "error":
            raise ValueError(validation["message"])
            
        try:
            return ((ebit - interest) * (1 - tax_rate)) / shares
        except ZeroDivisionError:
            raise ValueError("Shares outstanding cannot be zero")
        except Exception as e:
            logger.error(f"Error calculating EPS: {str(e)}")
            raise ValueError(f"EPS calculation error: {str(e)}")
    
    def handle_isin_mismatch(self, isin: str, claimed_issuer: str) -> Dict[str, Any]:
        """
        Handle ISIN-Issuer mismatch
        
        Args:
            isin (str): ISIN code
            claimed_issuer (str): Claimed issuer name
            
        Returns:
            Dict[str, Any]: Mismatch information
        """
        # Get actual issuer
        actual_issuer = None
        if self.bonds_df is not None and not self.bonds_df.empty:
            bond_row = self.bonds_df[self.bonds_df['isin'] == isin]
            if not bond_row.empty:
                actual_issuer = bond_row['issuer_name'].iloc[0]
        
        if not actual_issuer:
            return {
                "error": "UNKNOWN_ISIN",
                "message": f"ISIN {isin} not found in database"
            }
        
        # If claimed issuer matches actual issuer, no mismatch
        if claimed_issuer.lower() == actual_issuer.lower():
            return {
                "error": None,
                "actual_issuer": actual_issuer
            }
        
        # Find similar issuers using fuzzy matching
        similar_issuers = []
        if self.bonds_df is not None and not self.bonds_df.empty:
            unique_issuers = self.bonds_df['issuer_name'].unique()
            from rapidfuzz import process
            matches = process.extract(claimed_issuer, unique_issuers, scorer=fuzz.token_sort_ratio)
            similar_issuers = [m[0] for m in matches if m[1] > 85][:3]  # Top 3 matches with >85% similarity
        
        return {
            "error": "MISMATCH",
            "actual_issuer": actual_issuer,
            "similar_issuers": similar_issuers
        } 