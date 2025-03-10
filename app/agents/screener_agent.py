import logging
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from app.data.data_processor import FinancialDataProcessor
from app.core.pinecone_config import hybrid_search, NAMESPACES
import xgboost as xgb
import shap
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScreenerAgent:
    """
    Agent for handling bond screening and financial health analysis
    """
    def __init__(self):
        """
        Initialize the screener agent
        """
        self.data_processor = FinancialDataProcessor()
        self.financial_health_model = None
        self.feature_names = None
        
    def initialize(self):
        """
        Initialize the agent by loading data and models
        """
        self.data_processor.load_data()
        self._initialize_financial_health_model()
        
    def _initialize_financial_health_model(self):
        """
        Initialize the financial health model
        """
        try:
            # In a production environment, we would load a pre-trained model
            # For now, we'll train a simple model on the fly
            if self.data_processor.company_df is not None:
                # Prepare features
                features = self._engineer_financial_features(self.data_processor.company_df)
                
                if features is not None and not features.empty:
                    # Create a simple target variable based on financial metrics
                    # In production, this would be based on historical default data
                    features['financial_health'] = pd.qcut(
                        features['interest_coverage_ratio'].fillna(0) + 
                        features['debt_service_coverage_ratio'].fillna(0) + 
                        features['current_ratio'].fillna(0), 
                        q=3, labels=['Poor', 'Average', 'Good']
                    )
                    
                    # Prepare training data
                    X = features.drop(['financial_health'], axis=1)
                    y = features['financial_health']
                    
                    # Store feature names
                    self.feature_names = X.columns.tolist()
                    
                    # Train XGBoost model
                    params = {
                        'objective': 'multi:softprob',
                        'n_estimators': 200,
                        'max_depth': 7,
                        'learning_rate': 0.1,
                        'num_class': 3
                    }
                    
                    self.financial_health_model = xgb.XGBClassifier(**params)
                    self.financial_health_model.fit(X, y)
                    
                    logger.info("Financial health model initialized successfully")
                else:
                    logger.warning("Could not engineer financial features")
            else:
                logger.warning("Company data not loaded")
        except Exception as e:
            logger.error(f"Error initializing financial health model: {str(e)}")
    
    def _engineer_financial_features(self, company_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Engineer financial features for the model
        
        Args:
            company_df (pd.DataFrame): Company data
            
        Returns:
            Optional[pd.DataFrame]: Engineered features
        """
        try:
            # Create a copy to avoid modifying the original
            df = company_df.copy()
            
            # Calculate financial ratios
            # 1. Interest Coverage Ratio = EBIT / Interest Expense
            if 'ebit' in df.columns and 'interest_expense' in df.columns:
                df['interest_coverage_ratio'] = df['ebit'] / df['interest_expense'].replace(0, 0.001)
            else:
                df['interest_coverage_ratio'] = 1.0  # Default value
            
            # 2. Debt Service Coverage Ratio = (Net Income + Depreciation) / Total Debt
            if all(col in df.columns for col in ['net_income', 'depreciation', 'total_debt']):
                df['debt_service_coverage_ratio'] = (df['net_income'] + df['depreciation']) / df['total_debt'].replace(0, 0.001)
            else:
                df['debt_service_coverage_ratio'] = 1.0  # Default value
            
            # 3. Current Ratio = Current Assets / Current Liabilities
            if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
                df['current_ratio'] = df['current_assets'] / df['current_liabilities'].replace(0, 0.001)
            else:
                df['current_ratio'] = 1.0  # Default value
            
            # 4. Quick Ratio = (Current Assets - Inventory) / Current Liabilities
            if all(col in df.columns for col in ['current_assets', 'inventory', 'current_liabilities']):
                df['quick_ratio'] = (df['current_assets'] - df['inventory']) / df['current_liabilities'].replace(0, 0.001)
            else:
                df['quick_ratio'] = 1.0  # Default value
            
            # 5. Debt-to-Equity Ratio = Total Debt / Total Equity
            if 'total_debt' in df.columns and 'total_equity' in df.columns:
                df['debt_to_equity_ratio'] = df['total_debt'] / df['total_equity'].replace(0, 0.001)
            else:
                df['debt_to_equity_ratio'] = 1.0  # Default value
            
            # 6. Return on Assets (ROA) = Net Income / Total Assets
            if 'net_income' in df.columns and 'total_assets' in df.columns:
                df['return_on_assets'] = df['net_income'] / df['total_assets'].replace(0, 0.001)
            else:
                df['return_on_assets'] = 0.05  # Default value
            
            # 7. Return on Equity (ROE) = Net Income / Total Equity
            if 'net_income' in df.columns and 'total_equity' in df.columns:
                df['return_on_equity'] = df['net_income'] / df['total_equity'].replace(0, 0.001)
            else:
                df['return_on_equity'] = 0.1  # Default value
            
            # 8. Profit Margin = Net Income / Revenue
            if 'net_income' in df.columns and 'revenue' in df.columns:
                df['profit_margin'] = df['net_income'] / df['revenue'].replace(0, 0.001)
            else:
                df['profit_margin'] = 0.05  # Default value
            
            # Select only the calculated ratios
            features = df[[
                'interest_coverage_ratio',
                'debt_service_coverage_ratio',
                'current_ratio',
                'quick_ratio',
                'debt_to_equity_ratio',
                'return_on_assets',
                'return_on_equity',
                'profit_margin'
            ]]
            
            # Handle infinite values
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values with median
            features = features.fillna(features.median())
            
            return features
        except Exception as e:
            logger.error(f"Error engineering financial features: {str(e)}")
            return None
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query related to bond screening
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Any]: Response with screened bonds
        """
        # Extract parameters from query
        params = self._extract_parameters(query)
        
        # Screen bonds based on parameters
        screened_bonds = self._screen_bonds(params)
        
        if screened_bonds:
            # Assess financial health
            assessed_bonds = self._assess_financial_health(screened_bonds)
            
            return {
                "status": "success",
                "message": f"Found {len(assessed_bonds)} bonds matching your criteria",
                "data": assessed_bonds
            }
        else:
            return {
                "status": "error",
                "message": "No bonds found matching your criteria",
                "data": None
            }
    
    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract screening parameters from query
        
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
        
        # Extract maturity range
        min_maturity_match = re.search(r'(?:min|minimum|at least)\s+(\d+)\s*(?:year|yr)(?:s)?\s+maturity', query, re.IGNORECASE)
        if min_maturity_match:
            params['min_maturity'] = int(min_maturity_match.group(1))
        
        max_maturity_match = re.search(r'(?:max|maximum|at most)\s+(\d+)\s*(?:year|yr)(?:s)?\s+maturity', query, re.IGNORECASE)
        if max_maturity_match:
            params['max_maturity'] = int(max_maturity_match.group(1))
        
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
        
        # Extract financial health
        if 'good financial health' in query.lower() or 'strong financial' in query.lower():
            params['financial_health'] = 'Good'
        elif 'poor financial health' in query.lower() or 'weak financial' in query.lower():
            params['financial_health'] = 'Poor'
        
        # Extract limit
        limit_match = re.search(r'(?:top|limit)\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            params['limit'] = int(limit_match.group(1))
        else:
            params['limit'] = 10  # Default limit
        
        return params
    
    def _screen_bonds(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Screen bonds based on parameters
        
        Args:
            params (Dict[str, Any]): Screening parameters
            
        Returns:
            List[Dict[str, Any]]: Screened bonds
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
        
        if 'min_maturity' in params:
            # Calculate years to maturity
            current_date = datetime.now()
            filtered_df['years_to_maturity'] = (filtered_df['maturity_date'] - current_date).dt.days / 365.25
            filtered_df = filtered_df[filtered_df['years_to_maturity'] >= params['min_maturity']]
        
        if 'max_maturity' in params:
            if 'years_to_maturity' not in filtered_df.columns:
                current_date = datetime.now()
                filtered_df['years_to_maturity'] = (filtered_df['maturity_date'] - current_date).dt.days / 365.25
            filtered_df = filtered_df[filtered_df['years_to_maturity'] <= params['max_maturity']]
        
        if 'rating' in params:
            filtered_df = filtered_df[filtered_df['rating'] == params['rating']]
        
        if 'sector' in params:
            filtered_df = filtered_df[filtered_df['sector'].str.lower() == params['sector'].lower()]
        
        if 'issuer_type' in params:
            filtered_df = filtered_df[filtered_df['issuer_type'].str.lower() == params['issuer_type'].lower()]
        
        # Convert to list of dictionaries
        result = filtered_df.to_dict('records')
        
        # Apply limit
        limit = params.get('limit', 10)
        return result[:limit]
    
    def _assess_financial_health(self, bonds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assess the financial health of bond issuers
        
        Args:
            bonds (List[Dict[str, Any]]): List of bonds
            
        Returns:
            List[Dict[str, Any]]: Bonds with financial health assessment
        """
        if not bonds:
            return []
        
        if self.financial_health_model is None:
            logger.warning("Financial health model not initialized")
            return bonds
        
        # Extract company identifiers from bonds
        company_ids = [bond.get('issuer_id') for bond in bonds if bond.get('issuer_id')]
        
        if not company_ids:
            return bonds
        
        # Get financial data for these companies
        company_data = self.data_processor.company_df[
            self.data_processor.company_df['company_id'].isin(company_ids)
        ]
        
        if company_data.empty:
            return bonds
        
        # Prepare features
        features = self._engineer_financial_features(company_data)
        
        if features is None or features.empty:
            return bonds
        
        # Make predictions
        try:
            # Select only the features used in the model
            X = features[self.feature_names].fillna(0)
            
            # Predict financial health
            predictions = self.financial_health_model.predict(X)
            probabilities = self.financial_health_model.predict_proba(X)
            
            # Generate SHAP explanations
            shap_explanations = self._generate_shap_explanations(self.financial_health_model, X)
            
            # Map predictions back to bonds
            for i, bond in enumerate(bonds):
                issuer_id = bond.get('issuer_id')
                if issuer_id in features.index:
                    idx = features.index.get_loc(issuer_id)
                    bond['financial_health'] = {
                        'category': int(predictions[idx]),
                        'probability': float(max(probabilities[idx])),
                        'explanation': shap_explanations["explanations"][int(predictions[idx])]
                    }
            
            return bonds
        except Exception as e:
            logger.error(f"Error assessing financial health: {str(e)}")
            return bonds
            
    def _generate_shap_explanations(self, model, X_sample):
        """
        Generate SHAP explanations for model predictions
        
        Args:
            model: Trained model
            X_sample: Sample data for explanation
            
        Returns:
            Dict: SHAP explanations
        """
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Get feature importance
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                importance = np.abs(shap_values[:, i]).mean()
                feature_importance[feature] = float(importance)
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Generate explanations
            explanations = []
            for feature, importance in sorted_features[:5]:  # Top 5 features
                explanations.append({
                    "feature": feature,
                    "importance": importance,
                    "description": self._get_feature_description(feature)
                })
            
            return {
                "explanations": explanations,
                "summary": self._generate_explanation_summary(explanations)
            }
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}")
            return {
                "explanations": [],
                "summary": "Unable to generate explanations"
            }

    def _get_feature_description(self, feature):
        """
        Get description for a financial feature
        
        Args:
            feature: Feature name
            
        Returns:
            str: Feature description
        """
        descriptions = {
            "interest_coverage_ratio": "Ability to pay interest on outstanding debt",
            "debt_to_equity_ratio": "Proportion of equity and debt used to finance assets",
            "current_ratio": "Ability to pay short-term obligations",
            "quick_ratio": "Ability to pay short-term obligations with liquid assets",
            "return_on_assets": "How efficiently assets are being used",
            "return_on_equity": "How efficiently equity is being used",
            "profit_margin": "Percentage of revenue retained after all expenses",
            "debt_service_coverage_ratio": "Ability to service debt with operating income"
        }
        
        return descriptions.get(feature, "Financial health indicator")

    def _generate_explanation_summary(self, explanations):
        """
        Generate a summary of the explanations
        
        Args:
            explanations: List of explanations
            
        Returns:
            str: Summary
        """
        if not explanations:
            return "No explanations available"
        
        top_feature = explanations[0]["feature"]
        top_description = explanations[0]["description"]
        
        return f"The most important factor in this assessment is {top_feature} ({top_description}), followed by {explanations[1]['feature']} and {explanations[2]['feature']}."
    
    def analyze_portfolio(self, isins: List[str]) -> Dict[str, Any]:
        """
        Analyze a portfolio of bonds
        
        Args:
            isins (List[str]): List of ISINs in the portfolio
            
        Returns:
            Dict[str, Any]: Portfolio analysis
        """
        if self.data_processor.bonds_df is None:
            return {
                "status": "error",
                "message": "Bond data not loaded",
                "data": None
            }
        
        # Get bonds in portfolio
        portfolio_bonds = []
        for isin in isins:
            bond_data = self.data_processor.bonds_df[self.data_processor.bonds_df['isin'] == isin]
            if not bond_data.empty:
                portfolio_bonds.append(bond_data.iloc[0].to_dict())
        
        if not portfolio_bonds:
            return {
                "status": "error",
                "message": "No valid bonds found in portfolio",
                "data": None
            }
        
        # Assess financial health
        assessed_bonds = self._assess_financial_health(portfolio_bonds)
        
        # Calculate portfolio metrics
        total_value = sum(bond.get('face_value', 0) for bond in assessed_bonds)
        weighted_yield = sum(bond.get('yield', 0) * bond.get('face_value', 0) for bond in assessed_bonds) / total_value if total_value > 0 else 0
        
        # Count bonds by financial health
        health_counts = {
            "Good": 0,
            "Average": 0,
            "Poor": 0,
            "Unknown": 0
        }
        
        for bond in assessed_bonds:
            health = bond.get('financial_health', {}).get('rating', 'Unknown')
            health_counts[health] = health_counts.get(health, 0) + 1
        
        # Calculate diversification score
        sectors = set(bond.get('sector', 'Unknown') for bond in assessed_bonds)
        issuers = set(bond.get('issuer', 'Unknown') for bond in assessed_bonds)
        
        diversification_score = min(1.0, (len(sectors) / max(1, len(assessed_bonds))) * 0.5 + (len(issuers) / max(1, len(assessed_bonds))) * 0.5)
        
        return {
            "status": "success",
            "message": f"Analyzed portfolio with {len(assessed_bonds)} bonds",
            "data": {
                "bonds": assessed_bonds,
                "summary": {
                    "total_value": total_value,
                    "weighted_yield": weighted_yield,
                    "financial_health_distribution": health_counts,
                    "diversification_score": diversification_score,
                    "total_bonds": len(assessed_bonds)
                }
            }
        }

    def analyze_ebit_scenarios(self, ebit: float, interest_options: List[float], tax_rate: float, shares: int) -> Dict[str, Any]:
        """
        Analyze different capital structure scenarios using EBIT
        
        Args:
            ebit (float): Earnings Before Interest and Taxes
            interest_options (List[float]): Different interest payment scenarios
            tax_rate (float): Tax rate as decimal
            shares (int): Number of shares outstanding
            
        Returns:
            Dict[str, Any]: Analysis results of different scenarios
        """
        try:
            # Validate inputs
            validation = self.data_processor.validate_ebit_inputs(ebit, None, tax_rate, shares)
            if validation["status"] == "error":
                return {
                    "status": "error",
                    "message": validation["message"],
                    "data": None
                }
                
            # Calculate EPS for each interest scenario
            scenario_results = []
            for interest in interest_options:
                # Validate interest
                interest_validation = self.data_processor.validate_ebit_inputs(None, interest, None, None)
                if interest_validation["status"] == "error":
                    continue
                
                try:
                    eps = self.data_processor.calculate_eps(ebit, interest, tax_rate, shares)
                    scenario_results.append({
                        "interest_expense": interest,
                        "eps": eps,
                        "interest_coverage_ratio": ebit / interest if interest > 0 else float('inf'),
                        "debt_ratio": interest / ebit if ebit > 0 else float('inf')
                    })
                except ValueError as ve:
                    logger.warning(f"Skipping scenario due to error: {str(ve)}")
                    continue
            
            if not scenario_results:
                return {
                    "status": "error",
                    "message": "No valid scenarios could be calculated",
                    "data": None
                }
                
            # Sort scenarios by EPS
            scenario_results.sort(key=lambda x: x["eps"], reverse=True)
            
            # Calculate indifference point
            if len(scenario_results) > 1:
                # Find the interest level where EPS between two adjacent scenarios are closest
                min_diff = float('inf')
                indifference_point = None
                
                for i in range(len(scenario_results) - 1):
                    diff = abs(scenario_results[i]["eps"] - scenario_results[i+1]["eps"])
                    if diff < min_diff:
                        min_diff = diff
                        indifference_point = {
                            "interest_level": (scenario_results[i]["interest_expense"] + scenario_results[i+1]["interest_expense"]) / 2,
                            "eps_diff": diff
                        }
            else:
                indifference_point = None
            
            return {
                "status": "success",
                "message": f"Analyzed {len(scenario_results)} EBIT scenarios",
                "data": {
                    "scenarios": scenario_results,
                    "indifference_point": indifference_point,
                    "inputs": {
                        "ebit": ebit,
                        "tax_rate": tax_rate * 100,  # Convert to percentage for display
                        "shares": shares
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing EBIT scenarios: {str(e)}")
            return {
                "status": "error",
                "message": f"Error analyzing EBIT scenarios: {str(e)}",
                "data": None
            }

# Function to handle screener agent queries in the LangGraph workflow
def handle_screener(state):
    """
    Handler for screener agent in the LangGraph workflow
    
    Args:
        state: The current state
        
    Returns:
        Updated state
    """
    query = state["query"]
    
    # Initialize agent
    agent = ScreenerAgent()
    agent.initialize()
    
    # Check if this is an EBIT-related query
    ebit_match = re.search(r'ebit\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?k?m?b?)', query, re.IGNORECASE)
    
    if ebit_match:
        # Extract EBIT parameters
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
        
        # Extract EBIT value
        ebit = convert_value(ebit_match)
        
        # Extract other parameters
        interest_matches = re.findall(r'interest\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?k?m?b?)', query, re.IGNORECASE)
        interest_options = [convert_value(re.match(r'(.*)', interest)) for interest in interest_matches] if interest_matches else [100000, 150000, 200000]  # Default options
        
        tax_match = re.search(r'tax\s*(?:rate|percentage)?\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?)\s*%?', query, re.IGNORECASE)
        tax_rate = float(tax_match.group(1)) / 100 if tax_match else 0.4  # Default 40%
        
        shares_match = re.search(r'shares\s*(?:outstanding|count|number)?\s*(?:of|is|at)?\s*(\d+(?:\.\d+)?k?m?b?)', query, re.IGNORECASE)
        shares = convert_value(shares_match) if shares_match else 50000  # Default 50,000 shares
        
        # Analyze EBIT scenarios
        result = agent.analyze_ebit_scenarios(ebit, interest_options, tax_rate, shares)
    elif "portfolio" in query.lower() and "analyze" in query.lower():
        # Extract ISINs from query
        isins = re.findall(r'INE[A-Z0-9]{10}', query)
        
        # If ISINs found, analyze portfolio
        if isins:
            result = agent.analyze_portfolio(isins)
        else:
            # Process as a regular query
            result = agent.process_query(query)
    else:
        # Process normal screening query
        result = agent.process_query(query)
    
    # Update state
    state["agent_results"] = result
    
    return state 