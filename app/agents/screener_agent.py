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
            explainer = shap.TreeExplainer(self.financial_health_model)
            shap_values = explainer.shap_values(X)
            
            # Map predictions back to bonds
            for i, bond in enumerate(bonds):
                issuer_id = bond.get('issuer_id')
                if issuer_id in features.index:
                    idx = features.index.get_loc(issuer_id)
                    bond['financial_health'] = {
                        'category': int(predictions[idx]),
                        'probability': float(max(probabilities[idx])),
                        'explanation': self._generate_shap_explanation(shap_values[int(predictions[idx])][idx], X.iloc[idx], self.feature_names)
                    }
            
            return bonds
        except Exception as e:
            logger.error(f"Error assessing financial health: {str(e)}")
            return bonds
            
    def _generate_shap_explanation(self, shap_values, feature_values, feature_names):
        """
        Generate human-readable explanations from SHAP values
        
        Args:
            shap_values: SHAP values for a single prediction
            feature_values: Feature values for the prediction
            feature_names: Names of the features
            
        Returns:
            Dict[str, Any]: Human-readable explanation
        """
        # Pair feature names with their SHAP values
        feature_impacts = list(zip(feature_names, shap_values))
        
        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take top 5 most impactful features
        top_features = feature_impacts[:5]
        
        # Generate explanations
        explanations = []
        for feature_name, impact in top_features:
            if impact > 0:
                direction = "positively"
            else:
                direction = "negatively"
                
            value = feature_values[feature_name]
            explanations.append({
                "feature": feature_name,
                "impact": float(abs(impact)),
                "direction": direction,
                "value": float(value),
                "explanation": f"{feature_name.replace('_', ' ').title()} ({value:.2f}) impacts {direction} with strength {abs(impact):.2f}"
            })
            
        return explanations
    
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
    
    # Check if we're analyzing a portfolio
    if "portfolio" in query.lower() and "analyze" in query.lower():
        # Extract ISINs from query
        isins = re.findall(r'INE[A-Z0-9]{10}', query)
        
        # If ISINs found, analyze portfolio
        if isins:
            result = agent.analyze_portfolio(isins)
        else:
            # Process as a regular query
            result = agent.process_query(query)
    else:
        # Process as a regular query
        result = agent.process_query(query)
    
    # Update state
    state["agent_results"] = result
    
    return state 