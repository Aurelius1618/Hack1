import unittest
import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required modules
from app.core.workflow import workflow
from app.data.data_processor import FinancialDataProcessor
from app.agents.directory_agent import DirectoryAgent
from app.agents.finder_agent import FinderAgent
from app.agents.cashflow_agent import CashFlowAgent
from app.agents.screener_agent import ScreenerAgent
from app.utils.document_processor import DocumentProcessor

class TestValidation(unittest.TestCase):
    """
    Test cases for validating the Tap Bonds AI Layer implementation
    """
    
    def setUp(self):
        """
        Set up the test environment
        """
        # Initialize data processor
        self.data_processor = FinancialDataProcessor()
        self.data_processor.load_data()
        
        # Initialize agents
        self.directory_agent = DirectoryAgent()
        self.directory_agent.initialize()
        
        self.finder_agent = FinderAgent()
        self.finder_agent.initialize()
        
        self.cashflow_agent = CashFlowAgent()
        self.cashflow_agent.initialize()
        
        self.screener_agent = ScreenerAgent()
        self.screener_agent.initialize()
        
        # Initialize document processor
        self.document_processor = DocumentProcessor()
        
        # Test cases
        self.test_cases = [
            {
                "input": "Compare 5-year AA+ bonds from financial sector",
                "expected": ["SMEST", "FixedIncome"],
                "min_results": 3
            },
            {
                "input": "Cash flow schedule for INE08XP07258",
                "expected_dates": ["2025-03-22", "2025-09-07"],
                "tolerance": 2
            },
            {
                "input": "Find bonds with yield above 5% in technology sector",
                "expected_field": "yield",
                "min_value": 5.0,
                "min_results": 2
            },
            {
                "input": "What is the financial health of company issuing INE08XP07258?",
                "expected_field": "financial_health",
                "should_contain": True
            }
        ]
    
    def test_workflow(self):
        """
        Test the LangGraph workflow
        """
        for test_case in self.test_cases:
            # Invoke the workflow
            result = workflow.invoke({
                "query": test_case["input"],
                "isin": None,
                "results": {},
                "routing_data": {},
                "agent_results": {}
            })
            
            # Validate the result
            self.assertIsNotNone(result)
            self.assertIn("agent_results", result)
            
            # Specific validations based on test case
            if "expected" in test_case:
                for expected in test_case["expected"]:
                    self.assertIn(expected, str(result))
            
            if "min_results" in test_case:
                if "data" in result["agent_results"]:
                    self.assertGreaterEqual(len(result["agent_results"]["data"]), test_case["min_results"])
            
            if "expected_dates" in test_case:
                if "data" in result["agent_results"] and "cashflow" in result["agent_results"]["data"]:
                    cashflow = result["agent_results"]["data"]["cashflow"]
                    dates = [cf["payment_date"] for cf in cashflow]
                    for expected_date in test_case["expected_dates"]:
                        self.assertTrue(any(expected_date in date for date in dates))
            
            if "expected_field" in test_case:
                if "data" in result["agent_results"]:
                    if isinstance(result["agent_results"]["data"], list):
                        for item in result["agent_results"]["data"]:
                            if "min_value" in test_case:
                                self.assertGreaterEqual(float(item[test_case["expected_field"]]), test_case["min_value"])
                    elif "should_contain" in test_case and test_case["should_contain"]:
                        self.assertIn(test_case["expected_field"], result["agent_results"]["data"])
    
    def test_financial_ratios(self):
        """
        Test the financial ratio calculations
        """
        # Check if financial ratios are calculated
        self.assertIsNotNone(self.data_processor.company_df)
        
        # Check if the critical ratios are present
        critical_ratios = [
            'interest_coverage_ratio',
            'debt_service_coverage_ratio',
            'current_ratio',
            'debt_to_equity',
            'return_on_assets',
            'return_on_equity',
            'profit_margin',
            'asset_turnover'
        ]
        
        for ratio in critical_ratios:
            self.assertIn(ratio, self.data_processor.company_df.columns)
    
    def test_shap_explanations(self):
        """
        Test the SHAP explanations
        """
        # Create a sample bond list
        bonds = [
            {
                "isin": "INE08XP07258",
                "issuer": "Sample Company",
                "issuer_id": "COMP001",
                "yield": 5.5,
                "rating": "AA+"
            }
        ]
        
        # Assess financial health
        assessed_bonds = self.screener_agent._assess_financial_health(bonds)
        
        # Check if financial health assessment is present
        self.assertGreaterEqual(len(assessed_bonds), 1)
        
        # Check if explanation is present
        if 'financial_health' in assessed_bonds[0]:
            self.assertIn('explanation', assessed_bonds[0]['financial_health'])
            
            # Check if explanation has the required fields
            explanation = assessed_bonds[0]['financial_health']['explanation']
            if explanation:
                self.assertGreaterEqual(len(explanation), 1)
                self.assertIn('feature', explanation[0])
                self.assertIn('impact', explanation[0])
                self.assertIn('direction', explanation[0])
                self.assertIn('value', explanation[0])
    
    def test_document_processor(self):
        """
        Test the document processor
        """
        # This test would require a sample PDF file
        # For now, we'll just test the entity extraction from text
        
        sample_text = """
        ISIN: INE08XP07258
        Issuer: Sample Company Ltd.
        Maturity Date: 22/03/2025
        Coupon: 5.5%
        Face Value: Rs. 1,000
        Yield: 6.2%
        Payment Frequency: Semi-Annual
        Rating: AA+ (CRISIL)
        """
        
        entities = self.document_processor.extract_entities_from_text(sample_text)
        
        # Check if entities are extracted correctly
        self.assertEqual(entities['isin'], 'INE08XP07258')
        self.assertEqual(entities['issuer'], 'Sample Company Ltd.')
        self.assertEqual(entities['coupon_rate'], 5.5)
        self.assertEqual(entities['face_value'], 1000.0)
        self.assertEqual(entities['yield'], 6.2)
        self.assertEqual(entities['payment_frequency'], 'Semi-Annual')
        self.assertEqual(entities['rating'], 'AA+ (CRISIL)')

if __name__ == '__main__':
    unittest.main() 
