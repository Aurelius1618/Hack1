import unittest
import os
import sys
import json
import pandas as pd
from datetime import datetime, date, timedelta

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required modules
from app.core.workflow import workflow
from app.data.data_processor import FinancialDataProcessor
from app.agents.directory_agent import DirectoryAgent
from app.agents.finder_agent import FinderAgent
from app.agents.cashflow_agent import CashFlowAgent
from app.agents.screener_agent import ScreenerAgent
from app.agents.yield_calculator_agent import YieldCalculatorAgent
from app.utils.document_processor import DocumentProcessor
from app.core.pinecone_config import initialize_pinecone, hybrid_search, NAMESPACES
from app.utils.model_config import get_mistral_model

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
        
        # Initialize Lamini model
        self.llm = get_mistral_model()
        
        # Skip tests that require Lamini if API key is not available
        self.skip_lamini_tests = os.getenv("LAMINI_API_KEY") is None or os.getenv("FINETUNED_MODEL_ID") is None
        if self.skip_lamini_tests:
            print("WARNING: LAMINI_API_KEY or FINETUNED_MODEL_ID not found. Skipping Lamini tests.")
        
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

    def test_lamini_model(self):
        """
        Test the Lamini model
        """
        if self.skip_lamini_tests:
            self.skipTest("Skipping Lamini test due to missing API key or model ID")
            
        # Test that the model can be initialized
        self.assertIsNotNone(self.llm, "Lamini model should be initialized")
        
        # Test a simple query
        try:
            response = self.llm.generate("What is a bond?")
            self.assertIsInstance(response, str, "Response should be a string")
            self.assertGreater(len(response), 0, "Response should not be empty")
        except Exception as e:
            self.fail(f"Lamini API call failed: {str(e)}")
    
    def test_isin_lookup(self):
        """
        Test ISIN lookup with Lamini
        """
        if self.skip_lamini_tests:
            self.skipTest("Skipping Lamini test due to missing API key or model ID")
            
        # Test ISIN lookup
        isin = "INE123456789"  # Example ISIN
        
        try:
            query = f"Show me details for ISIN {isin}"
            response = self.llm.generate(query)
            
            self.assertIsInstance(response, str, "Response should be a string")
            self.assertGreater(len(response), 0, "Response should not be empty")
        except Exception as e:
            self.fail(f"Lamini API call failed: {str(e)}")

class TestDirectoryAgent(unittest.TestCase):
    """Test the DirectoryAgent implementation"""
    
    def setUp(self):
        """Set up the test environment"""
        self.agent = DirectoryAgent()
        
    def test_chunking_rules(self):
        """Test that chunking rules are properly defined"""
        self.assertIn("legal_docs", self.agent.chunking_rules)
        self.assertIn("financial_terms", self.agent.chunking_rules)
        self.assertIn("isin_details", self.agent.chunking_rules)
        
        # Check that sizes and overlaps are reasonable
        self.assertEqual(self.agent.chunking_rules["legal_docs"]["size"], 1024)
        self.assertEqual(self.agent.chunking_rules["legal_docs"]["overlap"], 256)
        
    def test_chunk_document(self):
        """Test document chunking functionality"""
        # Create a test document
        test_doc = "A" * 2000
        
        # Chunk the document
        chunks = self.agent.chunk_document(test_doc, "legal_docs")
        
        # Check that chunks were created
        self.assertGreater(len(chunks), 1)
        
    def test_get_security_details(self):
        """Test security details retrieval"""
        # This is a mock test since we don't have actual data
        result = self.agent.get_security_details("INE123456789")
        
        # Check that the result has the expected structure
        self.assertIn("status", result)
        if result["status"] == "success":
            self.assertIn("data", result)
            self.assertIn("collateral", result["data"])
            self.assertIn("coverage_ratio", result["data"])
            self.assertIn("liquidation_priority", result["data"])

class TestFinderAgent(unittest.TestCase):
    """Test the FinderAgent implementation"""
    
    def setUp(self):
        """Set up the test environment"""
        self.agent = FinderAgent()
        
    def test_platform_weights(self):
        """Test that platform weights are properly defined"""
        self.assertIn("SMEST", self.agent.platform_weights)
        self.assertIn("FixedIncome", self.agent.platform_weights)
        
        # Check that weights include multiple factors
        self.assertIn("reliability", self.agent.platform_weights["SMEST"])
        self.assertIn("freshness", self.agent.platform_weights["SMEST"])
        self.assertIn("settlement_speed", self.agent.platform_weights["SMEST"])
        
    def test_normalize_yield(self):
        """Test yield normalization"""
        # Test with known platform
        normalized_yield = self.agent.normalize_yield(10.0, "SMEST")
        self.assertIsInstance(normalized_yield, float)
        
        # Test with unknown platform (should use default)
        normalized_yield = self.agent.normalize_yield(10.0, "UnknownPlatform")
        self.assertIsInstance(normalized_yield, float)
        
    def test_compare_platforms(self):
        """Test platform comparison"""
        result = self.agent.compare_platforms("INE123456789")
        
        # Check that the result has the expected structure
        self.assertIn("status", result)
        if result["status"] == "success":
            self.assertIn("data", result)
            self.assertIn("platforms", result["data"])
            self.assertGreater(len(result["data"]["platforms"]), 0)
            self.assertIn("normalized_yield", result["data"]["platforms"][0])

class TestCashflowAgent(unittest.TestCase):
    """Test the CashflowAgent implementation"""
    
    def setUp(self):
        """Set up the test environment"""
        self.agent = CashFlowAgent()
        
    def test_day_count_conventions(self):
        """Test that day count conventions are properly defined"""
        self.assertIn("30/360", self.agent.day_count_conventions)
        self.assertIn("ACT/ACT", self.agent.day_count_conventions)
        self.assertIn("ACT/360", self.agent.day_count_conventions)
        self.assertIn("ACT/365", self.agent.day_count_conventions)
        
    def test_calculate_30_360_days(self):
        """Test 30/360 day count calculation"""
        start_date = date(2023, 1, 15)
        end_date = date(2023, 7, 15)
        
        days = self.agent._calculate_30_360_days(start_date, end_date)
        self.assertEqual(days, 180)  # 30/360 should give exactly 180 days for 6 months
        
    def test_calculate_accrued_interest(self):
        """Test accrued interest calculation"""
        principal = 1000.0
        rate = 0.05  # 5%
        settlement = date(2023, 7, 15)
        last_coupon = date(2023, 1, 15)
        
        # Test with 30/360 convention
        interest = self.agent.calculate_accrued_interest(principal, rate, settlement, last_coupon, "30/360")
        self.assertAlmostEqual(interest, 25.0)  # 1000 * 0.05 * 180/360 = 25.0
        
        # Test with ACT/365 convention
        interest = self.agent.calculate_accrued_interest(principal, rate, settlement, last_coupon, "ACT/365")
        days = (settlement - last_coupon).days
        expected = principal * rate * days / 365
        self.assertAlmostEqual(interest, expected)

class TestScreenerAgent(unittest.TestCase):
    """Test the ScreenerAgent implementation"""
    
    def setUp(self):
        """Set up the test environment"""
        self.agent = ScreenerAgent()
        
    def test_calculate_altman_z_score(self):
        """Test Altman Z-Score calculation"""
        # Create test company data
        company_data = {
            "working_capital": 100,
            "total_assets": 1000,
            "retained_earnings": 200,
            "ebit": 150,
            "market_value_equity": 500,
            "total_liabilities": 400,
            "sales": 800
        }
        
        z_score = self.agent.calculate_altman_z_score(company_data)
        
        # Check that the Z-Score is calculated
        self.assertIsInstance(z_score, float)
        self.assertGreater(z_score, 0)
        
        # Calculate expected Z-Score manually
        t1 = company_data["working_capital"] / company_data["total_assets"]  # 0.1
        t2 = company_data["retained_earnings"] / company_data["total_assets"]  # 0.2
        t3 = company_data["ebit"] / company_data["total_assets"]  # 0.15
        t4 = company_data["market_value_equity"] / company_data["total_liabilities"]  # 1.25
        t5 = company_data["sales"] / company_data["total_assets"]  # 0.8
        
        expected_z_score = 1.2*t1 + 1.4*t2 + 3.3*t3 + 0.6*t4 + 1.0*t5
        self.assertAlmostEqual(z_score, expected_z_score)

class TestPineconeConfig(unittest.TestCase):
    """Test the Pinecone configuration"""
    
    def test_namespaces(self):
        """Test that namespaces are properly defined"""
        self.assertIn("BOND_METADATA", NAMESPACES)
        self.assertIn("FINANCIAL_METRICS", NAMESPACES)
        self.assertIn("CASHFLOW_PATTERNS", NAMESPACES)
        
        # Check that namespace configurations are complete
        self.assertIn("dimension", NAMESPACES["BOND_METADATA"])
        self.assertIn("metric", NAMESPACES["BOND_METADATA"])
        self.assertIn("content_type", NAMESPACES["BOND_METADATA"])
        
        # Check dimensions
        self.assertEqual(NAMESPACES["BOND_METADATA"]["dimension"], 768)
        self.assertEqual(NAMESPACES["FINANCIAL_METRICS"]["dimension"], 512)
        self.assertEqual(NAMESPACES["CASHFLOW_PATTERNS"]["dimension"], 256)

class TestDocumentProcessor(unittest.TestCase):
    """Test the DocumentProcessor implementation"""
    
    def setUp(self):
        """Set up the test environment"""
        self.processor = DocumentProcessor()
        
    def test_generate_pdf_preview_method_exists(self):
        """Test that the PDF preview method exists"""
        self.assertTrue(hasattr(self.processor, "generate_pdf_preview"))
        self.assertTrue(callable(getattr(self.processor, "generate_pdf_preview")))
        
    def test_get_document_metadata_method_exists(self):
        """Test that the document metadata method exists"""
        self.assertTrue(hasattr(self.processor, "get_document_metadata"))
        self.assertTrue(callable(getattr(self.processor, "get_document_metadata")))

if __name__ == '__main__':
    unittest.main() 
