#!/usr/bin/env python
import unittest
import os
import sys
import time
import logging
from datetime import datetime

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_tests():
    """Run all tests and generate a report"""
    start_time = time.time()
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(os.path.abspath(__file__)), pattern="test_*.py")
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Log the results
    logger.info(f"Test run completed in {duration:.2f} seconds")
    logger.info(f"Tests run: {test_result.testsRun}")
    logger.info(f"Errors: {len(test_result.errors)}")
    logger.info(f"Failures: {len(test_result.failures)}")
    logger.info(f"Skipped: {len(test_result.skipped)}")
    
    # Calculate success rate
    success_rate = (test_result.testsRun - len(test_result.errors) - len(test_result.failures)) / test_result.testsRun * 100
    logger.info(f"Success rate: {success_rate:.2f}%")
    
    # Log any errors or failures
    if test_result.errors:
        logger.error("Errors:")
        for test, error in test_result.errors:
            logger.error(f"{test}: {error}")
    
    if test_result.failures:
        logger.error("Failures:")
        for test, failure in test_result.failures:
            logger.error(f"{test}: {failure}")
    
    return test_result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 