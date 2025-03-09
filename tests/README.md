# Tap Bonds AI Layer - Test Suite

This directory contains the test suite for the Tap Bonds AI Layer.

## Running the Tests

To run all tests, use the following command from the project root:

```bash
python -m tests.run_tests
```

This will run all tests and generate a report with the results.

## Test Coverage

The test suite covers the following areas:

1. **ISIN Validation**: Tests the validation of ISIN codes
2. **Yield Calculation**: Tests the accuracy of yield calculations
3. **Financial Ratio Sanity**: Tests the sanity of financial ratio calculations
4. **Cashflow Schedule**: Tests the cashflow schedule calculation
5. **Workflow Query Routing**: Tests the routing of queries to the appropriate agent
6. **API Response Format**: Tests the format of API responses
7. **Performance Metrics**: Tests the performance metrics of the system

## Adding New Tests

To add a new test, create a new file in the `tests` directory with the prefix `test_`. For example, `test_new_feature.py`. The test runner will automatically discover and run all tests in files with this prefix.

## Test Requirements

The test suite requires the following dependencies:

- unittest
- pandas
- numpy

These dependencies are already included in the project's `requirements.txt` file.

## Continuous Integration

The test suite is designed to be run as part of a continuous integration pipeline. The test runner will exit with a non-zero status code if any tests fail, which can be used to trigger a build failure in a CI/CD pipeline. 