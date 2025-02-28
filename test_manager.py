"""
test_manager.py - Entry point for running all tests using unittest discovery.
This module discovers and runs all test cases in the tests folder, providing a centralized command
for executing the entire test suite.
"""

import unittest

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover('tests')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    exit(not result.wasSuccessful())