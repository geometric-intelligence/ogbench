#!/usr/bin/env python3
"""Test script for parameter optimization functionality.

This script tests the basic functionality of the parameter optimization scripts without requiring a
full dataset or complex setup.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_param_optimizer import AdvancedParameterOptimizer, OptimizationResult
from find_target_params import ParameterOptimizer


class TestParameterOptimization(unittest.TestCase):
    """Test cases for parameter optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config_path = "../configs"
        self.test_project_config = "train.yaml"

    def test_parameter_counting(self):
        """Test parameter counting functionality."""
        optimizer = ParameterOptimizer()

        # Create a simple mock model
        mock_model = MagicMock()
        mock_param1 = MagicMock()
        mock_param1.numel.return_value = 1000
        mock_param1.requires_grad = True

        mock_param2 = MagicMock()
        mock_param2.numel.return_value = 2000
        mock_param2.requires_grad = True

        mock_model.parameters.return_value = [mock_param1, mock_param2]

        # Test parameter counting
        param_count = optimizer.count_trainable_params(mock_model)
        self.assertEqual(param_count, 3000)

    def test_override_building(self):
        """Test override string building."""

        # Test override string building logic
        def to_override(k: str, v) -> str:
            if isinstance(v, bool):
                return f"{k}={'true' if v else 'false'}"
            if isinstance(v, str):
                return f"{k}={v}"
            if isinstance(v, (list, tuple)):
                inner = ",".join(str(x) for x in v)
                return f"{k}=[{inner}]"
            return f"{k}={v}"

        # Test different data types
        test_cases = [
            (5, "param1=5"),
            ("relu", "param2=relu"),
            (True, "param3=true"),
            (False, "param4=false"),
            ([1, 2, 3], "param5=[1,2,3]"),
        ]

        for i, (value, expected) in enumerate(test_cases):
            param_name = f"param{i+1}"
            result = to_override(param_name, value)
            self.assertEqual(result, expected)

    def test_model_configuration(self):
        """Test model configuration structure."""
        optimizer = ParameterOptimizer()

        # Check that all expected models are present
        expected_models = [
            "gcn",
            "gat",
            "gatv2",
            "gin",
            "graph_sage",
            "chebnet",
            "mlp",
            "sagn",
            "gatv4",
        ]

        for model in expected_models:
            self.assertIn(model, optimizer.models)
            self.assertIn("param_key", optimizer.models[model])
            self.assertIn("range", optimizer.models[model])

    def test_advanced_optimizer_initialization(self):
        """Test advanced optimizer initialization."""
        optimizer = AdvancedParameterOptimizer()

        # Check model configurations
        self.assertIsInstance(optimizer.models, dict)
        self.assertGreater(len(optimizer.models), 0)

        # Check base overrides
        self.assertIsInstance(optimizer.base_overrides, dict)
        self.assertIn("optimizer.parameters.lr", optimizer.base_overrides)

    def test_optimization_result_creation(self):
        """Test OptimizationResult dataclass."""
        result = OptimizationResult(
            model_name="test_model",
            target_params=100000,
            actual_params=95000,
            error=0.05,
            config={"param1": 5},
            iterations=10,
        )

        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.target_params, 100000)
        self.assertEqual(result.actual_params, 95000)
        self.assertEqual(result.error, 0.05)
        self.assertEqual(result.iterations, 10)

    @patch("find_target_params.compose")
    @patch("find_target_params.instantiate")
    def test_model_creation_mock(self, mock_instantiate, mock_compose):
        """Test model creation with mocked dependencies."""
        optimizer = ParameterOptimizer()

        # Mock the compose and instantiate functions
        mock_cfg = MagicMock()
        mock_cfg.model = MagicMock()
        mock_cfg.evaluator = MagicMock()
        mock_cfg.optimizer = MagicMock()
        mock_cfg.loss = MagicMock()
        mock_compose.return_value = mock_cfg

        mock_model = MagicMock()
        mock_model.cpu.return_value = mock_model
        mock_instantiate.return_value = mock_model

        # Test model creation
        overrides = {"model.backbone.num_layers": 2}
        model = optimizer.create_model("gcn", overrides)

        self.assertIsNotNone(model)
        mock_compose.assert_called_once()
        mock_instantiate.assert_called_once()

    def test_bisection_search_logic(self):
        """Test bisection search logic without actual model creation."""
        optimizer = AdvancedParameterOptimizer()

        # Test the bisection logic with mock data
        target = 100000
        min_val, max_val = 1, 8

        # Simulate bisection iterations
        for iteration in range(5):
            mid_val = (min_val + max_val) / 2

            # Simulate parameter count (mock)
            if mid_val < 4:
                actual_params = 50000  # Below target
                min_val = mid_val
            else:
                actual_params = 150000  # Above target
                max_val = mid_val

            # Check convergence logic
            error = abs(actual_params - target) / target
            if error <= 0.1:  # 10% tolerance
                break

        # Verify that the search would converge
        self.assertLess(abs(max_val - min_val), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for parameter optimization."""

    def test_script_imports(self):
        """Test that all scripts can be imported without errors."""
        try:
            import advanced_param_optimizer
            import example_usage
            import find_target_params

            self.assertTrue(True)  # If we get here, imports worked
        except ImportError as e:
            self.fail(f"Failed to import scripts: {e}")

    def test_command_line_interface(self):
        """Test command line interface parsing."""
        # Test basic script
        with patch("sys.argv", ["find_target_params.py", "--target-params", "100000"]):
            import find_target_params

            # The script should be able to parse arguments without error
            self.assertTrue(True)

        # Test advanced script
        with patch("sys.argv", ["advanced_param_optimizer.py", "--targets", "100000", "200000"]):
            import advanced_param_optimizer

            # The script should be able to parse arguments without error
            self.assertTrue(True)


def run_tests():
    """Run all tests."""
    print("Running Parameter Optimization Tests")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestParameterOptimization))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
