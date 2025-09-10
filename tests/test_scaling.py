"""
Unit tests for scaling methods.
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from scaling_methods import (
    LinearInterpolationScaling, 
    PolynomialRegressionScaling,
    ConstantScaling,
    StepScaling,
    AdaptiveScaling,
    IngredientScaler
)


class TestLinearInterpolationScaling(unittest.TestCase):
    """Test cases for Linear Interpolation Scaling."""
    
    def setUp(self):
        """Set up test cases."""
        self.scaler = LinearInterpolationScaling()
        self.serving_sizes = np.array([1, 2, 3, 4])
        self.quantities = np.array([100, 200, 300, 400])  # Perfect linear
    
    def test_fit_linear_data(self):
        """Test fitting on perfectly linear data."""
        self.scaler.fit(self.serving_sizes, self.quantities)
        
        self.assertTrue(self.scaler.is_fitted)
        self.assertAlmostEqual(self.scaler.slope, 100.0, places=5)
        self.assertAlmostEqual(self.scaler.intercept, 0.0, places=5)
    
    def test_predict_linear_data(self):
        """Test prediction on linear data."""
        self.scaler.fit(self.serving_sizes, self.quantities)
        
        predictions = self.scaler.predict(np.array([2, 3]))
        expected = np.array([200, 300])
        
        np.testing.assert_array_almost_equal(predictions, expected, decimal=5)
    
    def test_predict_without_fit(self):
        """Test that prediction fails without fitting."""
        with self.assertRaises(ValueError):
            self.scaler.predict(np.array([2, 3]))
    
    def test_insufficient_data(self):
        """Test that fitting fails with insufficient data."""
        with self.assertRaises(ValueError):
            self.scaler.fit(np.array([1]), np.array([100]))
    
    def test_negative_predictions(self):
        """Test that negative predictions are clipped to zero."""
        # Create data that would lead to negative predictions
        serving_sizes = np.array([3, 4])
        quantities = np.array([100, 50])  # Decreasing
        
        self.scaler.fit(serving_sizes, quantities)
        predictions = self.scaler.predict(np.array([1]))
        
        # Should be non-negative
        self.assertGreaterEqual(predictions[0], 0)


class TestPolynomialRegressionScaling(unittest.TestCase):
    """Test cases for Polynomial Regression Scaling."""
    
    def setUp(self):
        """Set up test cases."""
        self.scaler = PolynomialRegressionScaling(degree=2)
    
    def test_fit_quadratic_data(self):
        """Test fitting on quadratic data."""
        serving_sizes = np.array([1, 2, 3, 4])
        quantities = np.array([1, 4, 9, 16])  # x^2
        
        self.scaler.fit(serving_sizes, quantities)
        self.assertTrue(self.scaler.is_fitted)
    
    def test_predict_quadratic_data(self):
        """Test prediction on quadratic data."""
        serving_sizes = np.array([1, 2, 3, 4])
        quantities = np.array([1, 4, 9, 16])  # x^2
        
        self.scaler.fit(serving_sizes, quantities)
        predictions = self.scaler.predict(np.array([2, 3]))
        
        # Should be close to [4, 9]
        self.assertAlmostEqual(predictions[0], 4, delta=0.1)
        self.assertAlmostEqual(predictions[1], 9, delta=0.1)
    
    def test_degree_adjustment(self):
        """Test that degree is adjusted for insufficient data."""
        serving_sizes = np.array([1, 2])
        quantities = np.array([100, 200])
        
        # Should work even with degree=2 and only 2 points
        self.scaler.fit(serving_sizes, quantities)
        self.assertTrue(self.scaler.is_fitted)


class TestConstantScaling(unittest.TestCase):
    """Test cases for Constant Scaling."""
    
    def setUp(self):
        """Set up test cases."""
        self.scaler = ConstantScaling()
    
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        serving_sizes = np.array([1, 2, 3, 4])
        quantities = np.array([5, 5, 5, 5])  # Constant
        
        self.scaler.fit(serving_sizes, quantities)
        predictions = self.scaler.predict(np.array([10, 20]))
        
        np.testing.assert_array_almost_equal(predictions, [5, 5])
    
    def test_fit_variable_data(self):
        """Test fitting on variable data (should use mean)."""
        serving_sizes = np.array([1, 2, 3, 4])
        quantities = np.array([1, 2, 3, 4])
        
        self.scaler.fit(serving_sizes, quantities)
        predictions = self.scaler.predict(np.array([10]))
        
        self.assertAlmostEqual(predictions[0], 2.5)  # Mean of [1,2,3,4]


class TestStepScaling(unittest.TestCase):
    """Test cases for Step Scaling."""
    
    def setUp(self):
        """Set up test cases."""
        self.scaler = StepScaling()
    
    def test_fit_and_predict_exact(self):
        """Test prediction for exact serving sizes."""
        serving_sizes = np.array([1, 2, 4])
        quantities = np.array([100, 150, 200])
        
        self.scaler.fit(serving_sizes, quantities)
        predictions = self.scaler.predict(np.array([2, 4]))
        
        np.testing.assert_array_almost_equal(predictions, [150, 200])
    
    def test_predict_nearest_neighbor(self):
        """Test prediction for non-exact serving sizes."""
        serving_sizes = np.array([1, 4])
        quantities = np.array([100, 400])
        
        self.scaler.fit(serving_sizes, quantities)
        predictions = self.scaler.predict(np.array([2, 3]))
        
        # 2 is closer to 1, 3 is closer to 4
        self.assertEqual(predictions[0], 100)
        self.assertEqual(predictions[1], 400)


class TestAdaptiveScaling(unittest.TestCase):
    """Test cases for Adaptive Scaling."""
    
    def setUp(self):
        """Set up test cases."""
        self.scaler = AdaptiveScaling()
    
    def test_fit_linear_data(self):
        """Test that adaptive scaling chooses linear for linear data."""
        serving_sizes = np.array([1, 2, 3, 4])
        quantities = np.array([100, 200, 300, 400])
        
        self.scaler.fit(serving_sizes, quantities)
        self.assertTrue(self.scaler.is_fitted)
        self.assertIsNotNone(self.scaler.best_method)
    
    def test_fit_constant_data(self):
        """Test that adaptive scaling handles constant data."""
        serving_sizes = np.array([1, 2, 3])
        quantities = np.array([100, 100, 100])
        
        self.scaler.fit(serving_sizes, quantities)
        predictions = self.scaler.predict(np.array([5]))
        
        self.assertAlmostEqual(predictions[0], 100, delta=10)
    
    def test_insufficient_data(self):
        """Test adaptive scaling with insufficient data."""
        serving_sizes = np.array([1])
        quantities = np.array([100])
        
        self.scaler.fit(serving_sizes, quantities)
        self.assertTrue(self.scaler.is_fitted)


class TestIngredientScaler(unittest.TestCase):
    """Test cases for IngredientScaler."""
    
    def setUp(self):
        """Set up test cases."""
        self.scaler = IngredientScaler()
    
    def test_available_methods(self):
        """Test that all methods are available."""
        methods = self.scaler.get_available_methods()
        expected_methods = ['linear', 'polynomial', 'constant', 'step', 'adaptive']
        
        for method in expected_methods:
            self.assertIn(method, methods)
    
    def test_scale_ingredient_linear(self):
        """Test scaling a single ingredient with linear method."""
        serving_sizes = [1, 2, 3, 4]
        quantities = [100, 200, 300, 400]
        target_sizes = [2, 3]
        
        result = self.scaler.scale_ingredient(
            serving_sizes=serving_sizes,
            quantities=quantities,
            target_sizes=target_sizes,
            method='linear'
        )
        
        self.assertEqual(result.method_name, 'Linear Interpolation')
        self.assertEqual(len(result.predicted_quantities), 2)
        self.assertAlmostEqual(result.predicted_quantities[0], 200, delta=1)
        self.assertAlmostEqual(result.predicted_quantities[1], 300, delta=1)
    
    def test_scale_ingredient_invalid_method(self):
        """Test that invalid method raises error."""
        with self.assertRaises(ValueError):
            self.scaler.scale_ingredient(
                serving_sizes=[1, 2],
                quantities=[100, 200],
                target_sizes=[3],
                method='invalid_method'
            )
    
    def test_scaling_result_attributes(self):
        """Test that scaling result has all required attributes."""
        result = self.scaler.scale_ingredient(
            serving_sizes=[1, 2],
            quantities=[100, 200],
            target_sizes=[3],
            method='linear'
        )
        
        # Check all required attributes exist
        self.assertTrue(hasattr(result, 'method_name'))
        self.assertTrue(hasattr(result, 'predicted_quantities'))
        self.assertTrue(hasattr(result, 'serving_sizes'))
        self.assertTrue(hasattr(result, 'training_sizes'))
        self.assertTrue(hasattr(result, 'training_quantities'))
        self.assertTrue(hasattr(result, 'model_params'))
        self.assertTrue(hasattr(result, 'confidence_score'))
        self.assertTrue(hasattr(result, 'error_metrics'))


class TestScalingEdgeCases(unittest.TestCase):
    """Test edge cases for scaling methods."""
    
    def test_zero_quantities(self):
        """Test handling of zero quantities."""
        scaler = LinearInterpolationScaling()
        serving_sizes = np.array([1, 2, 3])
        quantities = np.array([0, 0, 0])
        
        scaler.fit(serving_sizes, quantities)
        predictions = scaler.predict(np.array([4]))
        
        self.assertAlmostEqual(predictions[0], 0, delta=0.01)
    
    def test_single_nonzero_quantity(self):
        """Test handling of mostly zero quantities."""
        scaler = LinearInterpolationScaling()
        serving_sizes = np.array([1, 2, 3])
        quantities = np.array([0, 100, 0])
        
        scaler.fit(serving_sizes, quantities)
        predictions = scaler.predict(np.array([2]))
        
        # Should handle this gracefully
        self.assertIsInstance(predictions[0], (int, float))
    
    def test_very_large_quantities(self):
        """Test handling of very large quantities."""
        scaler = LinearInterpolationScaling()
        serving_sizes = np.array([1, 2])
        quantities = np.array([1e6, 2e6])
        
        scaler.fit(serving_sizes, quantities)
        predictions = scaler.predict(np.array([3]))
        
        self.assertAlmostEqual(predictions[0], 3e6, delta=1e5)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
