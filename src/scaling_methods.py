"""
Scaling methods for ingredient quantity prediction across different serving sizes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ScalingResult:
    """Data class to store scaling prediction results."""
    method_name: str
    predicted_quantities: List[float]
    serving_sizes: List[int]
    training_sizes: List[int]
    training_quantities: List[float]
    model_params: Dict[str, Any]
    confidence_score: float
    error_metrics: Dict[str, float] = None


class BaseScalingMethod:
    """Base class for all scaling methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
    
    def fit(self, serving_sizes: np.ndarray, quantities: np.ndarray) -> 'BaseScalingMethod':
        """
        Fit the scaling model to training data.
        
        Args:
            serving_sizes (np.ndarray): Training serving sizes
            quantities (np.ndarray): Training quantities
            
        Returns:
            BaseScalingMethod: Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, serving_sizes: np.ndarray) -> np.ndarray:
        """
        Predict quantities for given serving sizes.
        
        Args:
            serving_sizes (np.ndarray): Target serving sizes
            
        Returns:
            np.ndarray: Predicted quantities
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}


class LinearInterpolationScaling(BaseScalingMethod):
    """Linear interpolation scaling method."""
    
    def __init__(self):
        super().__init__("Linear Interpolation")
        self.slope = None
        self.intercept = None
    
    def fit(self, serving_sizes: np.ndarray, quantities: np.ndarray) -> 'LinearInterpolationScaling':
        """Fit linear interpolation model."""
        if len(serving_sizes) < 2:
            raise ValueError("Need at least 2 data points for linear interpolation")
        
        # Use linear regression for fitting
        X = serving_sizes.reshape(-1, 1)
        y = quantities
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        self.slope = self.model.coef_[0]
        self.intercept = self.model.intercept_
        self.is_fitted = True
        
        return self
    
    def predict(self, serving_sizes: np.ndarray) -> np.ndarray:
        """Predict using linear interpolation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = serving_sizes.reshape(-1, 1)
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        return np.maximum(predictions, 0)
    
    def get_params(self) -> Dict[str, Any]:
        """Get linear model parameters."""
        return {
            'slope': self.slope,
            'intercept': self.intercept,
            'r_squared': getattr(self.model, 'score', lambda x, y: 0)(
                np.array([[1], [2], [3], [4]]), 
                np.array([self.intercept + self.slope, 
                         self.intercept + 2*self.slope,
                         self.intercept + 3*self.slope,
                         self.intercept + 4*self.slope])
            ) if self.is_fitted else 0
        }


class PolynomialRegressionScaling(BaseScalingMethod):
    """Polynomial regression scaling method."""
    
    def __init__(self, degree: int = 2, alpha: float = 1.0):
        super().__init__(f"Polynomial Regression (degree={degree})")
        self.degree = degree
        self.alpha = alpha
        self.pipeline = None
    
    def fit(self, serving_sizes: np.ndarray, quantities: np.ndarray) -> 'PolynomialRegressionScaling':
        """Fit polynomial regression model."""
        if len(serving_sizes) <= self.degree:
            # Reduce degree if not enough points
            self.degree = max(1, len(serving_sizes) - 1)
        
        X = serving_sizes.reshape(-1, 1)
        y = quantities
        
        # Create polynomial features pipeline with ridge regression
        self.pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree, include_bias=True)),
            ('ridge', Ridge(alpha=self.alpha))
        ])
        
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, serving_sizes: np.ndarray) -> np.ndarray:
        """Predict using polynomial regression."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = serving_sizes.reshape(-1, 1)
        predictions = self.pipeline.predict(X)
        
        # Ensure non-negative predictions
        return np.maximum(predictions, 0)
    
    def get_params(self) -> Dict[str, Any]:
        """Get polynomial model parameters."""
        if not self.is_fitted:
            return {'degree': self.degree, 'alpha': self.alpha}
        
        return {
            'degree': self.degree,
            'alpha': self.alpha,
            'coefficients': self.pipeline.named_steps['ridge'].coef_.tolist(),
            'intercept': self.pipeline.named_steps['ridge'].intercept_
        }


class ConstantScaling(BaseScalingMethod):
    """Constant scaling method for ingredients that don't scale."""
    
    def __init__(self):
        super().__init__("Constant Scaling")
        self.constant_value = None
    
    def fit(self, serving_sizes: np.ndarray, quantities: np.ndarray) -> 'ConstantScaling':
        """Fit constant model (use mean value)."""
        self.constant_value = np.mean(quantities)
        self.is_fitted = True
        return self
    
    def predict(self, serving_sizes: np.ndarray) -> np.ndarray:
        """Predict constant value for all serving sizes."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return np.full(len(serving_sizes), self.constant_value)
    
    def get_params(self) -> Dict[str, Any]:
        """Get constant model parameters."""
        return {'constant_value': self.constant_value}


class StepScaling(BaseScalingMethod):
    """Step scaling method for ingredients that scale in discrete steps."""
    
    def __init__(self, step_threshold: float = 2.0):
        super().__init__("Step Scaling")
        self.step_threshold = step_threshold
        self.step_map = {}
    
    def fit(self, serving_sizes: np.ndarray, quantities: np.ndarray) -> 'StepScaling':
        """Fit step scaling model."""
        # Create a mapping from serving sizes to quantities
        for size, quantity in zip(serving_sizes, quantities):
            self.step_map[size] = quantity
        
        self.is_fitted = True
        return self
    
    def predict(self, serving_sizes: np.ndarray) -> np.ndarray:
        """Predict using step scaling (nearest neighbor approach)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        known_sizes = list(self.step_map.keys())
        
        for target_size in serving_sizes:
            if target_size in self.step_map:
                predictions.append(self.step_map[target_size])
            else:
                # Find nearest serving size
                nearest_size = min(known_sizes, key=lambda x: abs(x - target_size))
                predictions.append(self.step_map[nearest_size])
        
        return np.array(predictions)
    
    def get_params(self) -> Dict[str, Any]:
        """Get step scaling parameters."""
        return {
            'step_threshold': self.step_threshold,
            'step_map': self.step_map
        }


class AdaptiveScaling(BaseScalingMethod):
    """Adaptive scaling that chooses the best method based on data characteristics."""
    
    def __init__(self):
        super().__init__("Adaptive Scaling")
        self.best_method = None
        self.methods = [
            LinearInterpolationScaling(),
            PolynomialRegressionScaling(degree=2),
            ConstantScaling(),
            StepScaling()
        ]
    
    def fit(self, serving_sizes: np.ndarray, quantities: np.ndarray) -> 'AdaptiveScaling':
        """Fit adaptive model by selecting best method."""
        if len(serving_sizes) < 2:
            self.best_method = ConstantScaling().fit(serving_sizes, quantities)
            self.is_fitted = True
            return self
        
        best_score = float('inf')
        
        # Try each method and select based on cross-validation error
        for method in self.methods:
            try:
                # Simple leave-one-out cross validation
                errors = []
                
                if len(serving_sizes) >= 3:  # Need at least 3 points for CV
                    for i in range(len(serving_sizes)):
                        # Leave one out
                        train_sizes = np.concatenate([serving_sizes[:i], serving_sizes[i+1:]])
                        train_quantities = np.concatenate([quantities[:i], quantities[i+1:]])
                        test_size = serving_sizes[i:i+1]
                        test_quantity = quantities[i]
                        
                        # Fit and predict
                        temp_method = type(method)()
                        if hasattr(temp_method, 'degree'):
                            temp_method = type(method)(degree=method.degree)
                        
                        temp_method.fit(train_sizes, train_quantities)
                        pred = temp_method.predict(test_size)
                        
                        error = abs(pred[0] - test_quantity)
                        errors.append(error)
                    
                    avg_error = np.mean(errors)
                else:
                    # Use simple fit for small datasets
                    method.fit(serving_sizes, quantities)
                    pred = method.predict(serving_sizes)
                    avg_error = mean_absolute_error(quantities, pred)
                
                if avg_error < best_score:
                    best_score = avg_error
                    self.best_method = method
                    
            except Exception as e:
                continue  # Skip methods that fail
        
        # Fallback to linear if no method worked
        if self.best_method is None:
            self.best_method = LinearInterpolationScaling()
        
        # Fit the best method on all data
        self.best_method.fit(serving_sizes, quantities)
        self.is_fitted = True
        
        return self
    
    def predict(self, serving_sizes: np.ndarray) -> np.ndarray:
        """Predict using the best selected method."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.best_method.predict(serving_sizes)
    
    def get_params(self) -> Dict[str, Any]:
        """Get adaptive scaling parameters."""
        return {
            'selected_method': self.best_method.name,
            'method_params': self.best_method.get_params()
        }


class IngredientScaler:
    """Main class for scaling ingredient quantities across serving sizes."""
    
    def __init__(self):
        """Initialize the ingredient scaler."""
        self.available_methods = {
            'linear': LinearInterpolationScaling,
            'polynomial': PolynomialRegressionScaling,
            'constant': ConstantScaling,
            'step': StepScaling,
            'adaptive': AdaptiveScaling
        }
    
    def scale_ingredient(
        self, 
        serving_sizes: List[int], 
        quantities: List[float],
        target_sizes: List[int],
        method: str = 'adaptive',
        **method_kwargs
    ) -> ScalingResult:
        """
        Scale ingredient quantities to target serving sizes.
        
        Args:
            serving_sizes (List[int]): Known serving sizes
            quantities (List[float]): Known quantities
            target_sizes (List[int]): Target serving sizes to predict
            method (str): Scaling method to use
            **method_kwargs: Additional arguments for the scaling method
            
        Returns:
            ScalingResult: Scaling results with predictions and metadata
        """
        # Convert to numpy arrays
        X_train = np.array(serving_sizes)
        y_train = np.array(quantities)
        X_target = np.array(target_sizes)
        
        # Get scaling method
        if method not in self.available_methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.available_methods.keys())}")
        
        method_class = self.available_methods[method]
        
        # Initialize method with kwargs
        if method == 'polynomial':
            scaler = method_class(**method_kwargs)
        else:
            scaler = method_class()
        
        # Fit and predict
        scaler.fit(X_train, y_train)
        predictions = scaler.predict(X_target)
        
        # Calculate confidence score based on training data fit
        y_pred_train = scaler.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
        # Confidence based on relative error
        mean_quantity = np.mean(y_train)
        relative_error = train_mae / (mean_quantity + 1e-8)
        confidence = max(0, 1 - relative_error)
        
        return ScalingResult(
            method_name=scaler.name,
            predicted_quantities=predictions.tolist(),
            serving_sizes=target_sizes,
            training_sizes=serving_sizes,
            training_quantities=quantities,
            model_params=scaler.get_params(),
            confidence_score=confidence,
            error_metrics={
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'relative_error': relative_error
            }
        )
    
    def scale_recipe(
        self, 
        recipe_data: pd.DataFrame,
        training_sizes: List[int],
        target_sizes: List[int],
        method: str = 'adaptive'
    ) -> Dict[str, ScalingResult]:
        """
        Scale all ingredients in a recipe.
        
        Args:
            recipe_data (pd.DataFrame): Recipe data with ingredient quantities
            training_sizes (List[int]): Serving sizes to use for training
            target_sizes (List[int]): Target serving sizes to predict
            method (str): Scaling method to use
            
        Returns:
            Dict[str, ScalingResult]: Results for each ingredient
        """
        results = {}
        
        # Group by ingredient
        for ingredient in recipe_data['ingredient'].unique():
            ingredient_data = recipe_data[
                recipe_data['ingredient'] == ingredient
            ].sort_values('serving_size')
            
            # Filter training data
            train_data = ingredient_data[
                ingredient_data['serving_size'].isin(training_sizes)
            ]
            
            if len(train_data) >= 1:  # Need at least one training point
                serving_sizes = train_data['serving_size'].tolist()
                quantities = train_data['quantity_grams'].tolist()
                
                try:
                    result = self.scale_ingredient(
                        serving_sizes=serving_sizes,
                        quantities=quantities,
                        target_sizes=target_sizes,
                        method=method
                    )
                    results[ingredient] = result
                except Exception as e:
                    print(f"Error scaling {ingredient}: {e}")
                    continue
        
        return results
    
    def get_available_methods(self) -> List[str]:
        """Get list of available scaling methods."""
        return list(self.available_methods.keys())


if __name__ == "__main__":
    # Example usage
    scaler = IngredientScaler()
    
    # Test data
    serving_sizes = [1, 2, 3, 4]
    quantities = [100, 200, 300, 400]  # Linear scaling
    target_sizes = [2, 3]
    
    print("Testing scaling methods:")
    for method in scaler.get_available_methods():
        try:
            result = scaler.scale_ingredient(
                serving_sizes=serving_sizes,
                quantities=quantities,
                target_sizes=target_sizes,
                method=method
            )
            print(f"\n{method.upper()}:")
            print(f"  Predictions: {result.predicted_quantities}")
            print(f"  Confidence: {result.confidence_score:.3f}")
            print(f"  Method: {result.method_name}")
        except Exception as e:
            print(f"\n{method.upper()}: Error - {e}")
