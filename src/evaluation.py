"""
Evaluation framework for comparing scaling methods and calculating metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EvaluationMetrics:
    """Data class to store evaluation metrics."""
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Squared Error
    r2: float  # R-squared
    median_ae: float  # Median Absolute Error
    max_ae: float  # Maximum Absolute Error


@dataclass
class MethodEvaluation:
    """Data class to store evaluation results for a scaling method."""
    method_name: str
    overall_metrics: EvaluationMetrics
    per_ingredient_metrics: Dict[str, EvaluationMetrics]
    per_dish_metrics: Dict[str, EvaluationMetrics]
    predictions: Dict[str, Dict[str, List[float]]]  # dish -> ingredient -> predictions
    actuals: Dict[str, Dict[str, List[float]]]      # dish -> ingredient -> actuals
    training_config: Dict[str, Any]


class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(mean_absolute_error(y_true, y_pred))
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Calculate Mean Absolute Percentage Error."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Avoid division by zero
        denominator = np.maximum(np.abs(y_true), epsilon)
        return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared."""
        if len(y_true) < 2:
            return 0.0
        try:
            return float(r2_score(y_true, y_pred))
        except:
            return 0.0
    
    @staticmethod
    def calculate_median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Median Absolute Error."""
        return float(np.median(np.abs(y_true - y_pred)))
    
    @staticmethod
    def calculate_max_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Maximum Absolute Error."""
        return float(np.max(np.abs(y_true - y_pred)))
    
    @classmethod
    def calculate_all_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationMetrics:
        """Calculate all evaluation metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return EvaluationMetrics(
            mae=cls.calculate_mae(y_true, y_pred),
            mape=cls.calculate_mape(y_true, y_pred),
            rmse=cls.calculate_rmse(y_true, y_pred),
            r2=cls.calculate_r2(y_true, y_pred),
            median_ae=cls.calculate_median_ae(y_true, y_pred),
            max_ae=cls.calculate_max_ae(y_true, y_pred)
        )


class CrossValidationEvaluator:
    """Cross-validation evaluator for scaling methods."""
    
    def __init__(self, data_loader, ingredient_scaler):
        """
        Initialize the evaluator.
        
        Args:
            data_loader: RecipeDataLoader instance
            ingredient_scaler: IngredientScaler instance
        """
        self.data_loader = data_loader
        self.ingredient_scaler = ingredient_scaler
        self.metrics_calculator = MetricsCalculator()
    
    def leave_one_out_evaluation(
        self, 
        dish_name: str, 
        method: str = 'adaptive',
        **method_kwargs
    ) -> Dict[str, Any]:
        """
        Perform leave-one-out cross-validation for a single dish.
        
        Args:
            dish_name (str): Name of the dish to evaluate
            method (str): Scaling method to use
            **method_kwargs: Additional method parameters
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        dish_data = self.data_loader.get_dish_data(dish_name)
        serving_sizes = sorted(dish_data['serving_size'].unique())
        
        if len(serving_sizes) < 3:
            raise ValueError(f"Need at least 3 serving sizes for evaluation, got {len(serving_sizes)}")
        
        results = {
            'dish': dish_name,
            'method': method,
            'ingredient_results': {},
            'overall_metrics': None
        }
        
        all_predictions = []
        all_actuals = []
        
        # For each ingredient
        for ingredient in dish_data['ingredient'].unique():
            ingredient_data = dish_data[
                dish_data['ingredient'] == ingredient
            ].sort_values('serving_size')
            
            if len(ingredient_data) < 3:
                continue
            
            ingredient_predictions = []
            ingredient_actuals = []
            
            # Leave-one-out cross validation
            for test_idx in range(len(ingredient_data)):
                # Split data
                train_data = ingredient_data.drop(ingredient_data.index[test_idx])
                test_data = ingredient_data.iloc[test_idx:test_idx+1]
                
                # Extract training and test values
                train_sizes = train_data['serving_size'].tolist()
                train_quantities = train_data['quantity_grams'].tolist()
                test_size = test_data['serving_size'].iloc[0]
                test_quantity = test_data['quantity_grams'].iloc[0]
                
                try:
                    # Scale ingredient
                    scaling_result = self.ingredient_scaler.scale_ingredient(
                        serving_sizes=train_sizes,
                        quantities=train_quantities,
                        target_sizes=[test_size],
                        method=method,
                        **method_kwargs
                    )
                    
                    predicted_quantity = scaling_result.predicted_quantities[0]
                    ingredient_predictions.append(predicted_quantity)
                    ingredient_actuals.append(test_quantity)
                    
                except Exception as e:
                    print(f"Error in LOO for {ingredient}: {e}")
                    continue
            
            if ingredient_predictions:
                # Calculate metrics for this ingredient
                ingredient_metrics = self.metrics_calculator.calculate_all_metrics(
                    ingredient_actuals, ingredient_predictions
                )
                
                results['ingredient_results'][ingredient] = {
                    'metrics': ingredient_metrics,
                    'predictions': ingredient_predictions,
                    'actuals': ingredient_actuals
                }
                
                all_predictions.extend(ingredient_predictions)
                all_actuals.extend(ingredient_actuals)
        
        # Calculate overall metrics
        if all_predictions:
            overall_metrics = self.metrics_calculator.calculate_all_metrics(
                all_actuals, all_predictions
            )
            results['overall_metrics'] = overall_metrics
        
        return results
    
    def k_fold_evaluation(
        self, 
        dish_name: str, 
        training_sizes: List[int],
        test_sizes: List[int],
        method: str = 'adaptive',
        **method_kwargs
    ) -> Dict[str, Any]:
        """
        Perform k-fold style evaluation using specific training and test sizes.
        
        Args:
            dish_name (str): Name of the dish to evaluate
            training_sizes (List[int]): Serving sizes to use for training
            test_sizes (List[int]): Serving sizes to test on
            method (str): Scaling method to use
            **method_kwargs: Additional method parameters
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        dish_data = self.data_loader.get_dish_data(dish_name)
        
        results = {
            'dish': dish_name,
            'method': method,
            'training_sizes': training_sizes,
            'test_sizes': test_sizes,
            'ingredient_results': {},
            'overall_metrics': None
        }
        
        all_predictions = []
        all_actuals = []
        
        # For each ingredient
        for ingredient in dish_data['ingredient'].unique():
            ingredient_data = dish_data[
                dish_data['ingredient'] == ingredient
            ].sort_values('serving_size')
            
            # Get training and test data
            train_data = ingredient_data[
                ingredient_data['serving_size'].isin(training_sizes)
            ]
            test_data = ingredient_data[
                ingredient_data['serving_size'].isin(test_sizes)
            ]
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            try:
                # Scale ingredient
                scaling_result = self.ingredient_scaler.scale_ingredient(
                    serving_sizes=train_data['serving_size'].tolist(),
                    quantities=train_data['quantity_grams'].tolist(),
                    target_sizes=test_data['serving_size'].tolist(),
                    method=method,
                    **method_kwargs
                )
                
                predictions = scaling_result.predicted_quantities
                actuals = test_data['quantity_grams'].tolist()
                
                # Calculate metrics for this ingredient
                ingredient_metrics = self.metrics_calculator.calculate_all_metrics(
                    actuals, predictions
                )
                
                results['ingredient_results'][ingredient] = {
                    'metrics': ingredient_metrics,
                    'predictions': predictions,
                    'actuals': actuals,
                    'test_serving_sizes': test_data['serving_size'].tolist()
                }
                
                all_predictions.extend(predictions)
                all_actuals.extend(actuals)
                
            except Exception as e:
                print(f"Error in k-fold for {ingredient}: {e}")
                continue
        
        # Calculate overall metrics
        if all_predictions:
            overall_metrics = self.metrics_calculator.calculate_all_metrics(
                all_actuals, all_predictions
            )
            results['overall_metrics'] = overall_metrics
        
        return results
    
    def compare_methods(
        self, 
        dish_names: List[str],
        methods: List[str],
        evaluation_type: str = 'leave_one_out',
        **evaluation_kwargs
    ) -> Dict[str, MethodEvaluation]:
        """
        Compare multiple scaling methods across multiple dishes.
        
        Args:
            dish_names (List[str]): List of dishes to evaluate
            methods (List[str]): List of methods to compare
            evaluation_type (str): Type of evaluation ('leave_one_out' or 'k_fold')
            **evaluation_kwargs: Additional evaluation parameters
            
        Returns:
            Dict[str, MethodEvaluation]: Results for each method
        """
        method_results = {}
        
        for method in methods:
            print(f"Evaluating method: {method}")
            
            all_predictions = []
            all_actuals = []
            per_dish_metrics = {}
            per_ingredient_metrics = {}
            predictions_by_dish = {}
            actuals_by_dish = {}
            
            for dish_name in dish_names:
                try:
                    if evaluation_type == 'leave_one_out':
                        dish_results = self.leave_one_out_evaluation(dish_name, method)
                    else:  # k_fold
                        dish_results = self.k_fold_evaluation(dish_name, method, **evaluation_kwargs)
                    
                    if dish_results['overall_metrics']:
                        per_dish_metrics[dish_name] = dish_results['overall_metrics']
                        
                        # Collect predictions and actuals
                        dish_predictions = {}
                        dish_actuals = {}
                        
                        for ingredient, ingredient_result in dish_results['ingredient_results'].items():
                            ingredient_key = f"{dish_name}:{ingredient}"
                            per_ingredient_metrics[ingredient_key] = ingredient_result['metrics']
                            
                            dish_predictions[ingredient] = ingredient_result['predictions']
                            dish_actuals[ingredient] = ingredient_result['actuals']
                            
                            all_predictions.extend(ingredient_result['predictions'])
                            all_actuals.extend(ingredient_result['actuals'])
                        
                        predictions_by_dish[dish_name] = dish_predictions
                        actuals_by_dish[dish_name] = dish_actuals
                
                except Exception as e:
                    print(f"Error evaluating {dish_name} with {method}: {e}")
                    continue
            
            # Calculate overall metrics
            if all_predictions:
                overall_metrics = self.metrics_calculator.calculate_all_metrics(
                    all_actuals, all_predictions
                )
                
                method_results[method] = MethodEvaluation(
                    method_name=method,
                    overall_metrics=overall_metrics,
                    per_ingredient_metrics=per_ingredient_metrics,
                    per_dish_metrics=per_dish_metrics,
                    predictions=predictions_by_dish,
                    actuals=actuals_by_dish,
                    training_config=evaluation_kwargs
                )
        
        return method_results
    
    def generate_evaluation_report(
        self, 
        method_results: Dict[str, MethodEvaluation],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            method_results (Dict[str, MethodEvaluation]): Results from compare_methods
            output_path (Optional[str]): Path to save the report
            
        Returns:
            pd.DataFrame: Summary report
        """
        report_data = []
        
        for method_name, evaluation in method_results.items():
            metrics = evaluation.overall_metrics
            
            report_data.append({
                'Method': method_name,
                'MAE': metrics.mae,
                'MAPE (%)': metrics.mape,
                'RMSE': metrics.rmse,
                'R²': metrics.r2,
                'Median AE': metrics.median_ae,
                'Max AE': metrics.max_ae,
                'Num Dishes': len(evaluation.per_dish_metrics),
                'Num Ingredients': len(evaluation.per_ingredient_metrics)
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('MAE')  # Sort by MAE (lower is better)
        
        if output_path:
            report_df.to_csv(output_path, index=False)
            print(f"Report saved to: {output_path}")
        
        return report_df


    def create_method_evaluation_from_folds(
        self, 
        all_results: Dict[str, List[Dict]]
    ) -> Dict[str, MethodEvaluation]:
        """
        Aggregates results from multiple folds into a single MethodEvaluation object per method.
        """
        method_eval_objects = {}
        for method, fold_results in all_results.items():
            if not fold_results:
                continue

            all_preds = []
            all_acts = []
            per_ingredient_metrics_agg = {}
            
            # Aggregate predictions and actuals from all folds
            for fold in fold_results:
                for ing, res in fold.get('ingredient_results', {}).items():
                    all_preds.extend(res.get('predictions', []))
                    all_acts.extend(res.get('actuals', []))

            if not all_preds:
                continue

            # Calculate overall metrics across all folds
            overall_metrics = self.metrics_calculator.calculate_all_metrics(
                np.array(all_acts), np.array(all_preds)
            )

            # This is a simplified aggregation. A more detailed implementation
            # could average metrics per ingredient across folds.
            method_eval_objects[method] = MethodEvaluation(
                method_name=method,
                overall_metrics=overall_metrics,
                per_ingredient_metrics={}, # Simplified for this aggregation
                per_dish_metrics={}, # Simplified for this aggregation
                predictions={}, # Simplified for this aggregation
                actuals={}, # Simplified for this aggregation
                training_config={} # Simplified for this aggregation
            )
        return method_eval_objects

    def generate_evaluation_report_from_folds(
        self, 
        all_results: Dict[str, List[Dict]]
    ) -> pd.DataFrame:
        """
        Generates a summary report from aggregated fold results.
        """
        report_data = []
        
        for method, fold_results in all_results.items():
            if not fold_results:
                continue

            all_preds = []
            all_acts = []
            
            for fold in fold_results:
                for ing, res in fold.get('ingredient_results', {}).items():
                    all_preds.extend(res.get('predictions', []))
                    all_acts.extend(res.get('actuals', []))

            if not all_preds:
                continue

            metrics = self.metrics_calculator.calculate_all_metrics(
                np.array(all_acts), np.array(all_preds)
            )
            
            report_data.append({
                'Method': method,
                'MAE': metrics.mae,
                'MAPE (%)': metrics.mape,
                'RMSE': metrics.rmse,
                'R²': metrics.r2,
                'Median AE': metrics.median_ae,
                'Max AE': metrics.max_ae,
            })
        
        report_df = pd.DataFrame(report_data)
        return report_df.sort_values('MAE')


class EvaluationVisualizer:
    """Visualizer for evaluation results."""
    
    @staticmethod
    def plot_method_comparison(
        method_results: Dict[str, MethodEvaluation],
        metrics: List[str] = ['mae', 'mape', 'rmse', 'r2'],
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot comparison of methods across different metrics.
        
        Args:
            method_results (Dict[str, MethodEvaluation]): Results to plot
            metrics (List[str]): Metrics to include in the plot
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        methods = list(method_results.keys())
        
        for i, metric in enumerate(metrics[:4]):
            ax = axes[i]
            
            values = []
            for method in methods:
                evaluation = method_results[method]
                value = getattr(evaluation.overall_metrics, metric)
                values.append(value)
            
            bars = ax.bar(methods, values)
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel('Value')
            
            # Color bars (lower is better for MAE, MAPE, RMSE; higher is better for R²)
            if metric in ['mae', 'mape', 'rmse']:
                best_idx = np.argmin(values)
            else:  # R²
                best_idx = np.argmax(values)
            
            for j, bar in enumerate(bars):
                if j == best_idx:
                    bar.set_color('green')
                else:
                    bar.set_color('lightblue')
            
            # Rotate x-axis labels if needed
            if len(methods) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_predictions_vs_actual(
        method_results: Dict[str, MethodEvaluation],
        dish_name: str,
        ingredient: str,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot predictions vs actual values for a specific ingredient.
        
        Args:
            method_results (Dict[str, MethodEvaluation]): Results to plot
            dish_name (str): Dish name
            ingredient (str): Ingredient name
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for method_name, evaluation in method_results.items():
            if dish_name in evaluation.predictions and ingredient in evaluation.predictions[dish_name]:
                predictions = evaluation.predictions[dish_name][ingredient]
                actuals = evaluation.actuals[dish_name][ingredient]
                
                ax.scatter(actuals, predictions, label=method_name, alpha=0.7)
        
        # Plot perfect prediction line
        all_values = []
        for evaluation in method_results.values():
            if dish_name in evaluation.actuals and ingredient in evaluation.actuals[dish_name]:
                all_values.extend(evaluation.actuals[dish_name][ingredient])
        
        if all_values:
            min_val, max_val = min(all_values), max(all_values)
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Quantity (grams)')
        ax.set_ylabel('Predicted Quantity (grams)')
        ax.set_title(f'Predictions vs Actual: {dish_name} - {ingredient}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


if __name__ == "__main__":
    # Example usage would go here
    print("Evaluation module loaded successfully!")
    print("Use this module with a data loader and ingredient scaler to evaluate scaling methods.")
