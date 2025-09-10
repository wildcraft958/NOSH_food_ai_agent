"""
Main application module for the Food AI Ingredient Scaling System.
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_loader import RecipeDataLoader, load_recipe_data
from preprocessing import QuantityPreprocessor
from scaling_methods import IngredientScaler
from evaluation import CrossValidationEvaluator, EvaluationVisualizer
import matplotlib.pyplot as plt


class FoodAIScalingSystem:
    """Main system class for ingredient scaling."""
    
    def __init__(self, data_path: str):
        """
        Initialize the scaling system.
        
        Args:
            data_path (str): Path to the recipe data JSON file
        """
        self.data_path = data_path
        self.data_loader = None
        self.preprocessor = QuantityPreprocessor()
        self.scaler = IngredientScaler()
        self.evaluator = None
        self.processed_data = None
        
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess recipe data."""
        print("Loading recipe data...")
        self.data_loader = load_recipe_data(self.data_path)
        
        print("Preprocessing data...")
        raw_data = self.data_loader.processed_data
        self.processed_data = self.preprocessor.preprocess_recipe_data(raw_data)
        
        # Initialize evaluator
        self.evaluator = CrossValidationEvaluator(self.data_loader, self.scaler)
        
        print(f"Loaded {len(self.processed_data)} ingredient entries across {len(self.data_loader.get_available_dishes())} dishes")
    
    def scale_single_ingredient(
        self, 
        dish_name: str, 
        ingredient: str,
        training_sizes: List[int],
        target_sizes: List[int],
        method: str = 'adaptive'
    ) -> Dict[str, Any]:
        """
        Scale a single ingredient to target serving sizes.
        
        Args:
            dish_name (str): Name of the dish
            ingredient (str): Name of the ingredient
            training_sizes (List[int]): Serving sizes to use for training
            target_sizes (List[int]): Target serving sizes
            method (str): Scaling method to use
            
        Returns:
            Dict[str, Any]: Scaling results
        """
        # Get ingredient data
        ingredient_data = self.data_loader.get_ingredient_scaling_data(dish_name, ingredient)
        
        if ingredient_data.empty:
            raise ValueError(f"No data found for {ingredient} in {dish_name}")
        
        # Filter training data
        train_data = ingredient_data[ingredient_data['serving_size'].isin(training_sizes)]
        
        if train_data.empty:
            raise ValueError(f"No training data found for specified serving sizes: {training_sizes}")
        
        # Scale ingredient
        result = self.scaler.scale_ingredient(
            serving_sizes=train_data['serving_size'].tolist(),
            quantities=train_data['quantity_grams'].tolist(),
            target_sizes=target_sizes,
            method=method
        )
        
        return {
            'dish': dish_name,
            'ingredient': ingredient,
            'method': method,
            'training_sizes': training_sizes,
            'target_sizes': target_sizes,
            'predictions': result.predicted_quantities,
            'confidence': result.confidence_score,
            'model_params': result.model_params
        }
    
    def scale_full_recipe(
        self,
        dish_name: str,
        training_sizes: List[int],
        target_sizes: List[int],
        method: str = 'adaptive'
    ) -> Dict[str, Any]:
        """
        Scale all ingredients in a recipe to target serving sizes.
        
        Args:
            dish_name (str): Name of the dish
            training_sizes (List[int]): Serving sizes to use for training
            target_sizes (List[int]): Target serving sizes
            method (str): Scaling method to use
            
        Returns:
            Dict[str, Any]: Complete recipe scaling results
        """
        dish_data = self.data_loader.get_dish_data(dish_name)
        
        if dish_data.empty:
            raise ValueError(f"No data found for dish: {dish_name}")
        
        # Scale all ingredients
        ingredient_results = {}
        
        for ingredient in dish_data['ingredient'].unique():
            try:
                result = self.scale_single_ingredient(
                    dish_name, ingredient, training_sizes, target_sizes, method
                )
                ingredient_results[ingredient] = result
            except Exception as e:
                print(f"Warning: Could not scale {ingredient}: {e}")
                continue
        
        return {
            'dish': dish_name,
            'method': method,
            'training_sizes': training_sizes,
            'target_sizes': target_sizes,
            'ingredients': ingredient_results,
            'total_ingredients': len(ingredient_results)
        }
    
    def evaluate_methods(
        self,
        dish_names: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        evaluation_type: str = 'leave_one_out',
        training_sizes: Optional[List[int]] = None,
        test_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate scaling methods across dishes.
        
        Args:
            dish_names (Optional[List[str]]): Dishes to evaluate (all if None)
            methods (Optional[List[str]]): Methods to evaluate (all if None)
            evaluation_type (str): 'leave_one_out' or 'k_fold'
            training_sizes (Optional[List[int]]): Training sizes for k_fold
            test_sizes (Optional[List[int]]): Test sizes for k_fold
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        if dish_names is None:
            dish_names = self.data_loader.get_available_dishes()
        
        if methods is None:
            methods = self.scaler.get_available_methods()
        
        print(f"Evaluating {len(methods)} methods on {len(dish_names)} dishes...")
        
        # Prepare evaluation kwargs
        eval_kwargs = {}
        if evaluation_type == 'k_fold':
            if training_sizes is None or test_sizes is None:
                raise ValueError("training_sizes and test_sizes required for k_fold evaluation")
            eval_kwargs['training_sizes'] = training_sizes
            eval_kwargs['test_sizes'] = test_sizes
        
        # Run evaluation
        method_results = self.evaluator.compare_methods(
            dish_names=dish_names,
            methods=methods,
            evaluation_type=evaluation_type,
            **eval_kwargs
        )
        
        # Generate report
        report_df = self.evaluator.generate_evaluation_report(method_results)
        
        return {
            'method_results': method_results,
            'summary_report': report_df,
            'evaluation_config': {
                'dish_names': dish_names,
                'methods': methods,
                'evaluation_type': evaluation_type,
                **eval_kwargs
            }
        }
    
    def create_visualizations(
        self, 
        evaluation_results: Dict[str, Any],
        output_dir: str = 'plots'
    ):
        """
        Create visualization plots for evaluation results.
        
        Args:
            evaluation_results (Dict[str, Any]): Results from evaluate_methods
            output_dir (str): Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        method_results = evaluation_results['method_results']
        
        # Method comparison plot
        print("Creating method comparison plot...")
        fig1 = EvaluationVisualizer.plot_method_comparison(method_results)
        fig1.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Predictions vs actual plots for each dish and main ingredients
        for dish_name in self.data_loader.get_available_dishes():
            dish_data = self.data_loader.get_dish_data(dish_name)
            
            # Get main ingredients (top 3 by average quantity)
            ingredient_quantities = dish_data.groupby('ingredient')['quantity_grams'].mean().sort_values(ascending=False)
            main_ingredients = ingredient_quantities.head(3).index.tolist()
            
            for ingredient in main_ingredients:
                try:
                    print(f"Creating plot for {dish_name} - {ingredient}...")
                    fig2 = EvaluationVisualizer.plot_predictions_vs_actual(
                        method_results, dish_name, ingredient
                    )
                    
                    safe_dish_name = dish_name.replace(' ', '_').replace('/', '_')
                    safe_ingredient = ingredient.replace(' ', '_').replace('/', '_')
                    filename = f'predictions_{safe_dish_name}_{safe_ingredient}.png'
                    
                    fig2.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                except Exception as e:
                    print(f"Could not create plot for {dish_name} - {ingredient}: {e}")
                    continue
        
        print(f"Plots saved to: {output_dir}")
    
    def export_results(
        self, 
        results: Dict[str, Any], 
        output_path: str,
        format: str = 'json'
    ):
        """
        Export results to file.
        
        Args:
            results (Dict[str, Any]): Results to export
            output_path (str): Output file path
            format (str): Export format ('json' or 'csv')
        """
        if format == 'json':
            # Convert numpy arrays and other non-serializable objects
            serializable_results = self._make_serializable(results)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
        elif format == 'csv' and 'summary_report' in results:
            results['summary_report'].to_csv(output_path, index=False)
        
        print(f"Results exported to: {output_path}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def print_summary(self):
        """Print system summary."""
        print("\n" + "="*60)
        print("FOOD AI INGREDIENT SCALING SYSTEM SUMMARY")
        print("="*60)
        
        dishes = self.data_loader.get_available_dishes()
        print(f"Available dishes: {len(dishes)}")
        for dish in dishes:
            ingredients = self.data_loader.get_ingredients_for_dish(dish)
            print(f"  - {dish}: {len(ingredients)} ingredients")
        
        print(f"\nAvailable scaling methods: {', '.join(self.scaler.get_available_methods())}")
        
        print(f"\nTotal data points: {len(self.processed_data)}")
        print("="*60)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Food AI Ingredient Scaling System')
    
    parser.add_argument('--data', type=str, default='data/paneer_recipes.json',
                       help='Path to recipe data JSON file')
    parser.add_argument('--action', type=str, choices=['scale', 'evaluate', 'demo'], 
                       default='demo', help='Action to perform')
    
    # Scaling arguments
    parser.add_argument('--dish', type=str, help='Dish name for scaling')
    parser.add_argument('--ingredient', type=str, help='Ingredient name for single ingredient scaling')
    parser.add_argument('--train-sizes', type=int, nargs='+', default=[1, 4],
                       help='Training serving sizes')
    parser.add_argument('--target-sizes', type=int, nargs='+', default=[2, 3],
                       help='Target serving sizes')
    parser.add_argument('--method', type=str, default='adaptive',
                       help='Scaling method to use')
    
    # Evaluation arguments
    parser.add_argument('--eval-type', type=str, choices=['leave_one_out', 'k_fold'],
                       default='leave_one_out', help='Evaluation type')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['linear', 'polynomial', 'adaptive'],
                       help='Methods to evaluate')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--plots', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize system
    try:
        system = FoodAIScalingSystem(args.data)
        system.print_summary()
    except Exception as e:
        print(f"Error initializing system: {e}")
        return 1
    
    if args.action == 'demo':
        # Demo mode - run evaluation and scaling examples
        print("\n" + "="*60)
        print("RUNNING DEMO")
        print("="*60)
        
        # Example evaluation
        print("\n1. Evaluating scaling methods...")
        try:
            eval_results = system.evaluate_methods(
                methods=['linear', 'polynomial', 'adaptive'],
                evaluation_type='leave_one_out'
            )
            
            print("\nEvaluation Summary:")
            print(eval_results['summary_report'])
            
            # Save evaluation results
            system.export_results(
                eval_results, 
                os.path.join(args.output_dir, 'evaluation_results.json')
            )
            
            eval_results['summary_report'].to_csv(
                os.path.join(args.output_dir, 'evaluation_summary.csv'), 
                index=False
            )
            
            # Create plots if requested
            if args.plots:
                system.create_visualizations(eval_results, 
                                           os.path.join(args.output_dir, 'plots'))
        
        except Exception as e:
            print(f"Error in evaluation: {e}")
        
        # Example scaling
        print("\n2. Example recipe scaling...")
        try:
            dish_name = system.data_loader.get_available_dishes()[0]  # First dish
            
            scaling_result = system.scale_full_recipe(
                dish_name=dish_name,
                training_sizes=[1, 4],
                target_sizes=[2, 3],
                method='adaptive'
            )
            
            print(f"\nScaling {dish_name} from [1, 4] to [2, 3] servings:")
            print(f"Successfully scaled {scaling_result['total_ingredients']} ingredients")
            
            # Show first few ingredients
            for i, (ingredient, result) in enumerate(scaling_result['ingredients'].items()):
                if i >= 3:  # Show only first 3
                    break
                print(f"  {ingredient}:")
                print(f"    Target 2 servings: {result['predictions'][0]:.2f}g")
                print(f"    Target 3 servings: {result['predictions'][1]:.2f}g")
                print(f"    Confidence: {result['confidence']:.3f}")
            
            # Save scaling results
            system.export_results(
                scaling_result,
                os.path.join(args.output_dir, 'scaling_example.json')
            )
        
        except Exception as e:
            print(f"Error in scaling example: {e}")
    
    elif args.action == 'scale':
        # Scaling mode
        if not args.dish:
            print("Error: --dish required for scaling action")
            return 1
        
        try:
            if args.ingredient:
                # Single ingredient scaling
                result = system.scale_single_ingredient(
                    dish_name=args.dish,
                    ingredient=args.ingredient,
                    training_sizes=args.train_sizes,
                    target_sizes=args.target_sizes,
                    method=args.method
                )
            else:
                # Full recipe scaling
                result = system.scale_full_recipe(
                    dish_name=args.dish,
                    training_sizes=args.train_sizes,
                    target_sizes=args.target_sizes,
                    method=args.method
                )
            
            print("Scaling Results:")
            print(json.dumps(result, indent=2, default=str))
            
            # Save results
            output_file = os.path.join(args.output_dir, 'scaling_results.json')
            system.export_results(result, output_file)
        
        except Exception as e:
            print(f"Error in scaling: {e}")
            return 1
    
    elif args.action == 'evaluate':
        # Evaluation mode
        try:
            eval_kwargs = {}
            if args.eval_type == 'k_fold':
                eval_kwargs['training_sizes'] = args.train_sizes
                eval_kwargs['test_sizes'] = args.target_sizes
            
            results = system.evaluate_methods(
                methods=args.methods,
                evaluation_type=args.eval_type,
                **eval_kwargs
            )
            
            print("Evaluation Results:")
            print(results['summary_report'])
            
            # Save results
            system.export_results(
                results,
                os.path.join(args.output_dir, 'evaluation_results.json')
            )
            
            results['summary_report'].to_csv(
                os.path.join(args.output_dir, 'evaluation_summary.csv'),
                index=False
            )
            
            # Create plots if requested
            if args.plots:
                system.create_visualizations(results, 
                                           os.path.join(args.output_dir, 'plots'))
        
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return 1
    
    print(f"\nAll outputs saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
