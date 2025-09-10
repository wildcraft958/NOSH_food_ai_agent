#!/usr/bin/env python3
"""
Simplified demo script for the Food AI Ingredient Scaling System.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

def main():
    print("🍳 Food AI Ingredient Scaling System - Demo")
    print("=" * 60)
    
    try:
        # Initialize the system
        from data_loader import load_recipe_data
        from scaling_methods import IngredientScaler
        from evaluation import CrossValidationEvaluator, MetricsCalculator
        
        print("Loading data...")
        loader = load_recipe_data("data/paneer_recipes.json")
        scaler = IngredientScaler()
        evaluator = CrossValidationEvaluator(loader, scaler)
        metrics_calc = MetricsCalculator()
        
        print(f"✓ Loaded {len(loader.processed_data)} ingredient entries")
        print(f"✓ Available dishes: {loader.get_available_dishes()}")
        print(f"✓ Available methods: {scaler.get_available_methods()}")
        
        # Demo 1: Single ingredient scaling
        print("\n" + "=" * 60)
        print("DEMO 1: Scaling Paneer in Palak Paneer")
        print("=" * 60)
        
        dish = "palak_paneer"
        ingredient = "Paneer"
        
        # Get data for paneer
        paneer_data = loader.get_ingredient_scaling_data(dish, ingredient)
        print(f"\nOriginal Paneer quantities:")
        for _, row in paneer_data.iterrows():
            print(f"  {row['serving_size']} serving(s): {row['quantity_grams']}g")
        
        # Scale using different methods
        serving_sizes = paneer_data['serving_size'].tolist()
        quantities = paneer_data['quantity_grams'].tolist()
        
        print(f"\nTraining on servings {serving_sizes[:2]}, predicting for serving 3:")
        
        for method in ['linear', 'adaptive']:
            try:
                result = scaler.scale_ingredient(
                    serving_sizes=serving_sizes[:2],
                    quantities=quantities[:2],
                    target_sizes=[3],
                    method=method
                )
                actual = quantities[2]  # 3rd serving actual value
                predicted = result.predicted_quantities[0]
                error = abs(predicted - actual)
                
                print(f"  {method.upper()}: {predicted:.1f}g (actual: {actual}g, error: {error:.1f}g)")
                
            except Exception as e:
                print(f"  {method.upper()}: Error - {e}")
        
        # Demo 2: Full recipe scaling
        print("\n" + "=" * 60)
        print("DEMO 2: Scaling Full Palak Paneer Recipe")
        print("=" * 60)
        
        dish_data = loader.get_dish_data("palak_paneer")
        ingredients = dish_data['ingredient'].unique()
        
        print(f"\nScaling recipe from 1&4 servings to 2&3 servings:")
        print(f"Recipe has {len(ingredients)} ingredients")
        
        scaling_results = {}
        for ingredient in ingredients[:5]:  # Show first 5 ingredients
            try:
                ingredient_data = loader.get_ingredient_scaling_data("palak_paneer", ingredient)
                if len(ingredient_data) >= 3:
                    serving_sizes = ingredient_data['serving_size'].tolist()
                    quantities = ingredient_data['quantity_grams'].tolist()
                    
                    result = scaler.scale_ingredient(
                        serving_sizes=[serving_sizes[0], serving_sizes[3]],  # 1st and 4th servings
                        quantities=[quantities[0], quantities[3]],
                        target_sizes=[2, 3],
                        method='adaptive'
                    )
                    
                    scaling_results[ingredient] = {
                        'predictions': result.predicted_quantities,
                        'actual': [quantities[1], quantities[2]],  # 2nd and 3rd servings
                        'confidence': result.confidence_score
                    }
                    
            except Exception as e:
                print(f"Error scaling {ingredient}: {e}")
                continue
        
        # Display results
        print(f"\nScaling Results (showing top {len(scaling_results)} ingredients):")
        print("-" * 80)
        print(f"{'Ingredient':<25} {'2 Servings':<15} {'3 Servings':<15} {'Confidence':<12}")
        print("-" * 80)
        
        for ingredient, result in scaling_results.items():
            pred_2, pred_3 = result['predictions']
            actual_2, actual_3 = result['actual']
            confidence = result['confidence']
            
            print(f"{ingredient[:24]:<25} {pred_2:.1f}g ({actual_2}g)  {pred_3:.1f}g ({actual_3}g)    {confidence:.3f}")
        
        # Demo 3: Method evaluation
        print("\n" + "=" * 60)
        print("DEMO 3: Method Evaluation")
        print("=" * 60)
        
        print("\nEvaluating methods on Palak Paneer using leave-one-out cross-validation...")
        
        try:
            loo_results = evaluator.leave_one_out_evaluation("palak_paneer", method="adaptive")
            
            if loo_results['overall_metrics']:
                metrics = loo_results['overall_metrics']
                print(f"\nAdaptive Method Performance:")
                print(f"  Mean Absolute Error (MAE): {metrics.mae:.2f}g")
                print(f"  Mean Absolute Percentage Error (MAPE): {metrics.mape:.1f}%")
                print(f"  Root Mean Squared Error (RMSE): {metrics.rmse:.2f}g")
                print(f"  R-squared (R²): {metrics.r2:.3f}")
                
                print(f"\nIngredient-wise results:")
                for ingredient, result in loo_results['ingredient_results'].items():
                    ing_metrics = result['metrics']
                    print(f"  {ingredient}: MAE={ing_metrics.mae:.1f}g, R²={ing_metrics.r2:.3f}")
            
        except Exception as e:
            print(f"Evaluation error: {e}")
        
        # Demo 4: Save results
        print("\n" + "=" * 60)
        print("DEMO 4: Saving Results")
        print("=" * 60)
        
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save scaling results
        output_data = {
            'dish': 'palak_paneer',
            'scaling_method': 'adaptive',
            'training_servings': [1, 4],
            'target_servings': [2, 3],
            'results': scaling_results,
            'evaluation': {
                'mae': metrics.mae if 'metrics' in locals() else None,
                'mape': metrics.mape if 'metrics' in locals() else None,
                'rmse': metrics.rmse if 'metrics' in locals() else None,
                'r2': metrics.r2 if 'metrics' in locals() else None
            }
        }
        
        output_file = output_dir / "demo_results.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {output_file}")
        
        # Create summary report
        summary_file = output_dir / "demo_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Food AI Ingredient Scaling System - Demo Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset: {len(loader.processed_data)} ingredient entries\n")
            f.write(f"Dishes: {', '.join(loader.get_available_dishes())}\n")
            f.write(f"Methods: {', '.join(scaler.get_available_methods())}\n\n")
            
            f.write("Scaling Results for Palak Paneer:\n")
            f.write("-" * 40 + "\n")
            for ingredient, result in scaling_results.items():
                pred_2, pred_3 = result['predictions']
                actual_2, actual_3 = result['actual']
                f.write(f"{ingredient}: 2 servings = {pred_2:.1f}g, 3 servings = {pred_3:.1f}g\n")
        
        print(f"✓ Summary saved to: {summary_file}")
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        print(f"\nResults available in: {output_dir.absolute()}")
        print("\nTo run specific operations:")
        print("1. Scale single ingredient: python src/main.py --action scale --dish palak_paneer --ingredient Paneer")
        print("2. Scale full recipe: python src/main.py --action scale --dish palak_paneer")
        print("3. Evaluate methods: python src/main.py --action evaluate")
        print("4. Run tests: python test_system.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
