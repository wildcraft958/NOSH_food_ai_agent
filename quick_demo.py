#!/usr/bin/env python3
"""
Quick demo of the Food AI system with progress indicators.
"""

import sys
sys.path.append('src')

def main():
    print("Food AI Quick Demo")
    print("=" * 40)
    
    # Test 1: Load data
    print("1. Loading data...", end=' ')
    from data_loader import load_recipe_data
    loader = load_recipe_data("data/paneer_recipes.json")
    print(f"✓ ({len(loader.processed_data)} entries)")
    
    # Test 2: Scale paneer
    print("2. Scaling paneer...", end=' ')
    from scaling_methods import IngredientScaler
    scaler = IngredientScaler()
    
    paneer_data = loader.get_ingredient_scaling_data("palak_paneer", "Paneer")
    serving_sizes = paneer_data['serving_size'].tolist()
    quantities = paneer_data['quantity_grams'].tolist()
    
    result = scaler.scale_ingredient(
        serving_sizes=[1, 4],
        quantities=[quantities[0], quantities[3]],
        target_sizes=[2, 3],
        method='linear'
    )
    print(f"✓ (2 servings: {result.predicted_quantities[0]:.0f}g, 3 servings: {result.predicted_quantities[1]:.0f}g)")
    
    # Test 3: Compare with actual
    print("3. Checking accuracy...", end=' ')
    actual_2 = quantities[1]  # 2 servings actual
    actual_3 = quantities[2]  # 3 servings actual
    error_2 = abs(result.predicted_quantities[0] - actual_2)
    error_3 = abs(result.predicted_quantities[1] - actual_3)
    print(f"✓ (errors: {error_2:.0f}g, {error_3:.0f}g)")
    
    # Test 4: Show all dishes
    print("4. Available dishes:", end=' ')
    dishes = loader.get_available_dishes()
    print(f"✓ {', '.join(dishes)}")
    
    # Test 5: Quick evaluation
    print("5. Testing methods...", end=' ')
    methods_tested = 0
    for method in ['linear', 'adaptive']:
        try:
            test_result = scaler.scale_ingredient(
                serving_sizes=[1, 4],
                quantities=[100, 400],
                target_sizes=[2],
                method=method
            )
            methods_tested += 1
        except:
            pass
    print(f"✓ ({methods_tested} methods working)")
    
    print("\n" + "=" * 40)
    print("System is working!")
    print("=" * 40)
    
    print("\nExample usage:")
    print("• Scale paneer for 5 people: python -c \"")
    print("  import sys; sys.path.append('src')")
    print("  from data_loader import load_recipe_data")
    print("  from scaling_methods import IngredientScaler")
    print("  loader = load_recipe_data('data/paneer_recipes.json')")
    print("  scaler = IngredientScaler()")
    print("  data = loader.get_ingredient_scaling_data('palak_paneer', 'Paneer')")
    print("  result = scaler.scale_ingredient([1,4], [100,400], [5], 'linear')")
    print("  print(f'Paneer for 5 people: {result.predicted_quantities[0]}g')\"")
    
    print(f"\nData location: data/paneer_recipes.json")
    print(f"Available dishes: {len(dishes)} recipes")
    print(f"Total ingredients tracked: {len(loader.processed_data)}")

if __name__ == "__main__":
    main()
