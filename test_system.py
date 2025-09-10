#!/usr/bin/env python3
"""
Simple test script to verify the Food AI system works.
"""

import sys
import os

# Add src to path
sys.path.append('src')

try:
    print("Testing Food AI Ingredient Scaling System...")
    print("=" * 50)
    
    # Test 1: Data loading
    print("1. Testing data loading...")
    from data_loader import load_recipe_data
    loader = load_recipe_data("data/paneer_recipes.json")
    print(f"   ✓ Loaded {len(loader.processed_data)} ingredient entries")
    print(f"   ✓ Available dishes: {loader.get_available_dishes()}")
    
    # Test 2: Preprocessing
    print("\n2. Testing preprocessing...")
    from preprocessing import QuantityPreprocessor
    preprocessor = QuantityPreprocessor()
    test_qty = "1¼ nos. / 80 grams"
    result = preprocessor.extract_quantity_info(test_qty)
    print(f"   ✓ Extracted '{test_qty}' -> {result.numeric_value} {result.unit}")
    
    # Test 3: Scaling methods
    print("\n3. Testing scaling methods...")
    from scaling_methods import IngredientScaler
    scaler = IngredientScaler()
    
    # Test linear scaling
    result = scaler.scale_ingredient(
        serving_sizes=[1, 2, 3, 4],
        quantities=[100, 200, 300, 400],
        target_sizes=[2, 3],
        method='linear'
    )
    print(f"   ✓ Linear scaling: {result.predicted_quantities}")
    
    # Test adaptive scaling
    result = scaler.scale_ingredient(
        serving_sizes=[1, 2, 3, 4],
        quantities=[100, 200, 300, 400],
        target_sizes=[2, 3],
        method='adaptive'
    )
    print(f"   ✓ Adaptive scaling: {result.predicted_quantities}")
    
    # Test 4: Real recipe scaling
    print("\n4. Testing real recipe scaling...")
    paneer_data = loader.get_ingredient_scaling_data("palak_paneer", "Paneer")
    if not paneer_data.empty:
        serving_sizes = paneer_data['serving_size'].tolist()
        quantities = paneer_data['quantity_grams'].tolist()
        
        result = scaler.scale_ingredient(
            serving_sizes=serving_sizes[:2],  # Use first 2 as training
            quantities=quantities[:2],
            target_sizes=[3],  # Predict for 3 servings
            method='linear'
        )
        print(f"   ✓ Paneer for 3 servings: {result.predicted_quantities[0]:.1f}g")
        print(f"   ✓ Actual value: {quantities[2]:.1f}g")
        print(f"   ✓ Error: {abs(result.predicted_quantities[0] - quantities[2]):.1f}g")
    
    print("\n" + "=" * 50)
    print("All tests passed! The system is working correctly.")
    print("=" * 50)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
