"""
Data loader module for parsing JSON recipe data and extracting quantities.
"""

import json
import re
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np


class RecipeDataLoader:
    """Class to load and parse recipe data from JSON files."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the JSON file containing recipe data
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> Dict[str, Any]:
        """
        Load raw JSON data from file.
        
        Returns:
            Dict[str, Any]: Raw recipe data
        """
        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                self.raw_data = json.load(file)
            return self.raw_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {self.data_path}")
    
    def extract_numeric_quantity(self, quantity_str: str) -> float:
        """
        Extract numeric value from quantity string.
        
        Args:
            quantity_str (str): Quantity string like "1 no. / 85 grams"
            
        Returns:
            float: Extracted numeric value in grams
        """
        if not quantity_str or pd.isna(quantity_str):
            return 0.0
            
        # Handle fractions and mixed numbers
        quantity_str = self._handle_fractions(quantity_str)
        
        # Extract the gram value (after the "/")
        if '/' in quantity_str:
            gram_part = quantity_str.split('/')[-1].strip()
            # Extract numeric value from gram part
            gram_match = re.search(r'(\d+(?:\.\d+)?)', gram_part)
            if gram_match:
                return float(gram_match.group(1))
        
        # If no "/" found, try to extract any numeric value
        numeric_match = re.search(r'(\d+(?:\.\d+)?)', quantity_str)
        if numeric_match:
            return float(numeric_match.group(1))
            
        return 0.0
    
    def _handle_fractions(self, text: str) -> str:
        """
        Convert fraction notation to decimal.
        
        Args:
            text (str): Text containing fractions
            
        Returns:
            str: Text with fractions converted to decimals
        """
        # Handle mixed numbers like "1¼" -> 1.25
        mixed_pattern = r'(\d+)([¼½¾⅛⅜⅝⅞⅓⅔⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒])'
        
        fraction_map = {
            '¼': 0.25, '½': 0.5, '¾': 0.75,
            '⅛': 0.125, '⅜': 0.375, '⅝': 0.625, '⅞': 0.875,
            '⅓': 0.333, '⅔': 0.667,
            '⅕': 0.2, '⅖': 0.4, '⅗': 0.6, '⅘': 0.8,
            '⅙': 0.167, '⅚': 0.833,
            '⅐': 0.143, '⅑': 0.111, '⅒': 0.1
        }
        
        def replace_mixed(match):
            whole = int(match.group(1))
            fraction_char = match.group(2)
            fraction_value = fraction_map.get(fraction_char, 0)
            return str(whole + fraction_value)
        
        text = re.sub(mixed_pattern, replace_mixed, text)
        
        # Handle standalone fractions
        for fraction_char, value in fraction_map.items():
            text = text.replace(fraction_char, str(value))
            
        return text
    
    def create_structured_data(self) -> pd.DataFrame:
        """
        Create structured DataFrame from raw recipe data.
        
        Returns:
            pd.DataFrame: Structured data with columns [dish, serving_size, ingredient, quantity_grams]
        """
        if self.raw_data is None:
            self.load_data()
            
        structured_data = []
        
        for dish_name, servings_data in self.raw_data.items():
            for serving_size, ingredients in servings_data.items():
                for ingredient, quantity_str in ingredients.items():
                    quantity_grams = self.extract_numeric_quantity(quantity_str)
                    
                    structured_data.append({
                        'dish': dish_name,
                        'serving_size': int(serving_size),
                        'ingredient': ingredient,
                        'quantity_str': quantity_str,
                        'quantity_grams': quantity_grams
                    })
        
        self.processed_data = pd.DataFrame(structured_data)
        return self.processed_data
    
    def get_dish_data(self, dish_name: str) -> pd.DataFrame:
        """
        Get data for a specific dish.
        
        Args:
            dish_name (str): Name of the dish
            
        Returns:
            pd.DataFrame: Data for the specified dish
        """
        if self.processed_data is None:
            self.create_structured_data()
            
        return self.processed_data[self.processed_data['dish'] == dish_name].copy()
    
    def get_ingredient_scaling_data(self, dish_name: str, ingredient: str) -> pd.DataFrame:
        """
        Get scaling data for a specific ingredient in a dish.
        
        Args:
            dish_name (str): Name of the dish
            ingredient (str): Name of the ingredient
            
        Returns:
            pd.DataFrame: Data with serving_size and quantity_grams for the ingredient
        """
        dish_data = self.get_dish_data(dish_name)
        ingredient_data = dish_data[dish_data['ingredient'] == ingredient][
            ['serving_size', 'quantity_grams']
        ].sort_values('serving_size')
        
        return ingredient_data
    
    def get_available_dishes(self) -> List[str]:
        """
        Get list of available dish names.
        
        Returns:
            List[str]: List of dish names
        """
        if self.raw_data is None:
            self.load_data()
        return list(self.raw_data.keys())
    
    def get_ingredients_for_dish(self, dish_name: str) -> List[str]:
        """
        Get list of ingredients for a specific dish.
        
        Args:
            dish_name (str): Name of the dish
            
        Returns:
            List[str]: List of ingredient names
        """
        if self.processed_data is None:
            self.create_structured_data()
            
        dish_data = self.get_dish_data(dish_name)
        return dish_data['ingredient'].unique().tolist()


def load_recipe_data(data_path: str) -> RecipeDataLoader:
    """
    Convenience function to create and return a RecipeDataLoader instance.
    
    Args:
        data_path (str): Path to the JSON data file
        
    Returns:
        RecipeDataLoader: Configured data loader instance
    """
    loader = RecipeDataLoader(data_path)
    loader.load_data()
    loader.create_structured_data()
    return loader


if __name__ == "__main__":
    # Example usage
    loader = load_recipe_data("../data/paneer_recipes.json")
    
    print("Available dishes:", loader.get_available_dishes())
    print("\nSample data:")
    print(loader.processed_data.head(10))
    
    # Test specific ingredient scaling data
    print("\nPaneer scaling in palak_paneer:")
    paneer_data = loader.get_ingredient_scaling_data("palak_paneer", "Paneer")
    print(paneer_data)
