"""
Preprocessing module for quantity extraction and normalization.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class QuantityInfo:
    """Data class to store extracted quantity information."""
    numeric_value: float
    unit: str
    original_text: str
    confidence: float = 1.0


class QuantityPreprocessor:
    """Class for preprocessing and normalizing ingredient quantities."""
    
    def __init__(self):
        """Initialize the preprocessor with unit conversion tables."""
        self.weight_conversions = {
            'kg': 1000.0,
            'kilogram': 1000.0,
            'kilograms': 1000.0,
            'g': 1.0,
            'gram': 1.0,
            'grams': 1.0,
            'mg': 0.001,
            'milligram': 0.001,
            'milligrams': 0.001,
            'lb': 453.592,
            'pound': 453.592,
            'pounds': 453.592,
            'oz': 28.3495,
            'ounce': 28.3495,
            'ounces': 28.3495
        }
        
        self.volume_conversions = {
            'l': 1000.0,
            'liter': 1000.0,
            'liters': 1000.0,
            'ml': 1.0,
            'milliliter': 1.0,
            'milliliters': 1.0,
            'cup': 240.0,  # Approximation
            'cups': 240.0,
            'tbsp': 15.0,
            'tablespoon': 15.0,
            'tablespoons': 15.0,
            'tsp': 5.0,
            'teaspoon': 5.0,
            'teaspoons': 5.0
        }
        
        # Fraction mappings
        self.fraction_map = {
            '¼': 0.25, '½': 0.5, '¾': 0.75,
            '⅛': 0.125, '⅜': 0.375, '⅝': 0.625, '⅞': 0.875,
            '⅓': 0.333, '⅔': 0.667,
            '⅕': 0.2, '⅖': 0.4, '⅗': 0.6, '⅘': 0.8,
            '⅙': 0.167, '⅚': 0.833,
            '⅐': 0.143, '⅑': 0.111, '⅒': 0.1,
            '1/4': 0.25, '1/2': 0.5, '3/4': 0.75,
            '1/8': 0.125, '3/8': 0.375, '5/8': 0.625, '7/8': 0.875,
            '1/3': 0.333, '2/3': 0.667,
            '1/5': 0.2, '2/5': 0.4, '3/5': 0.6, '4/5': 0.8,
            '1/6': 0.167, '5/6': 0.833
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra spaces and standardizing format.
        
        Args:
            text (str): Input text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text or pd.isna(text):
            return ""
            
        # Convert to lowercase and strip
        text = str(text).lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Standardize common abbreviations
        text = text.replace('no.', 'nos.')
        text = text.replace('no', 'nos.')
        
        return text
    
    def convert_fractions_to_decimal(self, text: str) -> str:
        """
        Convert fraction notations to decimal values.
        
        Args:
            text (str): Text containing fractions
            
        Returns:
            str: Text with fractions converted to decimals
        """
        # Handle mixed numbers first (e.g., "2¼" -> "2.25")
        mixed_pattern = r'(\d+)([¼½¾⅛⅜⅝⅞⅓⅔⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒])'
        
        def replace_mixed(match):
            whole = float(match.group(1))
            fraction_char = match.group(2)
            fraction_value = self.fraction_map.get(fraction_char, 0)
            return str(whole + fraction_value)
        
        text = re.sub(mixed_pattern, replace_mixed, text)
        
        # Handle standalone fractions
        for fraction_str, decimal_value in self.fraction_map.items():
            text = text.replace(fraction_str, str(decimal_value))
        
        return text
    
    def extract_quantity_info(self, quantity_str: str) -> QuantityInfo:
        """
        Extract detailed quantity information from a quantity string.
        
        Args:
            quantity_str (str): Quantity string like "1¼ nos. / 80 grams"
            
        Returns:
            QuantityInfo: Extracted quantity information
        """
        original_text = quantity_str
        
        if not quantity_str or pd.isna(quantity_str):
            return QuantityInfo(0.0, "", original_text, 0.0)
        
        # Normalize and convert fractions
        normalized_text = self.normalize_text(quantity_str)
        normalized_text = self.convert_fractions_to_decimal(normalized_text)
        
        # Split by "/" to get parts
        parts = normalized_text.split('/')
        
        # Prefer the gram/weight part if available
        for part in reversed(parts):  # Start from the end (likely to contain grams)
            part = part.strip()
            
            # Try to extract weight first
            weight_result = self._extract_weight(part)
            if weight_result[0] > 0:
                return QuantityInfo(
                    weight_result[0], 
                    weight_result[1], 
                    original_text, 
                    weight_result[2]
                )
            
            # Try to extract volume
            volume_result = self._extract_volume(part)
            if volume_result[0] > 0:
                return QuantityInfo(
                    volume_result[0], 
                    volume_result[1], 
                    original_text, 
                    volume_result[2]
                )
        
        # If no units found, try to extract any numeric value
        numeric_match = re.search(r'(\d+(?:\.\d+)?)', normalized_text)
        if numeric_match:
            return QuantityInfo(
                float(numeric_match.group(1)), 
                "units", 
                original_text, 
                0.5
            )
        
        return QuantityInfo(0.0, "", original_text, 0.0)
    
    def _extract_weight(self, text: str) -> Tuple[float, str, float]:
        """
        Extract weight information from text.
        
        Args:
            text (str): Text to extract weight from
            
        Returns:
            Tuple[float, str, float]: (value_in_grams, unit, confidence)
        """
        # Pattern to match number followed by weight unit
        weight_pattern = r'(\d+(?:\.\d+)?)\s*(' + '|'.join(self.weight_conversions.keys()) + r')\b'
        
        match = re.search(weight_pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            conversion_factor = self.weight_conversions.get(unit, 1.0)
            return value * conversion_factor, unit, 1.0
        
        return 0.0, "", 0.0
    
    def _extract_volume(self, text: str) -> Tuple[float, str, float]:
        """
        Extract volume information from text.
        
        Args:
            text (str): Text to extract volume from
            
        Returns:
            Tuple[float, str, float]: (value_in_ml, unit, confidence)
        """
        # Pattern to match number followed by volume unit
        volume_pattern = r'(\d+(?:\.\d+)?)\s*(' + '|'.join(self.volume_conversions.keys()) + r')\b'
        
        match = re.search(volume_pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            conversion_factor = self.volume_conversions.get(unit, 1.0)
            return value * conversion_factor, unit, 0.8  # Lower confidence for volume
        
        return 0.0, "", 0.0
    
    def preprocess_recipe_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a recipe DataFrame by extracting and normalizing quantities.
        
        Args:
            df (pd.DataFrame): Input DataFrame with quantity_str column
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with additional quantity columns
        """
        df_processed = df.copy()
        
        # Extract quantity information for each row
        quantity_info_list = []
        for quantity_str in df_processed['quantity_str']:
            info = self.extract_quantity_info(quantity_str)
            quantity_info_list.append(info)
        
        # Add new columns
        df_processed['extracted_value'] = [info.numeric_value for info in quantity_info_list]
        df_processed['extracted_unit'] = [info.unit for info in quantity_info_list]
        df_processed['confidence'] = [info.confidence for info in quantity_info_list]
        
        # Use extracted value if confidence is high, otherwise use original quantity_grams
        df_processed['final_quantity'] = df_processed.apply(
            lambda row: row['extracted_value'] if row['confidence'] > 0.7 
            else row['quantity_grams'], axis=1
        )
        
        return df_processed
    
    def identify_scaling_categories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize ingredients based on their scaling behavior.
        
        Args:
            df (pd.DataFrame): Processed recipe DataFrame
            
        Returns:
            Dict[str, List[str]]: Categories of ingredients
        """
        categories = {
            'linear_scaling': [],      # Ingredients that scale linearly
            'minimal_scaling': [],     # Spices and seasonings that don't scale much
            'step_scaling': [],        # Ingredients that scale in steps
            'constant': []             # Ingredients that remain constant
        }
        
        # Group by dish and ingredient
        for dish in df['dish'].unique():
            dish_data = df[df['dish'] == dish]
            
            for ingredient in dish_data['ingredient'].unique():
                ingredient_data = dish_data[dish_data['ingredient'] == ingredient]
                
                if len(ingredient_data) < 2:
                    continue
                
                # Calculate scaling behavior
                serving_sizes = ingredient_data['serving_size'].values
                quantities = ingredient_data['final_quantity'].values
                
                # Calculate coefficient of variation
                if np.std(quantities) == 0:
                    categories['constant'].append(f"{dish}:{ingredient}")
                else:
                    cv = np.std(quantities) / np.mean(quantities)
                    
                    # Check if scaling is roughly linear
                    if len(serving_sizes) >= 3:
                        correlation = np.corrcoef(serving_sizes, quantities)[0, 1]
                        
                        if correlation > 0.9 and cv > 0.3:
                            categories['linear_scaling'].append(f"{dish}:{ingredient}")
                        elif cv < 0.2:
                            categories['minimal_scaling'].append(f"{dish}:{ingredient}")
                        else:
                            categories['step_scaling'].append(f"{dish}:{ingredient}")
                    else:
                        if cv > 0.3:
                            categories['linear_scaling'].append(f"{dish}:{ingredient}")
                        else:
                            categories['minimal_scaling'].append(f"{dish}:{ingredient}")
        
        return categories
    
    def clean_outliers(self, df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers based on z-score for each ingredient within each dish.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            z_threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        df_cleaned = df.copy()
        
        for dish in df_cleaned['dish'].unique():
            dish_mask = df_cleaned['dish'] == dish
            
            for ingredient in df_cleaned[dish_mask]['ingredient'].unique():
                ingredient_mask = (df_cleaned['dish'] == dish) & (df_cleaned['ingredient'] == ingredient)
                quantities = df_cleaned.loc[ingredient_mask, 'final_quantity']
                
                if len(quantities) > 2:  # Need at least 3 points to detect outliers
                    z_scores = np.abs((quantities - quantities.mean()) / quantities.std())
                    outlier_mask = z_scores > z_threshold
                    
                    # Mark outliers (but don't remove them yet, just flag)
                    df_cleaned.loc[ingredient_mask & outlier_mask, 'is_outlier'] = True
        
        # Add outlier column if it doesn't exist
        if 'is_outlier' not in df_cleaned.columns:
            df_cleaned['is_outlier'] = False
        
        return df_cleaned


if __name__ == "__main__":
    # Example usage
    preprocessor = QuantityPreprocessor()
    
    # Test quantity extraction
    test_quantities = [
        "1¼ nos. / 80 grams",
        "½ cup / 125 grams",
        "2 tbsp / 15 grams",
        "3 pinch / 0.25 grams"
    ]
    
    print("Testing quantity extraction:")
    for qty in test_quantities:
        info = preprocessor.extract_quantity_info(qty)
        print(f"{qty} -> {info.numeric_value} {info.unit} (confidence: {info.confidence})")
