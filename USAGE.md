# Food AI System - Usage Guide

## Quick Start

Your virtual environment is set up! Here's how to use the system:

### 1. Activate Virtual Environment
```bash
cd /home/bakasur/Downloads/NOSH_TASK/food-ai-assignment
source .venv/bin/activate
```

### 2. Run Quick Demo
```bash
python run.py
```
OR
```bash
python quick_demo.py
```

### 3. Run Full System Tests
```bash
python test_system.py
```

## Individual Operations

### Scale a Single Ingredient
```bash
# Activate venv first
source .venv/bin/activate

# Scale paneer from 1&4 servings to predict 2&3 servings
python src/main.py --action scale \
  --dish "palak_paneer" \
  --ingredient "Paneer" \
  --train-sizes 1 4 \
  --target-sizes 2 3 \
  --method linear
```

### Scale Full Recipe
```bash
# Scale entire palak paneer recipe
python src/main.py --action scale \
  --dish "palak_paneer" \
  --train-sizes 1 4 \
  --target-sizes 2 3 \
  --method adaptive
```

### Evaluate Methods
```bash
# Compare all scaling methods
python src/main.py --action evaluate \
  --methods linear polynomial adaptive \
  --eval-type leave_one_out
```

## Available Dishes
- `palak_paneer` - Spinach with cottage cheese
- `shahi_paneer` - Royal cottage cheese curry  
- `matar_paneer` - Green peas with cottage cheese
- `paneer_masala` - Spiced cottage cheese

## Available Methods
- `linear` - Linear interpolation (best for main ingredients)
- `polynomial` - Polynomial regression (for complex scaling)
- `constant` - Constant scaling (for spices)
- `step` - Step scaling (for discrete ingredients)
- `adaptive` - Auto-selects best method (recommended)

## Python API Usage

```python
# Add to your Python script
import sys
sys.path.append('src')

from data_loader import load_recipe_data
from scaling_methods import IngredientScaler

# Load data
loader = load_recipe_data('data/paneer_recipes.json')
scaler = IngredientScaler()

# Get ingredient data
paneer_data = loader.get_ingredient_scaling_data('palak_paneer', 'Paneer')
serving_sizes = paneer_data['serving_size'].tolist()
quantities = paneer_data['quantity_grams'].tolist()

# Scale ingredient
result = scaler.scale_ingredient(
    serving_sizes=[1, 4],          # Training data
    quantities=[100, 400],         # Training quantities  
    target_sizes=[2, 3, 5],        # Target serving sizes
    method='adaptive'              # Scaling method
)

print(f"Predictions: {result.predicted_quantities}")
print(f"Confidence: {result.confidence_score}")
```

## Output Files

When you run the system, it creates:
- `output/` directory with results
- `demo_results.json` - Detailed scaling results
- `demo_summary.txt` - Human-readable summary

## Troubleshooting

1. **Import errors**: Make sure you activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. **Module not found**: Ensure you're in the right directory:
   ```bash
   cd /home/bakasur/Downloads/NOSH_TASK/food-ai-assignment
   ```

3. **Test failures**: Run the quick test:
   ```bash
   python test_system.py
   ```

## System Architecture

- `src/data_loader.py` - Data loading and parsing
- `src/preprocessing.py` - Quantity extraction and normalization  
- `src/scaling_methods.py` - All scaling algorithms
- `src/evaluation.py` - Evaluation metrics and cross-validation
- `src/main.py` - Command-line interface
- `tests/` - Unit tests for all components

## Performance Metrics

The system evaluates methods using:
- **MAE** (Mean Absolute Error) - Average prediction error
- **MAPE** (Mean Absolute Percentage Error) - Percentage error
- **RMSE** (Root Mean Squared Error) - Penalizes large errors
- **R²** (R-squared) - Explained variance (higher is better)

Perfect predictions would have MAE=0, MAPE=0%, RMSE=0, R²=1.0

## Next Steps

1. Try scaling your own recipes by adding data to `data/paneer_recipes.json`
2. Implement custom scaling methods by extending `BaseScalingMethod`
3. Add new evaluation metrics in `evaluation.py`
4. Create web interface using the included Streamlit/FastAPI dependencies
