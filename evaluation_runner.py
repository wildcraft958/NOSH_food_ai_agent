
import itertools
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src is in the Python path
import sys
sys.path.append('src')

from data_loader import load_recipe_data
from scaling_methods import IngredientScaler
from evaluation import CrossValidationEvaluator, EvaluationVisualizer

# --- Configuration ---
DATA_FILE = "data/paneer_recipes.json"
OUTPUT_DIR = Path("output")
RESULTS_CSV = OUTPUT_DIR / "evaluation_results.csv"
PLOT_FILE = OUTPUT_DIR / "method_comparison.png"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

def run_full_evaluation():
    """
    Runs the definitive evaluation as per the assignment instructions.
    - For each dish, it iterates through all combinations of 2 serving sizes for training.
    - It predicts the remaining 2 serving sizes.
    - It aggregates the results and compares all scaling methods.
    """
    print("Starting full evaluation...")

    # 1. Load Data
    print(f"Loading data from {DATA_FILE}...")
    try:
        loader = load_recipe_data(DATA_FILE)
        dishes = loader.get_available_dishes()
        all_serving_sizes = sorted(loader.processed_data['serving_size'].unique())
    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{DATA_FILE}'. Please ensure it exists.")
        return

    # 2. Initialize Tools
    scaler = IngredientScaler()
    evaluator = CrossValidationEvaluator(loader, scaler)
    methods_to_compare = scaler.get_available_methods()
    
    print(f"Dishes to evaluate: {dishes}")
    print(f"Methods to compare: {methods_to_compare}")
    print(f"Serving sizes available: {all_serving_sizes}")

    # 3. Run Evaluation Loop
    # This will store results from all folds to be aggregated later
    all_results = {method: [] for method in methods_to_compare}
    
    # Generate all combinations of 2 training sizes from the available sizes
    training_combinations = list(itertools.combinations(all_serving_sizes, 2))
    print(f"\nRunning evaluation for {len(dishes)} dishes and {len(training_combinations)} training combinations...")

    for i, (train_sizes) in enumerate(training_combinations):
        test_sizes = [s for s in all_serving_sizes if s not in train_sizes]
        print(f"  [{i+1}/{len(training_combinations)}] Training on: {train_sizes}, Testing on: {test_sizes}")

        for method in methods_to_compare:
            for dish in dishes:
                # Perform evaluation for this specific fold (dish, train/test split)
                fold_result = evaluator.k_fold_evaluation(
                    dish_name=dish,
                    training_sizes=list(train_sizes),
                    test_sizes=test_sizes,
                    method=method
                )
                if fold_result and fold_result['overall_metrics']:
                    all_results[method].append(fold_result)

    # 4. Aggregate and Report Results
    print("\nAggregating results...")
    final_report = evaluator.generate_evaluation_report_from_folds(all_results)
    
    # Save the report to CSV
    final_report.to_csv(RESULTS_CSV, index=False)
    print(f"Evaluation report saved to '{RESULTS_CSV}'")
    print("\n--- Evaluation Summary ---")
    print(final_report.to_string())
    print("------------------------\n")

    # 5. Generate and Save Visualizations
    print("Generating visualizations...")
    # We need to re-format the aggregated data for the visualizer
    method_eval_objects = evaluator.create_method_evaluation_from_folds(all_results)
    
    if method_eval_objects:
        fig = EvaluationVisualizer.plot_method_comparison(method_eval_objects)
        fig.suptitle("Overall Performance of Scaling Methods", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(PLOT_FILE)
        print(f"Comparison plot saved to '{PLOT_FILE}'")
        plt.close(fig)
    else:
        print("No results to visualize.")

    print("\nFull evaluation complete!")

if __name__ == "__main__":
    run_full_evaluation()
