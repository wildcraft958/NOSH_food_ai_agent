# Food AI Ingredient Scaling System

This project provides a Python-based system for scaling recipe ingredient quantities to different serving sizes. It includes multiple scaling algorithms and a comprehensive evaluation framework to test their performance.

This system was developed to fulfill the requirements of the **Food AI Intern Assignment**.

## Project Structure

```
.
├── data/
│   └── paneer_recipes.json   # Input data file
├── output/
│   ├── evaluation_results.csv  # Aggregated performance metrics
│   └── method_comparison.png   # Visual comparison of scaling methods
├── src/
│   ├── data_loader.py        # Loads and parses recipe data
│   ├── evaluation.py         # Evaluation framework and metrics
│   ├── scaling_methods.py    # All scaling algorithms (Linear, Polynomial, etc.)
│   └── ...
├── evaluation_runner.py      # Main script to run the full evaluation
├── quick_demo.py             # A simple script to verify system functionality
├── REPORT.md                 # Detailed report on methods and findings
└── README.md                 # This file
```

## Features

- **Multiple Scaling Methods:** Implements five different algorithms:
  1.  `linear`: Linear interpolation.
  2.  `polynomial`: 2nd-degree polynomial regression.
  3.  `constant`: Assumes quantity does not scale.
  4.  `step`: Uses the quantity of the nearest known serving size.
  5.  `adaptive`: A meta-algorithm that automatically selects the best method from the above.
- **Comprehensive Evaluation:** A rigorous evaluation script (`evaluation_runner.py`) that tests all methods across all dishes using the specified cross-validation strategy.
- **Rich Metrics:** Calculates six different metrics (MAE, MAPE, RMSE, R², Median AE, Max AE) to provide a holistic view of performance.
- **Automated Reporting:** Generates a CSV report and a visual plot comparing the performance of all methods.

## How to Run the Evaluation

To run the full evaluation process as required by the assignment, execute the `evaluation_runner.py` script.

```bash
python3 evaluation_runner.py
```

This will:
1.  Load the `paneer_recipes.json` data.
2.  Iterate through all dishes and all combinations of two training serving sizes.
3.  Evaluate all five scaling methods by predicting the remaining serving sizes.
4.  Aggregate the results.
5.  Save a summary report to `output/evaluation_results.csv`.
6.  Save a comparison bar chart to `output/method_comparison.png`.

## Quick Demo

To quickly check if the system is set up correctly, you can run the `quick_demo.py` script:

```bash
python3 quick_demo.py
```

This script performs a few basic checks, such as loading data and running a simple scaling prediction.

## Conclusion from Analysis

The final analysis, detailed in `REPORT.md`, shows that the **`linear` scaling method is the most accurate and reliable** for this dataset, outperforming all other methods by a significant margin.

For a detailed breakdown of the methods, metrics, and results, please see [REPORT.md](REPORT.md).
