# Linear Feature Engineering Project 1

This repo contains code for Project 1 of Quantitative Foundations, focusing on linear regression and feature engineering.

## Main Files

- **polynomial.py**: Implements polynomial feature engineering with cross-validation.
  - Command: `python polynomial.py`
  - Output: `predictions.txt`, `report.txt`, error plots (`holdout_R_vs_p_K{}.png`), and 1D fit plots (`fit_1d_holdout_K{}.png`).

- **piecewise.py**: Implements piecewise polynomial regression with ensemble methods.
  - Command: `python piecewise.py` (default) or 
   You can specify the argument; e.g, `python piecewise.py --n_models 3 --n_pieces 6 --alpha 0.5 --poly_degree 3 --min_samples_leaf 15 --k_folds 5`.
  - Output: `predictions.txt`, console output with training/test errors.

## Usage
- Place `traindata.txt` (926 rows × 9 cols) and `testinputs.txt` (103 rows × 8 cols) in the directory as the code to be executed.
- Install dependencies: `pip install numpy matplotlib scikit-learn`.
- Run scripts to generate `predictions.txt` with 103 predicted values.
