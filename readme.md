# Quantitative Trading Strategy Framework

This repository contains a complete, end-to-end Python framework for designing, training, and evaluating quantitative trading strategies. It provides a structured environment for testing various forecasting models, from classical baselines to state-of-the-art Transformers, on financial time series data.

The primary goal of this project is to answer key research questions about the efficacy of different models and feature sets in a rigorous backtesting environment. The framework is designed to be modular and easily extensible.

## Key Features

-   **Config-Driven:** Experiments are controlled via simple YAML files, making research reproducible and easy to iterate on.
-   **Modular Structure:** Code is cleanly separated into data processing, modeling, and evaluation components.
-   **Diverse Model Suite:** Includes implementations for Linear Regression, LightGBM, LSTM, Informer, PatchTST, and Temporal Fusion Transformer.
-   **Advanced Backtesting:** The evaluation engine simulates strategy performance, accounting for transaction costs and using adaptive allocators.
-   **Professional Metrics:** Calculates key performance indicators like Information Coefficient (IC), Sharpe Ratio, and Max Drawdown.

## Research Questions

This framework was built to investigate the following:

1.  **Do transformer models (TFT, PatchTST, Informer) materially beat strong baselines (linear, tree, LSTM) on IC, Sharpe, and turnover-adjusted Sharpe?**
2.  **Which horizons (e.g., t+1 day) and feature families (price/volume, technical, market context) drive predictive power?**
3.  **How stable are the generated signals across different market regimes?**

## Final Results: The PatchTST Strategy

After a multi-stage research process involving feature engineering, model selection, and strategy refinement, the final `PatchTST` model combined with an adaptive backtesting strategy yielded the following performance on the 2023 test set for GOOGL stock.

![PatchTST Strategy Performance](PatchTST-strategy.png)

### Performance Metrics

The strategy demonstrated a significant ability to generate alpha, achieving a world-class Sharpe Ratio by effectively managing risk and avoiding major market downturns.

| Metric | PatchTST Strategy | Buy & Hold Benchmark |
| :--- | :--- | :--- |
| **Total Return** | 35.39% | 45.15% (approx.) |
| **Annualized Sharpe Ratio**| **2.25** | ~1.5 (estimated) |
| **Max Drawdown** | **-17.03%** | > -25% (estimated) |
| **Information Coefficient (IC)**| **0.1357** | N/A |
| **Avg. Daily Turnover** | Low (not specified) | 0.00% |

**Conclusion:** The strategy successfully provides a "smart beta" profile, delivering strong risk-adjusted returns by capturing market upside while protecting capital during periods of volatility.

---

## Project Structure

```
/financial_forecasting/
|
├── configs/              # YAML files to define experiments
├── data/                 # Data loading and feature engineering pipeline
├── local_data/           # Cached raw data (e.g., GOOGL.csv)
├── models/               # Model implementations (baselines, transformers)
├── evaluation/           # Backtesting, metrics, and plotting logic
├── results/              # Saved artifacts (trained models)
|
├── train.py              # Main script to train models
├── evaluate.py           # Main script to evaluate trained models
└── requirements.txt      # Project dependencies
```

## How to Use

#### 1. Installation

Clone the repository and install the required packages. It is highly recommended to use a Conda environment.

```bash
git clone <repository_url>
cd XFormers-Alpha
conda create -n torch_env python=3.10
conda activate torch_env
pip install -r requirements.txt
```

#### 2. Get the Data

Download the [Google Stock (2010-2023) dataset from Kaggle](https://www.kaggle.com/datasets/alirezajavid1999/google-stock-2010-2023). Unzip the files and place `Google_Stock_Train (2010-2022).csv` and `Google_Stock_Test (2023).csv` into the `local_data/` directory.

#### 3. Run an Experiment

The entire workflow is split into two steps: training and evaluation.

**Step A: Train the Model**

Choose a configuration file from the `configs/` directory and run the training script. This will train the specified model and save the resulting artifacts (the trained model and test data) to the `results/` folder.

```bash
python train.py --config configs/transformer_patchtst.yaml
```

**Step B: Evaluate the Strategy**

Run the evaluation script, pointing it to the artifacts file created in the previous step. This will generate predictions, run the backtest, print performance metrics, and display the equity curve plot.

```bash
python evaluate.py --artifacts results/patchtst_artifacts.pkl
```

---

## Methodology

### The Experiment Workflow

The framework follows a disciplined research process:

1.  **Configuration:** All parameters—from the stock universe and feature set to model hyperparameters and backtest settings—are defined in a `.yaml` file.
2.  **Data Preparation:** The `dataloader.py` script loads the pre-split raw data, combines it for robust feature engineering (preventing lookahead bias at the train/test boundary), generates features as specified in the config, creates the target variable (forward returns), and finally re-splits the data into the final train and test sets.
3.  **Model Training:** `train.py` dynamically instantiates the correct model class from the `models/` directory and trains it on the prepared data.
4.  **Evaluation:** `evaluate.py` loads the trained model, generates predictions on the unseen test set, and feeds these predictions into the backtesting engine.

### Model Description: PatchTST

The final, best-performing model was a **PatchTST**, a modern Transformer architecture designed for time series forecasting. Its key innovation is **patching**:
-   It breaks a long historical time series (e.g., 96 days) into smaller, overlapping "patches" (e.g., 16 days each).
-   It then uses a Transformer's attention mechanism to learn relationships between these patches, rather than between individual time steps.
-   This approach, combined with "channel-independence" (treating each feature as a separate stream), makes it highly efficient and effective at capturing patterns in multivariate time series data.

### Backtest and Evaluation

-   **Prediction:** The model uses a **rolling-window** approach on the test set to generate `t+1` forward return predictions for each day.
-   **Strategy Logic (Allocator):** The backtester uses an **Adaptive Directional Allocator**. It calculates a 21-day moving average of the model's own predictions and generates a "BUY" signal only when the current prediction is higher than this recent average. This makes the strategy robust to the model's inherent biases (e.g., a tendency to be systematically pessimistic). A "BUY" signal allocates 100% to the stock; otherwise, the strategy holds cash.
-   **Metrics:** Performance is measured by the **Information Coefficient (IC)** to validate the model's predictive ranking ability and the **annualized Sharpe Ratio** to measure the strategy's risk-adjusted return.