# Time Series Demand Forecasting

## Project Overview

This project develops an end-to-end demand forecasting workflow using machine learning, statistical time-series forecasting, and deep learning.

The project contains two forecasting notebooks:

1. **Time Series Forecasting Notebook**

   * Linear Regression
   * Random Forest Regressor
   * Gradient Boosting Regressor
   * SARIMAX

2. **LSTM Forecasting Notebook**

   * Long Short-Term Memory (LSTM) neural network

The objective is to analyse historical demand patterns, engineer useful time-series features, build forecasting models, evaluate their performance, and compare the strengths and limitations of different forecasting approaches.

---

## Business Objective

Demand forecasting can help businesses improve:

* Inventory planning
* Stock replenishment
* Production planning
* Supply-chain planning
* Workforce planning
* Resource allocation
* Demand management

The project demonstrates how different forecasting techniques can be applied to historical demand data.

---

## Project Structure

```text
Time-Series-Forecasting/
│
├── notebooks/
│   ├── TimeSeriesForecasting.ipynb
│   └── TimeSeriesForecasting_LSTM.ipynb
│
├── visualizations/
│   ├── model_performance_comparison.png
│   ├── sarimax_actual_vs_forecast.png
│   ├── sarimax_residuals.png
│   ├── lstm_training_history.png
│   └── lstm_actual_vs_predicted.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset Preparation

The forecasting workflow includes:

* Dataset loading
* Data inspection
* Data cleaning
* Date conversion
* Chronological sorting
* Exploratory time-series analysis
* Time-based feature engineering
* Lag feature creation
* Rolling statistics
* Model training
* Forecast evaluation

Maintaining chronological order is important in time-series forecasting because future observations should not be used to predict earlier observations.

---

# Machine Learning Forecasting

## Machine Learning Models

Three machine-learning regression models are evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

### Linear Regression

Linear Regression is used as a baseline model.

It assumes a linear relationship between the input features and demand and provides an interpretable benchmark for comparison.

### Random Forest Regressor

Random Forest is an ensemble learning algorithm based on multiple decision trees.

It can capture nonlinear relationships and interactions between features.

### Gradient Boosting Regressor

Gradient Boosting builds an ensemble sequentially, with each new model attempting to reduce errors made by previous models.

It can capture nonlinear relationships between engineered features and demand.

---

## Machine Learning Train/Test Split

The three machine-learning models use the same defined dataset split:

```text
Training observations → 100,000
Testing observations  → 182,500
```

The three models are therefore directly comparable within this machine-learning experiment because they are evaluated on the same 182,500 test observations.

```text
Full ML Dataset
       │
       ▼
Train/Test Split
       │
       ├───────────────┐
       ▼               ▼
100,000 Training   182,500 Testing
       │               │
       ▼               ▼
 Model Training    Model Evaluation
```

---

## Feature Engineering

The machine-learning workflow uses time-related and historical-demand features.

### Time Features

Time-based information can be extracted from the date, including:

* Day
* Week
* Month
* Year
* Day of week

### Lag Features

Lag features represent previous demand observations.

Examples include:

* Lag 1
* Lag 7
* Lag 14
* Lag 30

### Rolling Statistics

Rolling statistics are used to represent recent demand behaviour and smooth short-term fluctuations.

These features provide machine-learning models with additional information about historical demand patterns.

---

## Model Evaluation Metrics

The machine-learning models are evaluated using:

### Mean Absolute Error — MAE

MAE represents the average absolute difference between actual and predicted values.

**Lower MAE is better.**

### Root Mean Squared Error — RMSE

RMSE measures the square root of the average squared prediction error.

RMSE gives greater importance to larger prediction errors.

**Lower RMSE is better.**

### R² Score

R² measures the proportion of variation in the target explained by the model.

**Higher R² is generally better.**

---

## Machine Learning Results

| Model             | Training Size | Test Size |        MAE |       RMSE |         R² |
| ----------------- | ------------: | --------: | ---------: | ---------: | ---------: |
| **Random Forest** |       100,000 |   182,500 | **7.4897** | **9.9512** | **0.9005** |
| Gradient Boosting |       100,000 |   182,500 |     9.1205 |    12.2287 |     0.8498 |
| Linear Regression |       100,000 |   182,500 |    23.6496 |    28.9262 |     0.1596 |

### Machine Learning Interpretation

Random Forest produced the strongest performance among the three machine-learning models.

It achieved:

* Lowest MAE
* Lowest RMSE
* Highest R²

Random Forest achieved an R² of approximately **0.9005**.

Gradient Boosting also performed strongly, with an R² of approximately **0.8498**.

Linear Regression produced substantially higher errors and a considerably lower R².

The results indicate that the nonlinear tree-based ensemble models captured the demand relationships more effectively than the linear baseline in this experiment.

---

## Machine Learning Visualizations

The notebook includes model-performance visualizations for:

* MAE
* RMSE
* R²

Example visualization:

```text
visualizations/model_performance_comparison.png
```

These visualizations make it easier to compare the three machine-learning models.

---

# Statistical Time-Series Forecasting

## Time-Series Analysis

The notebook also performs traditional time-series analysis.

The analysis includes:

* Time-series decomposition
* Stationarity analysis
* Augmented Dickey-Fuller (ADF) testing
* Autocorrelation Function (ACF)
* Partial Autocorrelation Function (PACF)

These analyses help identify trend, seasonality, stationarity, and temporal relationships.

---

## SARIMAX Forecasting

SARIMAX is evaluated separately from the machine-learning models.

### Important Data Scope

The SARIMAX experiment uses:

```text
Store = 1
Item  = 1
```

Therefore, SARIMAX represents the demand behaviour of **Store 1 + Item 1 only**.

It is not trained across the entire dataset used by the machine-learning models.

---

## SARIMAX Forecast Horizon

The SARIMAX model generates a:

```text
365-observation forecast
```

The forecast is evaluated against the corresponding 365 actual observations.

```text
Full Dataset
     │
     ▼
Store = 1
     │
     ▼
Item = 1
     │
     ▼
Store 1 + Item 1 Time Series
     │
     ▼
SARIMAX
     │
     ▼
365-Step Forecast
     │
     ▼
Evaluation
```

---

## SARIMAX Configuration

The SARIMAX model uses a seasonal period of 7 to represent weekly seasonality.

The model configuration used in the notebook is:

```python
SARIMAX(
    train_ts,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 7)
)
```

---

## SARIMAX Results

| Model       | Scope                | Test/Forecast Size |        MAE |       RMSE |          R² |
| ----------- | -------------------- | -----------------: | ---------: | ---------: | ----------: |
| **SARIMAX** | **Store 1 + Item 1** |            **365** | **5.8691** | **7.4026** | **-0.1365** |

### SARIMAX Interpretation

SARIMAX achieved:

* MAE: **5.8691**
* RMSE: **7.4026**
* R²: **-0.1365**

The MAE and RMSE are lower than those obtained by the three machine-learning models.

However, the SARIMAX R² is negative.

A negative R² indicates that the predictions performed worse than the mean-based baseline according to the R² metric for this particular test period.

Therefore, SARIMAX should not automatically be considered the overall best model.

Its evaluation scope is different from that of the machine-learning models.

---

## SARIMAX Visualizations

### Actual vs Forecast

```text
visualizations/sarimax_actual_vs_forecast.png
```

This visualization compares actual Store 1 + Item 1 demand with the SARIMAX forecast.

### Residual Analysis

```text
visualizations/sarimax_residuals.png
```

Residual analysis helps investigate the differences between actual and forecast demand.

Residual:

```text
Residual = Actual − Forecast
```

---

# LSTM Deep Learning

## LSTM Forecasting Notebook

A separate notebook extends the project using a Long Short-Term Memory neural network.

Notebook:

```text
notebooks/TimeSeriesForecasting_LSTM.ipynb
```

The LSTM model uses a **30-day look-back window** to learn temporal dependencies from historical observations.

```text
Historical Demand
       │
       ▼
30-Day Look-Back Window
       │
       ▼
LSTM Layer
       │
       ▼
Dense Output
       │
       ▼
Demand Forecast
```

---

## LSTM Training

The LSTM training workflow includes:

* Sequence generation
* Feature scaling
* 30-day look-back window
* LSTM layer
* Dense output layer
* Adam optimizer
* Mean Squared Error loss
* Mean Absolute Error metric
* Early stopping

The LSTM notebook uses an early-stopping callback to help prevent unnecessary training once validation performance stops improving.

---

## LSTM Results

The LSTM notebook achieved:

| Model |     MAE |    RMSE |     R² |
| ----- | ------: | ------: | -----: |
| LSTM  | 21.5480 | 26.0865 | 0.0568 |

The current LSTM configuration provides a deep-learning benchmark.

Its R² of **0.0568** indicates that the current configuration explains only a relatively small proportion of the variation in the test data.

Further optimization may therefore be required.

---

# Overall Model Results

The current results across the project are:

| Model             | Scope             | Test Size |        MAE |       RMSE |         R² |
| ----------------- | ----------------- | --------: | ---------: | ---------: | ---------: |
| **SARIMAX**       | Store 1 + Item 1  |       365 | **5.8691** | **7.4026** |    -0.1365 |
| **Random Forest** | ML dataset        |   182,500 |     7.4897 |     9.9512 | **0.9005** |
| Gradient Boosting | ML dataset        |   182,500 |     9.1205 |    12.2287 |     0.8498 |
| Linear Regression | ML dataset        |   182,500 |    23.6496 |    28.9262 |     0.1596 |
| LSTM              | Separate notebook |         — |    21.5480 |    26.0865 |     0.0568 |

## Important Comparison Note

These models were **not evaluated under identical conditions**.

The machine-learning models use:

```text
100,000 training observations
182,500 testing observations
```

SARIMAX uses:

```text
Store 1 + Item 1
365 forecast/test observations
```

LSTM is evaluated in a separate notebook using its own sequence-based setup.

Therefore, the results should not be interpreted as a perfectly controlled head-to-head comparison.

The strongest direct comparison is between:

* Linear Regression
* Random Forest
* Gradient Boosting

because these three models use the same machine-learning test dataset.

---

# Key Findings

### Machine Learning

**Random Forest** is the strongest machine-learning model in this experiment.

It achieved:

```text
MAE  = 7.4897
RMSE = 9.9512
R²   = 0.9005
```

### Gradient Boosting

Gradient Boosting also produced strong performance:

```text
MAE  = 9.1205
RMSE = 12.2287
R²   = 0.8498
```

### Linear Regression

Linear Regression provides a baseline:

```text
MAE  = 23.6496
RMSE = 28.9262
R²   = 0.1596
```

### SARIMAX

SARIMAX produced the lowest MAE and RMSE within its Store 1 + Item 1 experiment:

```text
MAE  = 5.8691
RMSE = 7.4026
R²   = -0.1365
```

### LSTM

The current LSTM configuration achieved:

```text
MAE  = 21.5480
RMSE = 26.0865
R²   = 0.0568
```

---

# Limitations

## Different Evaluation Setups

The models use different data scopes and forecasting setups.

Therefore, their metrics cannot be interpreted as a fully controlled comparison.

## SARIMAX Scope

SARIMAX is evaluated only for:

```text
Store 1 + Item 1
```

Its results should not automatically be generalized to all stores and items.

## LSTM Scope

The LSTM model is implemented and evaluated in a separate notebook.

Its sequence-based evaluation setup differs from the machine-learning experiment.

## Current Model Performance

The LSTM and SARIMAX R² values indicate that additional investigation and optimization may be useful.

---

# Future Improvements

Potential improvements include:

* Hyperparameter tuning
* Time-series cross-validation
* Testing different forecast horizons
* Additional lag features
* Additional rolling-window features
* SARIMAX parameter optimization
* Testing different seasonal periods
* LSTM hyperparameter optimization
* Testing different LSTM architectures
* Additional external features
* Residual diagnostics
* Forecast confidence intervals
* Multi-store forecasting
* Multi-item forecasting
* Store-item specific forecasting

A future controlled experiment could evaluate all models using the same store-item combination, forecasting horizon, and test period.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Statsmodels
* TensorFlow
* Jupyter Notebook

---

# Project Skills Demonstrated

* Python
* Pandas
* NumPy
* Data Cleaning
* Exploratory Data Analysis
* Time-Series Analysis
* Feature Engineering
* Lag Features
* Rolling Statistics
* Time-Series Decomposition
* Stationarity Testing
* ADF Test
* ACF
* PACF
* Linear Regression
* Random Forest
* Gradient Boosting
* SARIMAX
* LSTM
* Deep Learning
* MAE
* RMSE
* R²
* Data Visualization
* Model Evaluation
* Model Comparison
* Business Interpretation

---

# How to Run the Project

## 1. Clone the repository

```bash
git clone <your-repository-url>
```

## 2. Navigate to the project

```bash
cd Time-Series-Forecasting
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the notebooks

Start with:

```text
notebooks/TimeSeriesForecasting.ipynb
```

Then review:

```text
notebooks/TimeSeriesForecasting_LSTM.ipynb
```

---

# Conclusion

This project demonstrates an end-to-end demand forecasting workflow using machine learning, statistical time-series forecasting, and deep learning.

The machine-learning experiment compares Linear Regression, Random Forest, and Gradient Boosting using the same 100,000-observation training dataset and 182,500-observation test dataset.

Among these models, Random Forest produced the strongest performance.

The statistical forecasting experiment uses SARIMAX specifically for Store 1 + Item 1 and generates a 365-observation forecast.

A separate LSTM notebook extends the project into deep learning using a 30-day look-back sequence.

The project demonstrates the importance of:

* Appropriate feature engineering
* Temporal data preparation
* Model selection
* Forecast evaluation
* Visualization
* Understanding evaluation scope
* Interpreting model results in their correct context

The results also show why model performance should not be judged using a single metric or by comparing models trained and evaluated under fundamentally different conditions.

---

# Author

**Murugavel Gnanasekaran**

Mechanical Engineer | Aspiring Data Scientist

Advanced Certification in Data Science & AI

