# **Inflation Rate Classification Based on Economic Indicators**

---

# Overview  
This notebook focuses on classifying inflation regimes using macroeconomic indicators. It preprocesses features like CPI, PPI, and unemployment rates, performs feature engineering (including differencing and percentage change), and applies machine learning models to classify inflation as "Low", "Moderate", or "High". The goal is to assess and compare various classification models for their ability to capture economic patterns effectively.

---

# Dataset  
- **Path/URL:** `/content/Macroeconomic_Inflation_Labels.csv`  
- **Target column:** `InflationLabel`  
- **Feature column(s):**  
  - `CPI`, `PPI`, `UnempRate` (raw indicators)  
  - `CPI_yoy`, `PPI_yoy`, `UnempRate_change` (engineered indicators)  
- **Feature count/types:** Not explicitly printed; includes both raw index values and derived percentage change features.

---

# Features & Preprocessing  
- **Missing value handling:** `.dropna()` applied after feature engineering  
- **Feature Engineering:**  
  - `CPI_yoy = CPI.pct_change(12) * 100`  
  - `PPI_yoy = PPI.pct_change(12) * 100`  
  - `UnempRate_change = UnempRate.diff()`  
- **Target Transformation:**  
  - `InflationLabel` created by binning `InflationRate` into:  
    - Low: `InflationRate < 0.5`  
    - Moderate: `0.5 <= InflationRate < 3.0`  
    - High: `InflationRate >= 3.0`

---

# Models  
The following models were used with default or specified hyperparameters:

- **Support Vector Machine (SVM)**  
  - `SVC(kernel='linear')`
  
- **Neural Network (MLPClassifier)**  
  - `MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)`
  
- **Random Forest**  
  - `RandomForestClassifier()`
  
- **Logistic Regression**  
  - `LogisticRegression(max_iter=1000)`

---

# Evaluation  

### Metrics Used  
- `accuracy_score`  
- `precision_score`  
- `recall_score`  
- `f1_score`  
- `classification_report`  
- `cross_val_score` (CV accuracy and F1 with mean and std)

### Visualizations  
- Confusion matrices using `ConfusionMatrixDisplay`  
- Model comparison bar plots (accuracy, F1, etc.)  
- Correlation heatmaps for EDA  
- Time series and scatter plots of CPI, PPI, and UnempRate

### Statistical Tests  
- **t-test** (paired):  
  Compared the outperforming model (Random Forest) with others using cross-validated accuracy and F1-score  
  - Example equation:  
    ```latex
    t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{2}{n}}}
    ```  
- **Wilcoxon Signed-Rank Test:**  
  Non-parametric alternative to the t-test to compare paired differences  
  - Example equation:  
    ```latex
    W = \sum_{i=1}^n R_i \cdot \text{sign}(d_i)
    ```

---

# Environment & Requirements  

### Libraries Used  
- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `scipy`  

### Install example  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```
