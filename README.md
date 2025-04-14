# ft_linear_regression

An educational project to implement a **simple linear regression** model from scratch — with **no external libraries** — to predict the **price of a car based on its mileage**.

---

##  Objective

This project introduces you to machine learning by implementing:

- A **gradient descent** algorithm from scratch.
- A **training program** to learn parameters `θ₀` and `θ₁`.
- A **normalization step** to improve training efficiency.
- An evaluation of **model performance** using MSE and R².

---

##  Theoretical Foundation

We model the relation between mileage and price with a **linear hypothesis function**:


## 🛠️ Implementation Details

- Language: **Python**
- **NO** external libraries like sklearn has been used .
- Manual implementation of:
  - CSV reading
  - Normalization
  - Gradient descent
  - Error functions (MSE and R²)

---

## 🚀 Usage

To train the model and save the results:

```bash
python train.py
```

### Gradient Descent Update Rules

To minimize the **Mean Squared Error (MSE)** loss function, we apply gradient descent:

θ₀ := θ₀ - α × (1/m) × Σ(estimated_price(xᵢ) - yᵢ) θ₁ := θ₁ - α × (1/m) × Σ(estimated_price(xᵢ) - yᵢ) × xᵢ

Where:
- `α` is the learning rate.
- `m` is the number of samples.
- `xᵢ` is mileage.
- `yᵢ` is the actual price.

---

## 🛠️ Implementation Details

- Language: **Python**
- **NO** external libraries like NumPy, pandas, or sklearn are used.
- Manual implementation of:
  - CSV reading
  - Normalization
  - Gradient descent
  - Error functions (MSE and R²)

---


✅ Evaluation Metrics
Metric	Description
MSE	Measures the average squared difference between estimated and true values
R²	Coefficient of determination indicating goodness of fit
🚫 What Was Not Used
❌ scikit-learn any machine learning libraries

✅ All logic and formulas are implemented manually
