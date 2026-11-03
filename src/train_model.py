import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create Dummy Data (Age, Salary -> Numerical Target, e.g., Max Credit Limit)
# In a real scenario, you would load this via pd.read_csv("data.csv")
np.random.seed(42)
n_samples = 100

data = {
    'age': np.random.randint(18, 65, size=n_samples),
    'salary': np.random.randint(30000, 150000, size=n_samples)
}
df = pd.DataFrame(data)

# Let's create a realistic target variable based on age and salary with some noise
df['credit_limit'] = (df['age'] * 200) + (df['salary'] * 0.4) + np.random.normal(0, 2000, n_samples)

# 2. Separate Features (X) and Target (y)
X = df[['age', 'salary']]
y = df['credit_limit']

# 3. Split into Training and Testing Sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the Model Performance
y_pred = model.predict(X_test)
print(# Coefficients: [Age_Coef, Salary_Coef]
f"Model Coefficients: {model.coef_}") 
print(f"R² Score (Accuracy): {r2_score(y_test, y_pred):.2f}")

# 6. Save the trained model using joblib
model_filename = 'models/regression_model.joblib'
joblib.dump(model, model_filename)
print(f"Model successfully saved to '{model_filename}'")
