# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# 1. Load the California Housing dataset
data = fetch_california_housing()

# Convert the dataset into a DataFrame for easier handling
df = pd.DataFrame(data=data.data, columns=data.feature_names)
print(df.head())
df['Target'] = data.target  # Add the target column (house prices)

# 2. Select 'MedInc' (Median Income) as the feature and 'Target' (House Price) as the target variable
X = df[['MedInc']]  # Feature (Median Income)
y = df['Target']    # Target (House Price)

# 3. Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict the target variable for test set
y_pred = model.predict(X_test)

# 6. Evaluate the model's performance using MSE and R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# 7. Plot the regression line with the data points
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.title('Linear Regression: Median Income vs. House Price')
plt.xlabel('Median Income (MedInc)')
plt.ylabel('House Price (Target)')
plt.legend()
plt.grid(True)
plt.show()