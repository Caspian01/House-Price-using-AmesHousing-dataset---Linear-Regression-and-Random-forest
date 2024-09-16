import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

# Loading the database
file_path = 'AmesHousing.csv'
data_frame = pd.read_csv(file_path)

# Printing column names
# print(data_frame.head())
# print(data_frame.columns)

# Features used
x = data_frame[['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Built']]
y = data_frame['SalePrice']

# Data cleaning by handling empty values
imputer = SimpleImputer(strategy='median')
x = imputer.fit_transform(x)

# Splitting data for training and testing
training_for_x, test_x, training_for_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# Train linear regression model
linear_model = LinearRegression()
linear_model.fit(training_for_x, training_for_y)

# Making prediction based on trained linear regression model
linear_prediction = linear_model.predict(test_x)

# Evaluating linear regression model using the MAE, MSE, and R2
linear_mean_absolute_error = mean_absolute_error(test_y, linear_prediction)
linear_mean_squared_error = mean_squared_error(test_y, linear_prediction)
linear_r2 = r2_score(test_y, linear_prediction)

print("Linear Regression Model Evaluation")
print("============================================================================")
print(f"Mean Absolute Error: {linear_mean_absolute_error}")
print("============================================================================")
print(f"Mean Squared Error: {linear_mean_squared_error}")
print("============================================================================")
print(f"R-squared: {linear_r2}")

# Train Random Forest model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(training_for_x, training_for_y)

# Making prediction based on trained Random Forest model
random_forest_prediction = random_forest_model.predict(test_x)

# Evaluating Random Forest model using the MAE, MSE, and R2
random_forest_mean_absolute_error = mean_absolute_error(test_y, random_forest_prediction)
random_forest_mean_squared_error = mean_squared_error(test_y, random_forest_prediction)
random_forest_r2 = r2_score(test_y, random_forest_prediction)

print("\n ***************************************************************************")
print("Random Forest Model Evaluation")
print("============================================================================")
print(f"Mean Absolute Error: {random_forest_mean_absolute_error}")
print("============================================================================")
print(f"Mean Squared Error: {random_forest_mean_squared_error}")
print("============================================================================")
print(f"R-squared: {random_forest_r2}")

# Actual vs predicted House prices for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(test_y, linear_prediction, alpha=0.7, color='y')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red', linestyle='--')
plt.show()

# Residuals for Linear Regression
linear_residuals = test_y - linear_prediction
plt.figure(figsize=(10, 6))
sns.histplot(linear_residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution (Linear Regression)')
plt.show()

# Actual vs predicted House prices for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(test_y, random_forest_prediction, alpha=0.7, color='b')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Actual vs Predicted Prices (Random Forest)')
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red', linestyle='--')
plt.show()

# Residuals for Random Forest
random_forest_residuals = test_y - random_forest_prediction
plt.figure(figsize=(10, 6))
sns.histplot(random_forest_residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution (Random Forest)')
plt.show()

# # Extra / Saving model to be later used on other projects
joblib.dump(linear_model, 'Linear_house_price_model.pkl')
joblib.dump(random_forest_model, 'Random_forest_house_price_model.pkl')
print("Model was saved")

overall_qual = float(input("Overall Quality (1-10): "))
gr_liv_area = float(input("Above ground living area in square feet: "))
garage_cars = float(input("Number of cars that fit in the garage: "))
garage_area = float(input("Garage area in square feet: "))
total_bsmt_sf = float(input("Total basement area in square feet: "))
full_bath = float(input("Number of full bathrooms: "))
year_built = float(input("Year the house was built: "))

# Create a numpy array with the input values
features = np.array([[overall_qual, gr_liv_area, garage_cars, garage_area, total_bsmt_sf, full_bath, year_built]])

# Predict the house price
predicted_price = linear_model.predict(features)

print(f"\nThe predicted house price is: ${predicted_price[0]:,.2f}")