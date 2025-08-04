🏠 House Price Prediction with Machine Learning (Sklearn)

This project demonstrates how to build a House Price Prediction model using Scikit-learn (Sklearn). It uses a dataset containing housing features like the number of bedrooms, bathrooms, square footage, etc., to predict the price of a house.

📁 Dataset

The dataset used should contain columns such as:

price: Target variable (house price)

bedrooms, bathrooms, sqft_living, sqft_lot, floors, etc.

Can be any CSV dataset from sources like Kaggle, UCI, or custom data.

📦 Libraries Used

pandas
numpy
scikit-learn
matplotlib
seaborn

Install via pip:

pip install pandas numpy scikit-learn matplotlib seaborn

📊 Project Steps

1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

2. Load Dataset

data = pd.read_csv('house_data.csv')
print(data.head())

3. Preprocess Data

data = data.dropna()  # Handle missing values
data = pd.get_dummies(data, drop_first=True)  # Convert categorical variables

4. Split Features and Target

X = data.drop("price", axis=1)
y = data["price"]

5. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

6. Train the Model

model = LinearRegression()
model.fit(X_train, y_train)

7. Make Predictions & Evaluate

y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

8. Visualize Results

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

📈 Future Improvements

Try other models: RandomForestRegressor, XGBoost, etc.

Use GridSearchCV for hyperparameter tuning

Feature scaling (StandardScaler/MinMaxScaler)

Add new features like location-based pricing

💡 Conclusion

This project is a great starting point for learning how to apply regression techniques using scikit-learn for real-world prediction problems.

📎 Author

Himanshu Tripathi

Made with ❤️ using Python and Scikit-learn
