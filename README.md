# Predicting House Prices

## Introduction
This project focuses on predicting house prices using machine learning techniques. We will walk through the entire process from data collection to model evaluation. The primary goal is to build a regression model that can accurately predict house prices based on various features.

## Table of Contents
1. [Data Collection](#data-collection)
2. [Data Exploration](#data-exploration)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Selection](#feature-selection)
5. [Model Selection](#model-selection)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Comparison of Different Models](#comparison-of-different-models)
8. [Conclusion](#conclusion)

## Data Collection <a name="data-collection"></a>

We start by obtaining a dataset of house prices. For this project, we used the Amsterdam house price prediction dataset, which is publicly available on Kaggle. The dataset contains information on various features such as square footage, location coordinates, and the number of rooms.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('/content/HousingPrices-Amsterdam-August-2021.csv')
```

## Data Exploration <a name="data-exploration"></a>

Data exploration helps us understand the structure of the dataset, check for missing values, and gain insights into the distribution of house prices and features.

```python
# Print the head of the dataset
print(df.head())

# Check for missing values
print(df.info())
```

## Data Preprocessing <a name="data-preprocessing"></a>

Data preprocessing involves cleaning the data by handling missing values and outliers. In this project, we replaced missing values with the median value for each feature. Additionally, we selected relevant features based on the correlation matrix or feature importance techniques.

## Feature Selection <a name="feature-selection"></a>

Feature selection is crucial for model efficiency and interpretability. We chose the following features for our model:

- Area
- Longitude (Lon)
- Latitude (Lat)
- Number of Rooms

## Model Selection <a name="model-selection"></a>

For this project, we selected the Random Forest Regression model for house price prediction. This model is known for its robustness and accuracy in regression tasks.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
```

## Model Training and Evaluation <a name="model-training-and-evaluation"></a>

We split the data into training and test sets and trained the Random Forest Regression model. We evaluated the model's performance using metrics such as Mean Squared Error (MSE) and R-squared (R2).

```python
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, df['Price'], test_size=0.25, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error:', mse)
print('R-squared:', r2)
```

## Comparison of Different Models <a name="comparison-of-different-models"></a>

We also compared the performance of various regression models, including Linear Regression, Lasso Regression, Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, and K-Nearest Neighbors. The best model was selected based on R-squared accuracy.

```python
# Create a DataFrame to compare model performance
results_df = pd.DataFrame({
    "Model": [name for name, _ in models],
    "MAE": mae_scores,
    "MSE": mse_scores,
    "RMSE": rmse_scores,
    "R2": r2_scores
})

# Find the best model
max = 0
for i in range(len(results_df['R2'])):
    if i == (len(results_df['R2'])) - 1:
        if results_df['R2'][i] > results_df['R2'][max]:
            max = i
    else:
        if results_df['R2'][i] > results_df['R2'][i + 1]:
            max = i

print(f"Best Model : {results_df['Model'][max]} with accuracy : {results_df['R2'][max]}")
```

## Conclusion <a name="conclusion"></a>

This project demonstrates the process of predicting house prices using machine learning. We collected, explored, and preprocessed data, selected relevant features, and trained a Random Forest Regression model. Additionally, we compared the model's performance with other regression algorithms to choose the best model for house price prediction. This information can be valuable for real estate applications and investors looking to make informed decisions regarding property investments.
