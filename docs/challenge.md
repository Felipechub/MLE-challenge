# Challenge Documentation

This document outlines the changes and improvements made during the process of transcribing the Jupyter Notebook `.ipynb` file to the `model.py` file.

## Improvements and Changes

## 2.Features Generation

### `get_period_day` Function

#### Changes Made

- Adjusted the logic of time period comparisons to avoid misclassification of times that fall exactly on the boundary of a period, such as 05:00.
- Incorporated error handling to accommodate situations where the input cannot be parsed into a datetime object.

#### Justification

- The adjustment in time period comparison logic rectified a bug in the original implementation, which led to certain times being misclassified.
- Error handling was introduced as a good programming practice to prevent function failure with unexpected input data.

### `is_high_season` Function

#### Changes Made

- Refactored the logic to use a list of high season date ranges, reducing code duplication and enhancing readability.
- Integrated error handling to manage instances where the input cannot be parsed into a datetime object.

#### Justification

- The refactoring of date range logic was an enhancement aimed at reducing code duplication and improving readability.
- Error handling was added as a good programming practice to prevent function failure with unexpected input data.

### `get_min_diff` Function

#### Changes Made

- Added error handling to manage instances where the input cannot be parsed into a datetime object.
- Kept the conversion of dates into datetime objects outside of the function to avoid undesired alterations in the original data.

#### Justification

- Error handling was introduced as a good programming practice to prevent function failure with unexpected input data.
- The conversion of data into datetime objects was kept outside of the function to avoid undesired side effects in other parts of the code.

### `calculate_delay` Function

#### Changes Made

- Transformed the original procedure into a function to enhance code reusability and comprehensibility.

#### Justification

- Transforming the procedure into a function is a good programming practice that enhances code reusability and readability.

## 4. Training
### 4.a. Data Split (Training and Validation)
- In the initial code, the data split and preparation was done in one chunk of code. The major issue was that the `data` DataFrame was used instead of `training_data` DataFrame, which was created earlier. This could lead to data leakage because the test data might be included in the training data. 

#### Changes Made
- Changed the use of the `training_data` DataFrame instead of the `data` DataFrame.
- To enhance readability, reusability and maintainability, the process was refactored into a class named `DataPreparation` with a method `prepare_data()`.

#### Example of Usage

Here's how you would use the `DataPreparation` class to prepare the data:

```python
data_preparation = DataPreparation(test_size=0.33, random_state=42)
x_train, x_test, y_train, y_test = data_preparation.prepare_data(data)

print(f"train shape: {x_train.shape} | test shape: {x_test.shape}")
```

#### Justification
- This change enhances the readability of the code by clearly delineating the steps in the data preparation process. It also increases the modularity of the code, making it easier to reuse the data preparation process in different parts of the project or in future projects.


## 6.b Model Selection

#### Justification
- Based on the conclusions from the Data Science team, the model should be trained with the top 10 most important features and with balanced classes.

- Regarding the choice of the model, it was decided to use Logistic Regression over XGBoost due to its simplicity, speed, and greater interpretability. Although both models showed similar performance in evaluation metrics (precision, recall, F1-score), Logistic Regression was faster. While the time difference may seem small, it can be significant when it comes to implementing the model in a production environment where efficiency and speed are critical. On the other hand, XGBoost is known for its ability to handle datasets with many features, but this is not a significant factor in our case since we have already selected the top 10 most important features for our model.

