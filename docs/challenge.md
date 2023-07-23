# Challenge Documentation

- This document provides a comprehensive overview of the improvements, changes, and decisions made throughout the process. It covers key aspects including the adoption of an object-oriented design, development of a FastAPI application, testing methodologies, and cloud deployment with CI/CD pipelines. It explains the rationale behind the choices and changes made, providing insight into the thought process and objectives of enhancing the scalability, maintainability, and reliability of the application.

## Model Selection

### Justification
- Based on the conclusions from the Data Science team, the model should be trained with the top 10 most important features and with balanced classes.

- Regarding the choice of the model, it was decided to use Logistic Regression over XGBoost due to its simplicity, speed, and greater interpretability. Although both models showed similar performance in evaluation metrics (precision, recall, F1-score), Logistic Regression was faster. While the time difference may seem small, it can be significant when it comes to implementing the model in a production environment where efficiency and speed are critical. On the other hand, XGBoost is known for its ability to handle datasets with many features, but this is not a significant factor in our case since we have already selected the top 10 most important features for our model.

## Test Script Changes

### `test_model_preprocess_for_serving` and `test_model_preprocess_for_training` Methods

#### Changes Made
- In both test methods, the assert statement comparing the DataFrame columns was modified. The original comparison was made directly with a list, however, the columns attribute of a pandas DataFrame returns a pandas Index object. To ensure a proper comparison, the .to_list() method was added to convert the DataFrame columns to a list.

#### Justification
- The use of the .to_list() method ensures a direct comparison of two lists. This rectifies a potential bug where, despite the DataFrame having the correct columns, the test would fail due to the type mismatch between a pandas Index object and a Python list. This adjustment maintains the spirit of the test while ensuring accurate results.


### Methods `test_health_check`, `test_should_failed_invalid_TIPOVUELO`, `test_should_failed_invalid_month`, `test_should_failed_missing_feature`, `test_should_fail_invalid_month_type` and `test_OPERA_Latin_American_Wings`

#### Changes Made
- Added new tests to cover various cases and situations that might arise when interacting with the API.

#### Justification
- test_health_check: This test checks that the /health route of the API is functioning correctly. It is good practice to have such a route in APIs for quick health checks of the service.

- test_should_failed_invalid_TIPOVUELO: This test checks that the API returns an error when given an invalid value for the TIPOVUELO field. This is useful for verifying that the API is correctly validating input data.

- test_should_failed_invalid_month: Similar to the previous test, this checks that the API returns an error when given an invalid value for the MES field.

- test_should_failed_missing_feature: This test checks that the API returns an error when a required field is missing in the input data. This is also useful for verifying input data validation.

- test_should_fail_invalid_month_type: This test checks that the API returns an error when the data type provided for the MES field is incorrect. Again, this checks input data validation.

- test_OPERA_Latin_American_Wings: This test seems to check that the API can handle a specific airline. This test makes sense if the API is expected to handle data from various airlines.

- These new tests cover a wider range of situations, thereby improving the robustness and quality of the code by ensuring it handles a variety of use cases and data input correctly.


## API Design and Implementation
- The API consists of a POST endpoint /predict which accepts a list of flight data and returns the corresponding predictions. The incoming data is validated and transformed into the correct format before being fed into the model.

- The data is expected in the following format:
```python
json
Copy code
{
    "flights": [
        {
            "OPERA": "<string>",
            "TIPOVUELO": "<string>",
            "MES": <integer>
        },
        ...
    ]
}
```

- Each flight object should include OPERA, TIPOVUELO, and MES. The API will return a corresponding list of predictions.

- Here is an example of the expected response:
```python
json
Copy code
{
    "predict": [0, 1, 0, ...]
}
```
### Error Handling
- The API also includes error handling for situations where the incoming data does not meet the expected format. If the data is missing any required fields or contains invalid values, the API will return a status code of 400 and a detailed error message.

### Testing the API
- The API has been thoroughly tested using unittest, a Python standard library for running tests. The tests ensure that the API handles both valid and invalid requests correctly, and that it returns the expected outputs.

### Future Implementation: Training Endpoint
- While not currently active in the deployed API, a /train endpoint was designed for potential future use. This endpoint would allow for the model to be retrained on updated data without needing to manually run the training process and redeploy the API.

- When a POST request is made to the /train endpoint, the server reads the latest data from the specified source, preprocesses the data, and retrains the model. The updated model is then saved for use in making future predictions.

- The /train endpoint provides an efficient way to keep the model up-to-date as new data becomes available, ensuring that the API always provides the most accurate predictions possible. 

- This approach showcases the flexibility and potential scalability of the API design, preparing for scenarios where continuous learning might be necessary to adapt to changing data trends.

## API Deployment on a Cloud Provider
- The API is deployed on Google Cloud Run, taking advantage of its serverless, containerized environment.

### Deployment Process
- The Docker image, built from the Dockerfile in the repository, is pushed to Google Container Registry (GCR) with the tag ml-challenge. It is then deployed to Cloud Run in the region northamerica-northeast1.

### Continuous Integration and Continuous Delivery (CI/CD)
- GitHub Actions is used for the CI/CD pipeline. On each push to the main branch, the workflow checks out the code, sets up Python, installs dependencies, runs unit tests, and if the tests pass, deploys the Docker image from GCR to Cloud Run.

- Sensitive information such as the Google Cloud Project ID and Service Account Key are securely stored and accessed using GitHub Secrets. This ensures the API is in a deployable state, and deployments are automated, secure, and up-to-date.


## Codebase Translation: Improvements and Changes

### 2.Features Generation

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


