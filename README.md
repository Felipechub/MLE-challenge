# Machine Learning Engineer Challenge

## Overview

Welcome to the **Machine Learning Engineer** Application Challenge. In this, you will have the opportunity to get closer to the reality of the role, and demonstrate your skills and knowledge in machine learning and cloud.

## Problem

A jupyter notebook (training.ipynb) has been provided with the work of a Data Scientist (from now on, the DS). The DS, trained a model to predict the probability of **delay** for a flight taking off or landing at SCL airport. The model was trained with public and real data, below we provide you with the description of the dataset:

|Column|Description|
|-----|-----------|
|`Fecha-I`|Scheduled date and time of the flight.|
|`Vlo-I`|Scheduled flight number.|
|`Ori-I`|Programmed origin city code.|
|`Des-I`|Programmed destination city code.|
|`Emp-I`|Scheduled flight airline code.|
|`Fecha-O`|Date and time of flight operation.|
|`Vlo-O`|Flight operation number of the flight.|
|`Ori-O`|Operation origin city code.|
|`Des-O`|Operation destination city code.|
|`Emp-O`|Airline code of the operated flight.|
|`DIA`|Day of the month of flight operation.|
|`MES`|Number of the month of operation of the flight.|
|`AÑO`|Year of flight operation.|
|`DIANOM`|Day of the week of flight operation.|
|`TIPOVUELO`|Type of flight, I =International, N =National.|
|`OPERA`|Name of the airline that operates.|
|`SIGLAORI`|Name city of origin.|
|`SIGLADES`|Destination city name.|

In addition, the DS considered relevant the creation of the following columns:

|Column|Description|
|-----|-----------|
|`high_season`|1 if `Date-I` is between Dec-15 and Mar-3, or Jul-15 and Jul-31, or Sep-11 and Sep-30, 0 otherwise.|
|`min_diff`|difference in minutes between `Date-O` and `Date-I`|
|`period_day`|morning (between 5:00 and 11:59), afternoon (between 12:00 and 18:59) and night (between 19:00 and 4:59), based on `Date-I`.|
|`delay`|1 if `min_diff` > 15, 0 if not.|

## Challenge


### Instructions

1. Create a github repository with all the contents of this folder. Remember that the repository must be public (or private but make sure we can access it).

2. Use a main branch for any official release that we should review, and a development branch for any increment. *Optional, take up some [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) development practice.*

3. You must send the link to the repository to the email from which you were contacted with the subject: MLE Challenge - [Name][Last Name],
Example: MLE Challenge - John Doe. Changes will be accepted in the repository until the date and time indicated in the email (or 5 days after receiving the challenge).

### Part I.

In order to operationalize the model, transcribe the `.ipynb` file into the `model.py` file:

- If you find something wrong (i.e. bug), fix it and/or change it, argue why.
- Choose the best model at your discretion, argue why.
- Apply all the good programming practices that you consider necessary in this item.
- The model should pass the tests by running `make model-test`.

Note:
- **You cannot** remove or change the name or arguments of provided methods.
- **You can** change/complete the implementation of the provided methods.
- **You can** create the extra classes and methods you seem necessary.

### Part II.

Deploy the model in an `API` with `FastAPI` (use the `api.py` file).

- The `API` should pass the tests by running `make api-test`.
- You should modify the tests in `tests/integration/test_api.py` to match the previous question.

Note: 
- **You can** use other framework but you should:
  - Argue why.
  - Implement your own tests (the `API` **MUST** be tested).

### Part III.

Deploy your `API` to you favorite cloud provider (we recomend to use GCP, you will need to put a credit card but no charge will be issue [[info]](https://cloud.google.com/free/docs/free-cloud-features#billing_verification)) where we going to run a series of test and run performance benchmark.

We are looking for seeing a proper `CI/CD` implementation for this development. We recommend using GitHub Actions but this is not a constraint, feel free to use whatever you wish (we will check the implementation).

Put your url in the `Makefile` (`line 26`). Then we will test the perfomance with the comand `make stress-test`. Make sure to go in `tests/stress/api-stress.py` and add the necesary inputs so the test run in the endpoint.

**It is important that the API is deployed until we review the tests. If your API needs any authentication method, you must indicate which one and ensure that we can access it.**

---

All documentation needed for the above steps should go in the folder `docs` in a file called `challenge.md`.
