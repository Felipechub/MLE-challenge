# Standard library imports
from datetime import datetime
from typing import Tuple, List

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import xgboost as xgb
from joblib import dump, load


class InvalidDateFormatError(Exception):
    """Exception raised for errors in the date format.

    Attributes:
        date -- input date which caused the error
        message -- explanation of the error
    """

    def __init__(self, date, message="Fecha tiene un formato inválido. Se espera '%Y-%m-%d %H:%M:%S'"):
        self.date = date
        self.message = message
        super().__init__(self.message)


# Constants
HIGH_SEASON_RANGES = [
    ('15-Dec', '31-Dec'),
    ('1-Jan', '3-Mar'),
    ('15-Jul', '31-Jul'),
    ('11-Sep', '30-Sep'),
]
THRESHOLD_IN_MINUTES = 15

# Functions
# def get_period_day(date):
#     # Convert string times to time objects once
#     morning_min = datetime.strptime("05:00", '%H:%M').time()
#     morning_max = datetime.strptime("11:59", '%H:%M').time()
#     afternoon_min = datetime.strptime("12:00", '%H:%M').time()
#     afternoon_max = datetime.strptime("18:59", '%H:%M').time()
#     evening_min = datetime.strptime("19:00", '%H:%M').time()

#     date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    
#     if morning_min <= date_time <= morning_max:
#         return 'mañana'
#     elif afternoon_min <= date_time <= afternoon_max:
#         return 'tarde'
#     else:
#         return 'noche'

def get_period_day(date):
    try:
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    except ValueError:
        raise InvalidDateFormatError(date)

    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()

    if morning_min <= date_time <= morning_max:
        return 'mañana'
    elif afternoon_min <= date_time <= afternoon_max:
        return 'tarde'
    else:
        return 'noche'

def get_date_range(start, end, year):
    """
    Helper function to get the start and end dates given the day and month.
    
    Args:
        start (str): Start date in '%d-%b' format.
        end (str): End date in '%d-%b' format.
        year (int): Year of the date range.
    
    Returns:
        tuple: A tuple containing the start and end dates.
    """
    start_date = datetime.strptime(start, '%d-%b').replace(year=year)
    end_date = datetime.strptime(end, '%d-%b').replace(year=year)
    return start_date, end_date



# def is_high_season(fecha):
#     """
#     Function to determine if a given date falls within the high season.
    
#     Args:
#         fecha (str): Date in '%Y-%m-%d %H:%M:%S' format.
        
#     Returns:
#         int: Returns 1 if the date falls within the high season, and 0 otherwise.
#         If the date format is invalid, the function returns None and prints an error message.
#     """
#     try:
#         fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
#     except ValueError:
#         print(f"Invalid date format: {fecha}")
#         return None
    
#     fecha_año = fecha.year

#     for start, end in HIGH_SEASON_RANGES:
#         range_start, range_end = get_date_range(start, end, fecha_año)
#         if range_start <= fecha <= range_end:
#             return 1

#     return 0

def is_high_season(date):
    try:
        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        raise InvalidDateFormatError(date)

    date_year = date.year

    for start, end in HIGH_SEASON_RANGES:
        range_start, range_end = get_date_range(start, end, date_year)
        if range_start <= date <= range_end:
            return 1

    return 0

def get_min_diff(row):
    """
    Function to calculate the difference in minutes between 'Fecha-O' and 'Fecha-I'.
    
    Args:
        row (Series): A row of the dataframe.
        
    Returns:
        float: The difference in minutes between 'Fecha-O' and 'Fecha-I'.
    """
    try:
        # Convert the dates to datetime within the function
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')

        # Calculate the difference in seconds and convert to minutes
        min_diff = (fecha_o - fecha_i).total_seconds() / 60
    except Exception as e:
        print(f"Error calculating min_diff: {e}")
        return None
    return min_diff


def calculate_delay(data, threshold=15):
    """
    Function to calculate delay given a threshold.
    
    Args:
        data (DataFrame): The dataframe with the 'min_diff' column.
        threshold (int): The threshold in minutes to determine delay.
        
    Returns:
        Series: A pandas Series indicating if the delay is more than the threshold.
    """
    return np.where(data['min_diff'] > threshold, 1, 0)


class DelayModel:
    def __init__(
        self
    ):
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)

    def preprocess(
        self, 
        data: pd.DataFrame, 
        target_column: str = None
        ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """        

        # Apply your preprocessing steps
        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis=1)
        data['delay'] = np.where(data['min_diff'] > THRESHOLD_IN_MINUTES, 1, 0)
        
        training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)
        features = pd.concat([
            pd.get_dummies(training_data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(training_data['TIPOVUELO'], prefix = 'TIPOVUELO'),
            pd.get_dummies(training_data['MES'], prefix = 'MES')],
            axis = 1
        )
        
        top_10_features = [
                    "OPERA_Latin American Wings",
                    "MES_7",
                    "MES_10",
                    "OPERA_Grupo LATAM",
                    "MES_12",
                    "TIPOVUELO_I",
                    "MES_4",
                    "MES_11",
                    "OPERA_Sky Airline",
                    "OPERA_Copa Air"
                ]
        
        # Check if the target column is specified
        if target_column != None:
            target = training_data[target_column]

            return features[top_10_features], pd.DataFrame(target)
        else:
            return features[top_10_features]

    def fit(
        self, 
        features: pd.DataFrame, 
        target: pd.DataFrame
        ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        
        y_train = target.iloc[:, 0]

        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0/n_y1

        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
        self._model.fit(features, y_train)
        
    def predict(
        self, 
        features: pd.DataFrame
        ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        predictions = self._model.predict(features)
        return [1 if pred > 0.5 else 0 for pred in predictions]

    def save(self, filepath):
        dump(self._model, filepath)

    def load(self, filepath):
        self._model = load(filepath)