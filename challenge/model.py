# Imports
from datetime import datetime
import pandas as pd
import xgboost as xgb
from typing import Tuple, Union, List

# Constants
HIGH_SEASON_RANGES = [
    ('15-Dec', '31-Dec'),
    ('1-Jan', '3-Mar'),
    ('15-Jul', '31-Jul'),
    ('11-Sep', '30-Sep'),
]
THRESHOLD_IN_MINUTES = 15

# Functions
def get_period_day(date):
    # Convert string times to time objects once
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()

    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    
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

# The function was optimized for better performance and readability, and includes error handling.
def is_high_season(fecha):
    """
    Function to determine if a given date falls within the high season.
    
    Args:
        fecha (str): Date in '%Y-%m-%d %H:%M:%S' format.
        
    Returns:
        int: Returns 1 if the date falls within the high season, and 0 otherwise.
        If the date format is invalid, the function returns None and prints an error message.
    """
    try:
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        print(f"Invalid date format: {fecha}")
        return None
    
    fecha_año = fecha.year

    for start, end in HIGH_SEASON_RANGES:
        range_start, range_end = get_date_range(start, end, fecha_año)
        if range_start <= fecha <= range_end:
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
import numpy as np

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

from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self, test_size=0.33, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def prepare_data(self, data):
        data = self._one_hot_encode(data)
        return self._train_test_split(data)

    def _one_hot_encode(self, data):
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix='MES')], 
            axis=1
        )
        target = data['delay']
        return features, target

    def _train_test_split(self, data):
        x_train, x_test, y_train, y_test = train_test_split(*data, test_size=self.test_size, random_state=self.random_state)
        return x_train, x_test, y_train, y_test


class DelayModel:

    def __init__(
        self
    ):
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        self._top_features = None
        
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
            Tuple[pd.DataFrame, pd.DataFrame]: features and target, if target_column is set.
            or
            pd.DataFrame: features, if target_column is not set.
        """
        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Determine the time of day for each flight
        data_copy['period_day'] = data_copy['Fecha-I'].apply(get_period_day)
        
        # Determine if each flight is in the high season
        data_copy['high_season'] = data_copy['Fecha-I'].apply(is_high_season)
        
        # Calculate the difference in minutes between the 'Fecha-O' and 'Fecha-I'
        data_copy['min_diff'] = data_copy.apply(get_min_diff, axis=1)
        
        if target_column:
            # Create a binary delay indicator based on the difference in minutes
            data_copy[target_column] = calculate_delay(data_copy, THRESHOLD_IN_MINUTES)
            
            # Separate features from the target variable
            target = data_copy[target_column]
            features = data_copy.drop(columns=target_column)
            
            return features, target
        else:
            return data_copy


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
        self.model.fit(features, target)


    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            List[int]: predicted targets.
        """
        if self._model is None:
            raise Exception("The model must be fitted with 'fit' before predictions can be made.")
        
        return self._model.predict(features)