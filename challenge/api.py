# Standard library imports
from typing import List

# Third-party imports
import fastapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn

# Local imports
from challenge.model import DelayModel, InvalidDateFormatError

def create_features(flight_info_list: dict) -> pd.DataFrame:
    """
    Creates a DataFrame of features from a list of flight information.

    Parameters:
    flight_info_list (dict): A dictionary containing a list of flight information.

    Returns:
    pd.DataFrame: A DataFrame containing the flights' features.
    """

    # Set all the columns that the output DataFrame should have.
    feature_columns = [
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

    flight_features_df = pd.DataFrame(columns=feature_columns).astype(int)

    for flight_info in flight_info_list['flights']:
        single_flight_features_df = pd.DataFrame(np.zeros((1,len(feature_columns))), columns=feature_columns).astype(int)

        for column in feature_columns:
            feature, value = column.split('_')

            if feature == "MES":
                # if type(flight_info[feature]) != int:
                #     raise ValueError("Month (MES) must be an integer.")
                if flight_info[feature] < 1 or flight_info[feature] > 12:
                    raise ValueError("Month (MES) must be between 1 and 12.")
            
            if feature == "TIPOVUELO" and flight_info[feature] not in ["I", "N"]:
                raise ValueError("TIPOVUELO must be either 'I' or 'N'.")

            if flight_info[feature] == value:
                single_flight_features_df[column] = 1
            else:
                single_flight_features_df[column] = 0

        flight_features_df = pd.concat([flight_features_df, single_flight_features_df], ignore_index=True)

    return flight_features_df


app = fastapi.FastAPI()


class FlightInfo(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class FlightList(BaseModel):
    flights: List[FlightInfo]


model = DelayModel()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(flight_info: FlightList) -> dict:
    try:
        flights = flight_info.flights
        data = pd.DataFrame([dict(flight) for flight in flights])
        
        model.load('data/modelo.joblib')
        
        data = create_features(flight_info.dict())
        predictions = model.predict(data)
        
        return {"predict": predictions}

    except (ValueError, KeyError, InvalidDateFormatError) as e:
        return JSONResponse(status_code=400, content={"message": str(e)})


# Possible training endpoint
# @app.post("/train", status_code=200)
# async def train_model() -> dict:
#     try:
#         data = pd.read_csv('data/data.csv', dtype={1: str, 6: str})
#     except Exception as e:
#         raise JSONResponse(status_code=400, detail="Failed to load data.")

#     model = DelayModel()

#     features, target = model.preprocess(data=data, target_column='delay')

#     model.fit(features=features, target=target)

#     model.save('data/modelo.joblib')

#     return {"status": "model trained and saved successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)