"""
This is the main file that contains the FastAPI app.
It should contain the following endpoints: 
- list_client_ids: List all client ids.
- summarize: Summarize the dataset.
- predict: Predict client id.
- client_info: Get client info.
- explanation: Get explanation for the decision.
- ping: Check if the app is up and running.
"""

import pickle, json
from fastapi import FastAPI
import pandas as pd

# create the FastAPI app
app = FastAPI()


# load the dataset
# df = pd.read_csv("MY-DATASET.csv")

# load the model and the explanation
# model = pickle.load("MY-MODEL.pkl")

# explanation_dict = json.load("MY-EXPLANATION.json")  #
# explanation_dict = pickle.load("MY-EXPLANATION.pkl")  #


@app.get("/list_client_ids")
async def list_client_ids():
    """List all client ids."""

    # should implement the logic to list all client ids

    # Build a list of client ids and return it

    return {"client_ids": [1, 2, 3]}


@app.get("/summarize")
async def summarize():
    """Summarize the dataset."""

    # should implement the logic to give dataset insights (e.g., mean, std, etc. on each feature for the dataset)

    # describe the dataset
    return {
        "salary": {"mean": 1, "std": 2},
        "age": {"mean": 3, "std": 4},
        "amount": {"mean": 5, "std": 6},
    }


@app.get("/client_info/{client_id}")
async def client_info(client_id):
    """Get client info."""

    # should implement the logic to client_info : dive all usefull information about the client

    # answer wih the client info

    return {"salary": 1, "age": 2, "amount": 3, "client_id": 4}


@app.get("/predict/{client_id}")
async def predict(client_id):
    """Predict client id."""

    # should implement the logic to predict the client id
    # 1 True False or 0/1
    # 2 Predtict probability -> force de prediction
    # predict proba 0.4999999 vs 0.000001

    # load the dataset
    # fin the reevant user for the client_id
    # make the prediction
    # return the prediction

    return {"prediction": 1}


@app.get("/explanation/{client_id}")
async def explanation(client_id):
    """Get client info."""

    # should implement the logic to explanation the decision with feature importance
    # answer wih the feature importance dic t
    return {"salary": 1, "age": 2, "amount": 3, "client_id": 4}


@app.get("/ping")
async def root():
    return "pong"
