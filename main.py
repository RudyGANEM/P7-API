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

import pickle, json, joblib
import pandas as pd
from fastapi import FastAPI
import uvicorn
from colorama import init
import shap
import streamlit as st

# create the FastAPI app
app = FastAPI()

df = pd.read_csv("df_preprocessed_1000.csv")

with open('feature_importance.json', 'r') as file:
    data = json.load(file)

with open('grid2_export.pk', 'rb') as file:
    model = joblib.load(file)


st.dataframe(df.head())

#df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].apply(lambda x: '{:,.0f}'.format(x))
#df['AGE'] = - df.DAYS_BIRTH / 365
df['DAYS_BIRTH'].round(1)
#df['AMT_CREDIT'] = df['AMT_CREDIT'].apply(lambda x: '{:,.0f}'.format(x))




best_estimator = model.best_estimator_
preprocessor_pipe = best_estimator[:-1]
df_transformed = best_estimator.named_steps['columntransformer'].transform(df)
explainer = shap.LinearExplainer(best_estimator.named_steps['estimator'], df_transformed, feature_perturbation="interventional")
shap_values = explainer.shap_values(df_transformed)


@app.get("/")
def index():
    return {"message": "Welcome to fastapi"}



@app.get("/ping")
async def root():
    return "pong"



@app.get("/list_client_ids")
async def list_client_ids():
    """List all client ids."""

    # should implement the logic to list all client ids

    # Build a list of client ids and return it
    client_ids_list = df['SK_ID_CURR'].tolist()

    return {"client_ids": client_ids_list}



@app.get("/summarize")
async def summarize():
    """Summarize the dataset."""

    # should implement the logic to give dataset insights (e.g., mean, std, etc. on each feature for the dataset)

    # describe the dataset

    return {
        "AMT_INCOME_TOTAL": {"mean": df['AMT_INCOME_TOTAL'].mean(), "std": df['AMT_INCOME_TOTAL'].std()},
        "AGE": {"mean": df['DAYS_BIRTH'].mean(), "std": df['DAYS_BIRTH'].std()},
        "AMT_CREDIT": {"mean": df['AMT_CREDIT'].mean(), "std": df['AMT_CREDIT'].std()},
    }



@app.get("/client_info/{client_id}")
async def client_info(client_id):
    """Get client info."""

    # should implement the logic to client_info : give all usefull information about the client

    # answer wih the client info

    return {"AMT_INCOME_TOTAL": df.loc[df['SK_ID_CURR'] == int(client_id), 'AMT_INCOME_TOTAL'].values[0], 
            "AGE": df.loc[df['SK_ID_CURR'] == int(client_id), 'DAYS_BIRTH'].values[0], 
            "AMT_CREDIT": df.loc[df['SK_ID_CURR'] == int(client_id), 'AMT_CREDIT'].values[0], 
            "client_id": int(client_id)
           }



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

    df_client_id = df.loc[df['SK_ID_CURR'] == int(client_id), :]
    df_client_id = df_client_id.drop(['SK_ID_CURR'], axis=1)
    prediction = model.predict(df_client_id)
    predict_proba = model.predict_proba(df_client_id)
    
    return {
            "prediction": prediction[0].item(),
            "predict_proba_0": predict_proba[0, 0].round(2),
            "predict_proba_1": predict_proba[0, 1].round(2),
           }


@app.get("/explanation/{client_id}")
async def explanation(client_id):
    """Get client info."""
    # should implement the logic to explanation the decision with feature importance
    # answer wih the feature importance dic t
    client_id_index = df.index[df['SK_ID_CURR'] == int(client_id)].tolist()[0]
    return dict(zip(preprocessor_pipe.get_feature_names_out(), shap_values[client_id_index]))









@app.get("/items")
def get_items():
    return [
        {"id": 1, "description": "Item 1"},
        {"id": 2, "description": "Item 2"},
        {"id": 3, "description": "Item 3"},
    ]




 
if __name__ == '__main__':
    init()
    uvicorn.run(app, host = "0.0.0.0", port=80)