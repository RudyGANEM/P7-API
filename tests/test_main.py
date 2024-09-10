import sys
import os
import pandas as pd

# Ajoute le répertoire parent au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to fastapi"}
    
def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == "pong"
    
    
df = pd.read_csv("df_preprocessed_1000.csv")


def test_list_client_ids():
    response = client.get("/list_client_ids")
    client_ids_list = df['SK_ID_CURR'].tolist()
    assert response.status_code == 200
    assert response.json() == {"client_ids": client_ids_list}
    
    
    
def test_summarize():
    response = client.get("/summarize")
    assert response.status_code == 200
    assert response.json() == {
            "AMT_INCOME_TOTAL": {"mean": df['AMT_INCOME_TOTAL'].mean(), "std": df['AMT_INCOME_TOTAL'].std()},
            "AGE": {"mean": df['DAYS_BIRTH'].mean(), "std": df['DAYS_BIRTH'].std()},
            "AMT_CREDIT": {"mean": df['AMT_CREDIT'].mean(), "std": df['AMT_CREDIT'].std()},
        }    