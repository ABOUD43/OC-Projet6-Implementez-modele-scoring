import numpy as np
import pandas as pd
import pickle
from flask import Flask, jsonify, request
from lightgbm import LGBMClassifier
import os

app = Flask(__name__)

Xtest = pd.read_csv("Xtest.csv")
#local:
#Xtest = pd.read_csv("Xtest.csv")
#with open('model_lgbm.pickle', 'rb') as file:
    #lgbm = pickle.load(file)
#with open('scaler.pkl', 'rb') as f:
    #std = pickle.load(f)  
#with open('imputer.pkl', 'rb') as f:
    #imputer = pickle.load(f)  

with open('model_lgbm.pickle', 'rb') as file:
    lgbm = pickle.load(file)

with open('scaler.pkl', 'rb') as f:
    std = pickle.load(f)

with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

@app.route('/')
def home():
    return 'Entrer une ID client dans la barre URL'


@app.route('/<int:ID>/')
def requet_ID(ID):
    if ID not in list(Xtest['SK_ID_CURR']):
        result = 'Ce client n\'est pas dans la base de données'
    else: 
        client_features = Xtest[Xtest['SK_ID_CURR'] == int(ID)].drop(columns=['TARGET','SK_ID_CURR','Unnamed: 0'], axis=1)
        
      
        client_features_imputed = imputer.transform(client_features)  # Imputation des valeurs manquantes
        client_features_scaled = std.transform(client_features_imputed)  # Standardisation des données
        y_proba = lgbm.predict_proba(client_features_scaled)[:, 1]

        # Calculer le risque et formuler une réponse
        if y_proba >= 0.478:
            result = ('Ce client est non solvable avec un taux de risque de ' + str(np.around(y_proba[0]*100,2))+'%')
        else:
            result = ('Ce client est solvable avec un taux de risque de ' + str(np.around(y_proba[0]*100,2))+'%')

    return result


@app.route('/test_model')
def test_model():
    try:
        # Test basique de prédiction avec un échantillon de validation
  
        sample = Xtest.drop(['SK_ID_CURR', 'TARGET', 'Unnamed: 0'], axis=1).iloc[0:1]  # Prendre un échantillon
        sample_imputed = imputer.transform(sample)
        sample_scaled = std.transform(sample_imputed)
        y_proba = lgbm.predict_proba(sample_scaled)[:, 1]
        return jsonify({'prediction': y_proba[0]})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()
