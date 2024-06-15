from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd  # Ajout de pandas pour la conversion JSON -> DataFrame

app = Flask(__name__)
CORS(app)

# Charger le modèle pré-entraîné
model = joblib.load('model/best_model.pkl')

# Endpoint pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données de la requête POST
        data = request.get_json()

        # Créer un DataFrame à partir des données JSON
        df = pd.DataFrame([data])

        df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce').fillna(0).astype(int)
        df['milage'] = pd.to_numeric(df['milage'], errors='coerce').fillna(0).astype(float)

        # Préparer les données pour la prédiction
        X = df[['brand', 'model','model_year', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title', 'milage']]

        # Prédiction avec le modèle chargé
        prediction = model.predict(X)

        # Formater la prédiction comme un prix en dollars
        predicted_price = '${:,.2f}'.format(prediction[0])

        # Renvoyer la prédiction comme réponse JSON
        return jsonify({'price': predicted_price})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
