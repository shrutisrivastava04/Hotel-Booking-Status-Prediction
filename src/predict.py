import pickle
import pandas as pd

# Load model + transformers
model = pickle.load(open("models/model.pkl", "rb"))
imputer = pickle.load(open("models/imputer.pkl", "rb"))
encoder = pickle.load(open("models/encoder.pkl", "rb"))

expected_cols = ['id', 'room_type', 'meal_type', 'segment', 'lead_time', 'price', 'adults', 
                    'children', 'weekends', 'weekdays', 'repeat', 'requests', 'arrival_month']

# Predictions
def predict(data_df):
    try:
        #Adding missing columns
        for col in expected_cols:
            if col not in data_df.columns:
                data_df[col] = 0
        #Keeping only expected columns and correct order
        data_df=data_df[expected_cols]
        # Applying preprocessing
        data_imputed = imputer.transform(data_df)
        data_encoded = encoder.transform(data_imputed)
        # Creating predictions
        prediction = model.predict(data_encoded)
        return prediction
    
    except Exception as e:
        print("Prediction Error:", e)
        raise e