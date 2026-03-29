import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

def create_preprocessor():
    # Differentiating categorical & numerical columns
    cat_cols = ['room_type', 'meal_type', 'segment']
    num_impute_cols = ['lead_time', 'price']
    
    # Imputations
    imputations = ColumnTransformer([
        ('median', SimpleImputer(strategy='median'), num_impute_cols),
        ('most_frequent', SimpleImputer(strategy='most_frequent'), ['room_type'])
    ], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')
    
    # Encoding
    encoder = ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')
    
    return imputations, encoder

def feature_engineering(df):
    # Converting arrival to month
    df['arrival'] = pd.to_datetime(df['arrival'], errors='coerce')
    df['arrival_month'] = df['arrival'].dt.month
    
    # Filling missing values in month column
    df['arrival_month'] = df['arrival_month'].fillna(df['arrival_month'].mode()[0])
    
    # Drop original column
    df = df.drop(columns=['arrival'])
    
    return df

def preprocess_train(df):
    imputations, encoder = create_preprocessor()
    
    # Target separation
    x = df.drop(columns=['booking_status'])
    y = df['booking_status']
    
    # Feature engineering
    x = feature_engineering(x)
    x_imputed = imputations.fit_transform(x)
    x_encoded = encoder.fit_transform(x_imputed)
    
    x_final = x_encoded.copy()
    x_final['booking_status'] = y.values
    
    return x_final, imputations, encoder

def preprocess_test(df, imputations, encoder):
    df = feature_engineering(df)
    x_imputed = imputations.transform(df)
    x_encoded = encoder.transform(x_imputed)
    
    return pd.DataFrame(x_encoded)