import pandas as pd
from preprocessing import preprocess_train, preprocess_test
import pickle

# Loading Datasets
train_df = pd.read_csv("data/raw/train_hb.csv")
test_df = pd.read_csv("data/raw/test_hb.csv")

# Processing Training & Test Dataset
processed_train, imputations, encoder = preprocess_train(train_df)
processed_test = preprocess_test(test_df, imputations, encoder)

# Saving processed data
processed_train.to_csv('data/processed/processed_train.csv', index=False)
processed_test.to_csv('data/processed/processed_test.csv', index=False)

# Saving transformers
pickle.dump(imputations, open('models/imputer.pkl', 'wb'))
pickle.dump(encoder, open('models/encoder.pkl', 'wb'))
