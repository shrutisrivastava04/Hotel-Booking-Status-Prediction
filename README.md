# Hotel-Booking-Status-Prediction

## Overview

This project is a Machine Learning powered web application built using Streamlit that predicts whether a hotel booking is likely to be cancelled or confirmed based on user-provided booking details.

The goal of this project is to help hotel management and staff make data-driven decisions by identifying high-risk bookings in advance, allowing better resource planning and customer handling.

## Features

- Predicts booking cancellation in real time
- Clean and interactive user interface using Streamlit
- Machine Learning model trained on booking data
- User-friendly form with guided inputs
- Instant prediction results with clear visual feedback
- Modular code structure (separation of processing, prediction, and UI)

## Machine Learning Pipeline

The model follows a structured pipeline:

1. Data Preprocessing
    - Handling missing values using SimpleImputer
    - Separating categorical and numerical features
2. Feature Engineering
    - Extracting 'arrival_month' from date features
    - Cleaning and transforming raw inputs
3. Encoding
    - Categorical variables encoded using OrdinalEncoder
4. Model Training
    - Trained of historical booking data
    - Learns patterns that lead to cancellations

## Tech Stack

- Python
- Pandas & NumPy (Data Handling)
- Scikit-Learn (Machine Learning)
- Seaborn & Matplotlib (Data Visualisation)
- Streamlit (Web App Interface)

## Network URL

Network URL: http://192.168.0.185:8501

## Use Case

This application can be used by:
- Hotel staff to identify risky bookings
- Management to optimize room allocation
- Businesses to reduce revenue loss due to cancellations

## Conclusion

This project demonstrates the integration of Machine Learning and Web Development to solve a real-world problem. It showcases skills in data preprocessing, model building, and interactive UI development.

## Author

Shruti Srivastava
- srivastavashruti218@gmail.com