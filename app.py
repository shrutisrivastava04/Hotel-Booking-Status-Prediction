import streamlit as st
import pandas as pd
from src.predict import predict

st.set_page_config(page_title="Booking Status Predictor", layout='wide')
st.image("https://images.unsplash.com/photo-1566073771259-6a8506099945", use_container_width=True)
st.markdown("<h1 style='text-align: center;'> Booking Status Prediction </h1>", unsafe_allow_html=True)
st.markdown("""
            <p style='text-align: center; font-size:18px;'>
            This application helps the hotel management predict whether a booking is likely to be <b>cancelled</b> or <b>not cancelled</b>.
            Fill in the booking details below to get an instant prediction.
            </p>
            """, unsafe_allow_html=True)
st.markdown("---")

st.subheader("Enter Booking Details")
col1, col2 = st.columns(2)

with col1:
    lead_time = st.number_input("Lead Time", min_value=0, value=30, help="Number of days between booking and arrival.")
    price = st.number_input("Price", min_value=0.0, value=100.0, help='Average price per room.')
    adults = st.number_input("Adults", min_value=0, value=2, help='Number of adults staying.')
    children = st.number_input("Children", min_value=0, value=0, help='Number of children staying.')
    meal_type = st.selectbox("Meal Type", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'], 
                            help="Type of meal plan selected.")

with col2:
    weekends = st.number_input("Weekend", min_value=0, value=0, help='Number of weekend nights booked.')
    weekdays = st.number_input("Weekday", min_value=0, value=0, help='Number of weekday nights booked.')
    room_type = st.selectbox("Room Type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4'],
                            help='Type of room reserved.')
    segment = st.selectbox("Segment", ['Online', 'Offline', 'Corporate'], help='Booking Segment Category.')
    repeat = st.selectbox("Repeat Customer", [0,1], help='Whether the customer has booked before (1 = Yes, 0 = No).')

requests = st.number_input("Special Requests", min_value=0, value=0, help='Number of special requests made.')
arrival_month = st.number_input("Arrival Month(1-12)", min_value=1, max_value=12, value=6, help='Month of arrival (integer coded).')

st.markdown("---")

# Prediction
input_data = pd.DataFrame([{'id':0,'room_type': room_type, 'meal_type': meal_type, 'segment': segment, 'lead_time': lead_time,'price': price, 
                            'adults': adults, 'children': children, 'weekends': weekends, 'weekdays': weekdays, 'repeat': repeat, 'requests': requests, 
                            'arrival_month': arrival_month}])

st.subheader("Input Summary")
st.dataframe(input_data, use_container_width=True)

if st.button("Predict Booking Status"):
    try:
        with st.spinner("Analyzing booking..."):
            result = predict(input_data)
        
        st.markdown("Prediction Result: ")

        if isinstance(result, (list, tuple)):
            pred = result[0]
        elif hasattr(result, "__getitem__"):
            pred = result[0]
        else:
            pred = result
            
        if pred == 1:
            st.markdown("""
                    <div style='padding:20px; border-radius:10px; background-color:#ff4b4b; color:white; text-align:center;'>
                    <h2> Likely to Cancel. </h2>
                    <p> This booking has a high probability of cancellation!</p>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
                    <div style='padding:20px; border-radius:10px; background-color:#4CAF50; color:white; text-align:center;'>
                    <h2> Likely to Cancel. </h2>
                    <p> This booking is expected to be honored!</p>
                    </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error during prediction{e}")
    
st.markdown("---")

st.markdown("""
            <div style='text-align:center; font-size:14px; color:gray;'>
            Developed by <b>Shruti Srivastava</b><br>
            srivastavashruti218@gmail.com
            </div>
            """, unsafe_allow_html=True)

