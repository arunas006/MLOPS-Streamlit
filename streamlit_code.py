import pandas as pd
import numpy as np
import streamlit as st
import pickle
import sklearn

st.set_page_config(layout='wide')

st.header(
    """
        CARS24 PRICE PREDICTION
    
    """,divider='red'
)

data=pd.read_csv("car24-data.csv")

st.dataframe(data.head())

col1, col2, col3 = st.columns(3)

with col1:
    fuel_type=st.selectbox("Select The Fuel Type",["Diesel","Petrol","CNG","LPG","Electric"])

with col3:
    engine=st.slider("Required Engine Power",min_value=500,max_value=5000,step=100)


col4,col5,col6=st.columns(3)

with col4:
    transmission_type=st.selectbox("Select The Required Transmission Type",["Manual","Automatic"])

with col6:
    seat=st.selectbox("Select The Required Passenger Capacity",[4,5,7,9,11])

encode_dict={
    "fuel_type":{'Diesel':1,'Petrol':2,'CNG':3,'LPG':4,'Electric':5},
    "seller_type":{'Dealer':1,'Individual':2,'Trustmark Dealer':3},
    "transmission_type":{'Manual':1,'Automatic':2}
}

def model_prediction(fuel_type,engine,transmission_type,seat):
    with open("car_pred_pickle",'rb') as file:
        reg_model=pickle.load(file)

    fuel=encode_dict["fuel_type"][fuel_type]
    transmission=encode_dict["transmission_type"][transmission_type]
    
    Input_feature=[[2018.0,1,4000,fuel,transmission,19.70,engine,86.30,seat]]

    return np.round(reg_model.predict(Input_feature)[0],3)

if st.button("PREDICT"):
    output=model_prediction(fuel_type,engine,transmission_type,seat)
    st.write(f"The Price of Car which your looking for is {output} lakhs Rupees")
else:
    st.write("Awaiting for Required features confirmation")