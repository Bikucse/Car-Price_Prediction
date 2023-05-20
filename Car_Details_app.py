import numpy as np
# numpy alised as np
import pandas as pd 
# Pandas alised as pd
from sklearn import *
import pickle
import streamlit as st



model=pickle.load(open('lr_model.pkl','rb'))
df=pickle.load(open('data.pkl','rb'))


st.title('Car Selling Price Prediction')
st.header('Fill the details of the car')



Brand=st.selectbox('Brand',df['Brand'].unique())
year=st.selectbox('year is',[2017, 2012, 2015, 2014, 2013, 2018, 2016])
km_driven=st.selectbox('km_driven',df['km_driven'].unique())
fuel=st.selectbox('fuel',['Petrol','Diesel','CNG','LPG','Electric'])
seller_type=st.selectbox('seller_type',df['seller_type'].unique())
transmission=st.selectbox('transmission',df['transmission'].unique())
owner=st.selectbox('owner',df['owner'].unique())


btn=st.button('Predict car Price')

if btn:
    test1 = np.array([Brand,year,km_driven,fuel,seller_type,transmission,owner])
    test1 = test1.reshape([1,7])

    st.success(model.predict(test1)[0])