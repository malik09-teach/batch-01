import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import  accuracy_score
from mlxtend.plotting import  plot_decision_regions
import pickle
import streamlit as st


model=pickle.load(open('placement.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

st.title("placement predictor")

cgpa=st.number_input("enter cgpa")
iq=st.number_input("enter iq")

if st.button("predict"):
    input_data=np.array([[cgpa,iq]])
    input_scale=scaler.fit_transform(input_data)
    
    predict=model.predict(input_scale)
    
    if predict[0]==1:
        st.success("placed")
    else:
        st.error("not place")    