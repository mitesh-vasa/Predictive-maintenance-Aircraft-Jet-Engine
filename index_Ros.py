import streamlit as st
import warnings
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

import pickle
file1=open('scale.pkl','rb')
file2=open('supportvectorclassifier.pkl','rb')
scale=pickle.load(file1)
svc=pickle.load(file2)

st.write("# Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation")
st.image('final.png')
st.write("## Please Enter Sensor Reading Values")
a=st.number_input('(LPC Outlet Temperature) (◦R)', step=0.01,format='%2f')
b=st.number_input('(HPC Outlet Temperature) (◦R)',step=0.01,format='%2f')
c=st.number_input('(LPT Outlet Temperature) (◦R)',step=0.01,format='%2f')
d=st.number_input('(HPC Outlet Pressure) (psia)',step=0.01,format='%2f')
e=st.number_input('(Physical Fan Speed) (rpm)',step=0.01,format='%2f')
f=st.number_input('(HPC Outlet Static Pressure) (psia)',step=0.01,format='%2f')
g=st.number_input('(Ratio of Fuel Flow to Ps30) (pps/psia)',step=0.01,format='%2f')
h=st.number_input('(Corrected Fan Speed) (rpm)',step=0.01,format='%2f')
i=st.number_input('(Bypass Ratio)',step=0.01,format='%2f')
j=st.number_input('(Bleed Enthalpy)',step=0.01,format='%2f')
k=st.number_input('(High-Pressure Turbines Cool Air Flow)',step=0.01,format='%2f')
l=st.number_input('(Low-Pressure Turbines Cool Air Flow)',step=0.01,format='%2f')

if st.button('Predict'):
    import numpy as np
    X=[a,b,c,d,e,f,g,h,i,j,k,l]
    features=np.array([X])
    X=scale.transform(features)
    Y_pred=svc.predict(X)[0]
    print(Y_pred)
    if Y_pred==1:
        st.image('xtick.jpg')
        st.write('Preventive maintenance is required for the jet engine')
        st.write('30 or less cycles are left before damage of engine')
    else:
        st.image('ytick.png')
        st.write('Preventive maintenance is not required for the jet engine') 
        st.write('more than 30 cycles are left before damage of engine')