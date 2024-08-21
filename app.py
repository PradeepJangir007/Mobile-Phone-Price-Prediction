import streamlit as st
import pandas as pd
import numpy as np

import pickle
df=pickle.load(open('df.pkl','rb'))
model=pickle.load(open('samart_phone_price.pkl','rb'))
st.title('smart_phone_price_prediction')
st.subheader('give some information about smart phone')
st.markdown('Note: yes = 1 ,No = 0')

# brand
Brand=st.selectbox('Brand',df.company.unique())
Dual_sim=st.selectbox('Dual sim',['1','0'])
VoLTE=st.selectbox('VoLTE',['1','0'])
G=st.selectbox('5G',['1','0'])
Vo5G=st.selectbox('Vo5G',['1','0'])
Foldable_Display=st.selectbox('Foldable Display',['1','0'])
Dual_Displayd=st.selectbox('Dual Display',['1','0'])
Water_Drop_Notch=st.selectbox('Water Drop Notch',['1','0'])
with_Punch_Hole=st.selectbox('with Punch Hole',['1','0'])
Processor=st.selectbox('Processor',df.Processor_.unique())
disply=st.text_input('Display resulation')
size=st.text_input('Display size')
Ram=st.text_input('Ram','GB')
External_Memory=st.text_input('External_Memory','GB')
Inbuilt_memory=st.text_input('Inbuilt_memory','GB')
fast_charging=st.text_input('fast_charging','Watt')
Battery=st.text_input('Battery capacity','mah')
No_of_Rear=st.text_input('No of Rear')
No_of_Front=st.text_input('No of Front')
Primary_rear_camera=st.text_input('Primary_rear_camera')
Primary_front_camera=st.text_input('Primary_front_camera')
Number_of_core	=st.text_input('Number_of_core')
b=st.button('predict price')
if b==True:
    v=int(str(disply).split('*')[0])
    h=int(str(disply).split('*')[1])
    pixal_D=(v**2+h**2)**0.5
    PPI=((v**2+h**2)/float(size))**0.5
    x={
        'Dual Sim':[int(Dual_sim)],'VoLTE':[int(VoLTE)],'5G': [int(G)],
        'Vo5G':[int(Vo5G)],'Ram':[int(Ram)],'Battery':[int(Battery)],'Foldable Display':[int(Foldable_Display)],
        'Dual Display':[int(Dual_Displayd)],'External_Memory':[int(External_Memory)],
        'company':[Brand],'Inbuilt_memory':[int(Inbuilt_memory)],'fast_charging':[float(fast_charging)],
        'Water Drop Notch':[int(Water_Drop_Notch)],'with Punch Hole':[int(with_Punch_Hole)],
        'Processor_':[Processor],'No _of_Rear':No_of_Rear,'No _of_Front':[int(No_of_Front)],
        'Primary_rear_camera':[Primary_rear_camera],'Primary_front_camera':[Primary_front_camera],
        'Number_of_core':[Number_of_core],
        'pixal_D':[pixal_D],'PPI':[PPI],
    }
    x=pd.DataFrame(x)
    print(x)
    st.title('price ' +str(round(10**((model.predict(x))[0]))))
