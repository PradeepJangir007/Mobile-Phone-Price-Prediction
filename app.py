import streamlit as st
import pandas as pd
import pickle
import sys

import my_module
from my_module import SparseToDenseTransformer,LogStandardScaler,confidence_
sys.modules['__main__'].SparseToDenseTransformer = SparseToDenseTransformer
sys.modules['__main__'].confidence_ = confidence_

df=pickle.load(open('df.pkl','rb'))
model=pickle.load(open('smart_phone_price_with_PLSR.pkl','rb'))
st.title('smart_phone_price_prediction')
st.subheader('give some information about smart phone')
st.markdown('Note: yes = 1 ,No = 0')

# brand
Brand=st.selectbox('Brand',df.company.unique())

model_series=st.selectbox('model_serie',df[df['company']==Brand].Model_series.unique())
Dual_sim=st.selectbox('Dual sim',['1','0'])
VoLTE=st.selectbox('VoLTE',['1','0'])
G=st.selectbox('5G',['1','0'])
Vo5G=st.selectbox('Vo5G',['1','0'])
Foldable_Display=st.selectbox('Foldable Display',['1','0'])
Dual_Displayd=st.selectbox('Dual Display',['1','0'])

Processor=st.selectbox('Processor',df.Processor_.unique())
disply=st.text_input('Display resulation')
size=st.text_input('Display size')
Ram=st.text_input('Ram','GB')
Processor_series=st.text_input('Processor_series')
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
    Processor_series=my_module.Processor_s.Transformer(Processor,Processor_series)
    x={'Model_series':[model_series],'Dual Sim':[int(Dual_sim)],'VoLTE':[int(VoLTE)],'5G': [int(G)],
        'Vo5G':[int(Vo5G)],'Ram':[int(Ram)],'Battery':[int(Battery)],'Foldable Display':[int(Foldable_Display)],
        'Dual Display':[int(Dual_Displayd)],'External_Memory':[int(External_Memory)],
        'company':[Brand],'Inbuilt_memory':[int(Inbuilt_memory)],'fast_charging':[float(fast_charging)],
        'Processor_':[Processor],'Processor_series':[int(Processor_series)],'No _of_Rear':No_of_Rear,'No _of_Front':[int(No_of_Front)],
        'Primary_rear_camera':[Primary_rear_camera],'Primary_front_camera':[Primary_front_camera],
        'Number_of_core':[Number_of_core],
        'PPI':[PPI]}
    x=pd.DataFrame(x)
    st.title('price ' +str(round(2.7183**((model.predict(x))[0]))))
    conf=confidence_()
    L=conf.interval(x)[0]
    U=conf.interval(x)[1]
    st.subheader(f'confidence interval with 90% \n {(round(2.7183 ** L), round(2.7183 ** U))}')
