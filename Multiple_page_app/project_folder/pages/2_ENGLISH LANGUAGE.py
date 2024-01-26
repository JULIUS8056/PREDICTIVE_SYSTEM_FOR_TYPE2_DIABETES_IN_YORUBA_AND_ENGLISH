import numpy as np
import pickle
from catboost import CatBoostClassifier
import streamlit as st

model=pickle.load(open('/home/julius/python folder/Type-2-prediction.pkl','rb'))
#global makeprediction
with open("design.css") as ds:
    st.markdown(f"<style>{ds.read()}</style>", unsafe_allow_html=True)
def main():
    with open('design.css') as f:
        #st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
            


#title of the app
        st.title("Clinical Decision Support System for Prediction of Type 2 Diabetes".upper())
#input setup
    
    fbs=st.text_input("Blood Sugar level ")
    bmi=st.text_input("Body Mass Index")
    waist=st.text_input("Waist Circumference")
    age=st.text_input("Age")
    gender=st.text_input("Gender","M/F ?")
    fhd=st.text_input("Family History of Diabetes","Y/N ?")
    fhh=st.text_input("Family History of Hypertension","Y/N ?")
    heu=st.text_input("History of Excess Urine","Y/N ?")
    hew=st.text_input("History of Excess Water Intake","Y/N ?")
    rex=st.text_input("Did you do excercise regularly","Y/N ?")
    hef=st.text_input("History of Excess Food Intake","Y/N ?")
    phd=st.text_input("Did you have other type of diabetes before","Y/N ?")
#action button
    makeprediction=[]
    if st.button("Predict"):
        makeprediction=model.predict([[fbs,bmi,waist,age,gender,fhd,fhh,heu,hew,rex,hef,phd]])
        result=''
        if makeprediction == 'Y':
            st.success("You either have diabetes or are likely to have it. Please visit the doctor as soon as possible.".upper())  
        else:
                st.success('You do not have Diabetes'.upper())
        #st.success(result)
#display of result
    #st.success("Your result shows that you {}".format(makeprediction))

if __name__=='__main__':
  main()
