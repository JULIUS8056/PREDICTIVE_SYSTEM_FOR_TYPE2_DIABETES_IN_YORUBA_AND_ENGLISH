from pyexpat import model
import numpy as np
import pickle
from catboost import CatBoostClassifier
import streamlit as st
st.title(" Ẹrọ Ayẹwo fun Aisan ito suga iru Keji".upper())
model=pickle.load(open('/home/julius/python folder/Type-2-prediction.pkl','rb'))
with open("design.css") as ds:
    st.markdown(f"<style>{ds.read()}</style>", unsafe_allow_html=True)
def main():
    gender=''
    fhd=''
    fhd=''
    fhh=''
    heu=''
    hew=''
    rex=''
    hef=''
    phd=''
    fbs=st.number_input("Iwọn suga ninu ẹjẹ (Blood sugar)".upper())    
    bmiy=st.number_input("Atọka Ibi Ara (BMI)".upper())
    bmi=bmiy
    waisty=st.number_input("Ayika ẹgbẹ-ikun".upper())
    waist=waisty
    agey=st.number_input("Ọjọ ori".upper())
    age=agey
    gendery=st.selectbox("iwa (ọkunrin tabi obinrin?)".upper(),("ọkunrin","obinrin"))
    if(gendery=="ọkunrin"):
        gender='M'
    else:
         gender='F'
    fhdy=st.selectbox("Itan idile nipa aisan ito suga (Bẹẹni tabi Bẹẹkọ ?)".upper(),("Bẹẹni","Bẹẹkọ"))
    if(fhdy=='Bẹẹni'):
      fhd='Y'
    else:
      fhd='N' 
    fhhy=st.selectbox("Itan idile nipa Arun Ẹjẹ riru (Bẹẹni tabi Bẹẹkọ ?)".upper(),("Bẹẹni","Bẹẹkọ"))
    if(fhhy=='Bẹẹni'):
      fhh='Y'
    else:
      fhh='N' 
    
    heuy=st.selectbox("Itan ti ito loorekoore (Bẹẹni tabi Bẹẹkọ ?)".upper(),("Bẹẹni","Bẹẹkọ"))
    if(heuy=='Bẹẹni'):
      heu='Y'
    else:
      heu='N' 
    hewy=st.selectbox("Itan-akọọlẹ mimu omi loorekoore (Bẹẹni tabi Bẹẹkọ ?)".upper(),("Bẹẹni","Bẹẹkọ"))
    if(hewy=='Bẹẹni'):
      hew='Y'
    else:
      hew='N' 
    regexcercisey=st.selectbox("Ṣe o ṣe ere idaraya nigbagbogbo? (Bẹẹni tabi Bẹẹkọ ?)".upper(),("Bẹẹni","Bẹẹkọ"))
    if(regexcercisey=='Bẹẹni'):
      rex='Y'
    else:
      rex='N' 
    hefy=st.selectbox("Itan-akọọlẹ ti jijẹ ounjẹ ju bi o ṣe yẹ lọ (Bẹẹni tabi Bẹẹkọ ?)".upper(),("Bẹẹni","Bẹẹkọ"))
    if(fhdy=='Bẹẹni'):
      hef='Y'
    else:
      hef='N' 
    phdy=st.selectbox("Njẹ o ni iru arun Aisan ito suga miiran ṣaaju akoko yii (Bẹẹni tabi Bẹẹkọ ?)".upper(),("Bẹẹni","Bẹẹkọ"))
    if(phdy=='Bẹẹni'):
      phd='Y'
    else:
      phd='N' 
    makeprediction=[]
    if st.button("Predict"): 
        makeprediction=model.predict(
          [[fbs,bmi,waist,age,gender,fhd,fhh,heu,hew,rex,hef,phd]]
          )
    
#    result=''
    if makeprediction == 'Y':
            st.success("AISAN ITO SUGA WA LARA RE, YARA LO RI DOKITA ONISEGUN OYINBO NI KIAKIA".upper())
    else:
                st.success("O ko ni AISAN ITO SUGA, ṣugbọn rii daju pe o jẹ ounjẹ iwontunwonsi".upper())


if __name__=='__main__':
  main()
