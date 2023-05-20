import streamlit as st
import pandas as pd

################################  PAGE CONTENT  ################################

st.set_page_config(
    page_title="🎈Wecome to JZ's BMI Prediction Site📷",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<h1 style='text-align: center; color: #B92708;'>BMI Prediction</h1>", unsafe_allow_html=True)

st.caption('This site only gives you BMI prediction. We do not take responsibility for providing accurate and credible BMI results. Thank you!')
st.caption('All rights reserved by Jason Zhu @ UChicago. If you have any questions, please contact jasonzhu@uchicago.edu')

cal, ref = st.columns([2, 1])

cal.subheader("🔎Calculate Your BMI")
with cal:
    height = st.number_input('Your Height in CM')
    weight = st.number_input('Your Weight in KG')

    if st.button('Predict My BMI'):
        if height>0 and weight>0:
            bmi = weight/((height/100)**2)
            st.markdown(f'Your BMI from your height and weight is: **{round(bmi,2)}**')
            if bmi<18.5:
                st.write('**Sorry you are UNDERWEIGHT. Eat More!!🥩**')
            elif 18.5<=bmi<=25:
                st.write('**Hurray! Your BMI looks good! Keep Going!💪**')
            elif 25<bmi<30:
                st.write('**Sorry you are OVERWEIGHT! Be careful about your diet.🥦**')
            elif 30<=bmi<35:
                st.write('**Hey, You are MODERATELY OBESE. Eat healthy and exercise more please.🥗**')
            elif 35<=bmi<=40:
                st.write('**Oh no! You are SEVERELY OBESE. Please eat healthy and exercise more.🏃**')
            elif bmi>40:
                st.write('**Watch out! You are VERY SEVERELY OBESE. Please reach out your doctor for professional advice on your health.😞**')
        else:
            st.markdown('Please enter valid numbers.')

ref.subheader("BMI Reference")
df = pd.DataFrame({'BMI':['below 18.5', '18.5 - 25', '25 - 30', '30 - 35', '35 - 40', 'above 40'],
                   'WEIGHT STATUS':['Underweight', 'Normal', 'Overweight', 'Moderately obese', 'Severely obese', 'Very severely obese']}).reset_index(drop=True)
with ref:
    st.table(df)

st.write(' ')
st.write('👈👈👈Or you can predict your BMI from your face using three ways from the sidebar on the left.😊')

hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)