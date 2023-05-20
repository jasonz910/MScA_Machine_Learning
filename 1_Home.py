import streamlit as st
import pandas as pd

# ###############################

st.set_page_config(
    page_title="🎈Wecome to JZ's BMI Prediction📷",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎈Wecome to JZ's BMI Prediction📷")

st.caption('This site only give you BMI prediction. This site does not take responsibility for providing accurate and credible BMI results. Thank you! 😊')

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

df = pd.DataFrame({'BMI':['16 ~ 18.5', '18.5 ~ 25', '25 ~ 30', '30 ~ 35', '35 ~ 40', '40~'],
                   'WEIGHT STATUS':['Underweight', 'Normal', 'Overweight', 'Moderately obese', 'Severely obese', 'Very severely obese']})

with ref:
    st.table(df)

st.write(' ')
st.write(' ')
st.write('👈👈👈Or you can predict your BMI from your face using three ways from the sidebar left :)')


st.markdown("""
    <style>
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    color: #B92708;
    text-align: center;
    }
    </style>
    <div class="footer">
    <p>All rights reserved by Jason Zhu. Only use for MSCA 31009 Machine Learning & Predictive Analytics Class @ UChicago.</p>
    </div>
    """, unsafe_allow_html=True)