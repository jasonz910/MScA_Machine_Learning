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

st.write('BMI Reference')
index = {'BMI':['16 ~ 18.5', '18.5 ~ 25', '25 ~ 30', '30 ~ 35', '35 ~ 40', '40~'],
        'WEIGHT STATUS':['Underweight', 'Normal', 'Overweight', 'Moderately obese', 'Severely obese', 'Very severely obese']}
df = pd.DataFrame(data=index)
st.table(df)

st.write('You can calculate your BMI based on your height and weight Below:')

height = st.number_input('Your Height in CM')
weight = st.number_input('Your Weight in KG')

if st.button('Predict My BMI'):
    if height>0 and weight>0:
        bmi = weight/((height/100)**2)
        st.write('Your BMI from your height and weight is:', bmi)
        if bmi<18.5:
            st.write('Sorry you are underweight. Eat More!!')
        elif 18.5<=bmi<=25:
            st.write('Hurray! Your BMI looks good! Keep Going!')
        elif 25<bmi<30:
            st.write('Sorry you are overweight! Be care about your diet.')
        elif bmi>30:
            st.write('Watch out! You reach the level of obesity. Try to eat less and exercise more.')
    else:
        st.write('Please enter valid numbers! 🙏🏻')

st.write(' ')
st.write(' ')
st.write('👈👈👈Or you can predict your BMI from your face using three ways from the sidebar left :)')