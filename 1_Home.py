import streamlit as st

# ###############################

st.set_page_config(
    page_title="🎈Wecome to JZ's BMI Prediction📷",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎈Wecome to JZ's BMI Prediction📷")

st.caption('This site only give you BMI prediction. This site does not take responsibility for providing accurate and credible BMI results. Thank you! 😊')

st.write('You can calculate your BMI based on your height and weight Below:')

height = st.number_input('Your Height in CM')
weight = st.number_input('Your Weight in KG')

if st.button('Predict My BMI'):
    if height>0 and weight>0:
        bmi = weight/((height/100)**2)
        st.write('Your BMI from your height and weight is:', bmi)
    else:
        st.write('Please enter valid numbers! 🙏🏻')


st.write('\n\nOr you can predict your BMI from your face using three ways from the sidebar left :)')