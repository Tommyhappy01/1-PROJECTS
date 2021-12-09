#st.markdown(html_temp,unsafe_allow_html=True)
#
import streamlit as st
import pickle
import pandas as pd
import numpy as np


st.sidebar.title("Will Your Employee Leave?")
st.title('Employee Leave Prediction')
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML App </h3>
</div>"""

#st.markdown(html_temp,unsafe_allow_html=True)
#html_temp = """
#<div style="background-color:tomato;padding:10px">
#<h2 style="color:white;text-align:center;">Streamlit ML App </h3>
#</div>"""
#st.markdown(html_temp,unsafe_allow_html=True)
#

from PIL import Image
im = Image.open("churn.jpg")
st.image(im, width=699, caption=("Streamlit Churn Prediction App"))


gb_model = pickle.load(open("GradientBoosting.pkl","rb"))
knn_model = pickle.load(open("KNeighbors.pkl","rb"))
rf_model = pickle.load(open("RandomForest.pkl","rb"))


Satisfaction= st.sidebar.slider("Please rate the satisfaction of the employee from 0 to 1.", 0.01, 1.00, step=0.01)
Evaluation= st.sidebar.slider("Please rate the performance of the employee from 0 to 1.", 0.01, 1.00, step=0.01)
Projects= st.sidebar.selectbox("How many of projects assigned to the employee?", (2, 3, 4, 5, 6, 7))
Hours= st.sidebar.slider("How many hours in average did the employee work in a month?", 96, 310, step=1)
Years= st.sidebar.selectbox("How many years did the employee spend in the company ?", (2, 3, 4, 5, 6, 7, 8, 9, 10))


modell=st.selectbox("Select a Model", ("Gradient Boosting Classifier", "Random Forest Classifier", "KNN Classifier"))
st.write(f"You selected {modell} model.")

my_dict = {
    "satisfaction_level": Satisfaction,
    "last_evaluation": Evaluation,
    "number_project":Projects,
    "average_montly_hours":Hours,
    "time_spend_company":Years
}

df = pd.DataFrame.from_dict([my_dict])


if modell=="Gradient Boosting Classifier":
    prediction = gb_model.predict(df)
elif modell=="KNN Classifier":
    prediction = knn_model.predict(df)
else:
    prediction = rf_model.predict(df)
    

st.markdown("## The information about the employee is below")
st.write(df)

st.subheader("""Press 'Predict' if configuration is right""")


if st.button('Predict'):
    if prediction==0:
        st.success("The employee seems NOT TO CHURN.")
        st.balloons()
    else:
        st.warning("The employee seems to CHURN.")  
        st.error("You should take some precautions.") 



