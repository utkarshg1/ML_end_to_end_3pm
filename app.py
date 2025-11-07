# In order to run streamlit app use below command
# streamlit run app.py
# Import necessary libraries
import streamlit as st
import joblib
import pandas as pd

# Intialize blank webpage
st.set_page_config(page_title="Iris end to end project")


# Load the model object using caching
# caching will store the model object in temporary memory
@st.cache_resource
def load_model(path: str = "notebook/iris_model.joblib"):
    return joblib.load(path)


# Load the model object
model = load_model()


# Write a function to perform prediction along with probability
def predict_species(
    model, sep_len: float, sep_wid: float, pet_len: float, pet_wid: float
) -> tuple:
    data = [
        {
            "sepal_length": sep_len,
            "sepal_width": sep_wid,
            "petal_length": pet_len,
            "petal_width": pet_wid,
        }
    ]
    xnew = pd.DataFrame(data)
    pred = model.predict(xnew)
    probs = model.predict_proba(xnew)
    df_probs = pd.DataFrame(probs, columns=model.classes_)
    return pred, df_probs


# Start building streamlit app
st.title("Iris End to End Deployment Project")
st.subheader("by Utkarsh Gaikwad")

# Take sep lengh, sep width, petal length and petal width as input from user
sep_len = st.number_input("Sepal Length : ", min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width : ", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length : ", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width : ", min_value=0.00, step=0.01)

# Create a button to perform prediction
button = st.button("Predict", type="primary")

# if button is pressed
if button:
    pred, prob = predict_species(model, sep_len, sep_wid, pet_len, pet_wid)
    st.subheader(f"Predicted Species : {pred[0]}")
    st.subheader("Probability : ")
    st.dataframe(prob)
    st.bar_chart(prob.T)
