import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import pickle
import requests

mLink = 'https://raw.githubusercontent.com/kayfilipp/HigherME/main/census_data/finalized_rf_model.pkl?raw=true'
mfile = BytesIO(requests.get(mLink).content)
model_rf = pickle.load(mfile)

st.header("HigherME Stem Placement Prediction App")
st.write("This MVP provides demo functionality to see if an individual is likely to land a job in a STEM field.")
data = pd.read_csv("https://raw.githubusercontent.com/kayfilipp/HigherME/main/census_data/streamlit/data.csv")
mu = data["AGE"].mean()
sigma = data["AGE"].std()

#to ensure consistency with training data, we need to sort this alphabetically
edus = np.array(data["EDU_verbose"].unique())
edus.sort()
edus = list(edus)

if st.checkbox('Show Training Dataframe sample'):
    data[0:10]

# load model - Random Forest
deployed_svm_model = 0

st.subheader("Please select relevant features for your individual.")
left_column, right_column = st.columns(2)
with left_column:
    inp_sex = st.radio(
        'What is the sex of the individual?',
        ['Male','Female'] #male=0,female=1
    )

    inp_domestic_born = st.radio(
        'Was the Individual born in the US?',
        ['yes','no']
    )

with right_column:
    inp_edu_level = st.radio(
        'What is the highest educational attainment for the individual?',
        edus
    )

    inp_is_STEM = st.radio(
        'Does the Individual have a STEM degree?',
        ['yes','no']
    )

with left_column:
    inp_under_represented = st.radio(
        'Is the individual from an under-represented\n racial background?',
        ['yes','no']
    )

def converter_func(radio):
    if radio=='yes' or radio=='Female':
        return 1
    return 0

inp_age = st.slider(
    'Age (Yrs)'
    , min_value= min(data["AGE"])
    , max_value= max(data["AGE"])
    , step=1
)

if st.button('Make Prediction'):

    #make an empty dictionary object that we'll turn into a dataframe for prediction.
    instance = {}



    #binarize all other inputs
    instance["SEX"] = [converter_func(inp_sex)]
    #Scale age
    inp_age = (inp_age - mu) / sigma
    instance["AGE"] = [inp_age]
    instance["domestic_born"] = [converter_func(inp_domestic_born)]
    instance["is_STEM_degree"] = [converter_func(inp_is_STEM)]
    instance["under_represented"] = [converter_func(inp_under_represented)]
    #Dummies for education
    for edu in edus:
        this_key = f"EDU_verbose_{edu}"
        if inp_edu_level == edu:
            instance[this_key] = [1]
        else:
            instance[this_key] = [0]

    predict = pd.DataFrame.from_dict(instance)
    st.text("Here's what your input data looks like:")
    predict

    result = model_rf.predict_proba(predict)
    nostem = round(result[0][0] * 100)
    yesstem = round(result[0][1] * 100)
    if nostem > yesstem:
        st.text(f"Prediction: Likely Not working in STEM, Chance: {nostem}%")
    else:
        st.text(f"Prediction: Likely Working in STEM, Chance: {yesstem}%")




