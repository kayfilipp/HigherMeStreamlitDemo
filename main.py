import streamlit as st
import pandas as pd
import numpy as np
import modeling_utils

st.header("HigherME Stem Placement Prediction App")
st.write("This MVP provides demo functionality to see if an individual is likely to land a job in a STEM field.")

intro_screen = st.empty()
intro_screen.text("Loading Random Forest Model...")

#see modeling_utils.py => check our session state for an RF model and reload it if we're missing one.
#this reduces processing time.
modeling_utils.main(st)

intro_screen.empty()

#load data.csv
modeling_utils.main_data(st)
mu = st.session_state['data']["AGE"].mean()
sigma = st.session_state['data']["AGE"].std()

#to ensure consistency with training data, we need to sort this alphabetically
edus = np.array(st.session_state['data']["EDU_verbose"].unique())
edus.sort()
edus = list(edus)

degree_areas = np.array(st.session_state['data']["STEM_Degree_Area"].unique())
degree_areas.sort()
degree_areas = list(degree_areas)

if st.checkbox('Show Training Dataframe sample'):
    st.session_state['data'][0:10]

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

    inp_under_represented = st.radio(
        'Is the individual from an under-represented\n racial background?',
        ['yes','no']
    )

with right_column:
    inp_edu_level = st.selectbox(
        'What is the highest educational attainment for the individual?',
        edus
    )

    if inp_edu_level not in ['None/Below HS', 'High School']:
        inp_is_STEM = st.radio(
            'Does the Individual have a STEM degree?',
            ['yes','no'],
            index=1
        )
    else:
        inp_is_STEM = 'no'

    if inp_is_STEM == 'yes' and inp_edu_level not in ['None/Below HS', 'High School']:
        inp_deg_area = st.selectbox(
            "STEM Degree Concentration"
            , degree_areas
        )
    else:
        inp_deg_area = 'None'





inp_age = st.slider(
    'Age (Yrs)'
    , min_value= min(st.session_state['data']["AGE"])
    , max_value= max(st.session_state['data']["AGE"])
    , step=1
)



if st.button('Make Prediction'):

    #make an empty dictionary object that we'll turn into a dataframe for prediction.
    instance = {}

    #binarize all other inputs
    instance["SEX"] = [modeling_utils.converter_func(inp_sex)]

    #Scale age
    inp_age = (inp_age - mu) / sigma

    instance["AGE"] = [inp_age]
    instance["domestic_born"] = [modeling_utils.converter_func(inp_domestic_born)]
    instance["is_STEM_degree"] = [modeling_utils.converter_func(inp_is_STEM)]
    instance["under_represented"] = [modeling_utils.converter_func(inp_under_represented)]


    #Dummies for degree area
    for deg in degree_areas:
        this_key = f"STEM_Degree_Area_{deg}"
        if inp_deg_area == deg:
            instance[this_key] = [1]
        else:
            instance[this_key] = [0]

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

    result = st.session_state['model_rf'].predict_proba(predict)
    nostem = round(result[0][0] * 100)
    yesstem = round(result[0][1] * 100)
    if nostem > yesstem:
        st.text(f"Prediction: Likely Not working in STEM, Chance: {nostem}%")
    else:
        st.text(f"Prediction: Likely Working in STEM, Chance: {yesstem}%")




