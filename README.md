# HigherMeStreamlitDemo
Demo for ML functionality of the HigherME project in streamlit - deploys a Random Forest Model to classify the probability of STEM employment for an individual with given demographic traits.
The URL for the web client can be found <a href="https://kayfilipp-highermestreamlitdemo-main-ji7d4t.streamlitapp.com/">here.</a>

![streamlit](https://user-images.githubusercontent.com/36943200/184756643-345e7b99-9b9d-4fb0-b58b-b17790009687.jpg)

### Overview of Assets

1. **main.py**: This script is the launching point for the streamlit app, and handles loading both the census dataset as well as the Random Forest model to be used
during the demo. The dataset is not needed for training, but does help us with encoding categorical variables, normalizing the age feature, and can be viewed by
the end user as a sample.

2. **Modeling_utils.py**: Main.py passes loading of both the data and the model to this module. The module has a waterfall for loading each object in order of checking the 
session state, then a local directory, and then loading the data from the github repository for HigherME, which can be found <a href="https://github.com/kayfilipp/HigherME">here.</a>

3. **Model/Rf_model.pkl**: A Pickle file containing the trained, deployable random forest model. We use this model to predict our test sample's likelihood of 
STEM employment.

4. **Census_data/data.csv**: A .csv file containing all of our training data. this dataset is served to the user as a sample, helps with normalization, one-hot encoding,
and feature preparation after the user submits their test sample. This logic is embedded into main.py

#### Further Documentation

This app was launched and built by reviewing the following article from <a href ="https://towardsdatascience.com/how-to-deploy-machine-learning-models-601f8c13ff45"> Towards Data Science.<a/>
