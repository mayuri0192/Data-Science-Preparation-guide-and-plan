# diabetes_app.py

# imports
import pandas as pd
import joblib
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
# Create virtual environment:
# python3 -m venv venv
# In your virtual environment:
# python3 -m pip install --upgrade pip
# python -m pip install -U autopep8
# pip install pandas
# pip install joblib
# pip install streamlit
# pip install scikit-learn
# streamlit run diabetes_app.py

# Header
st.write("""
# Diabetes Risk Prediction App
Answer 6 questions to find out if you may be at risk for Type II Diabetes.
""")

with st.expander("Click for FAQ:"):
    st.write("""
    * **Risk Factors**:
        1. Weight
        2. Height
        3. Age
        4. High Cholesterol
        5. High Blood Pressure
        6. General Health
        * Weight and Height are utilized to calculate BMI for model input.
    * **How does this tool work?**
        * Our model is trained using data from the CDC's Behavioral Risk Factor Surveillance System. It employs L1 Regularized Logistic Regression and a Decision Tree, a simple and interpretable machine learning technique. 
        * Decision Trees generate prediction rules based on past observations.
        * Specifically, our model predicts the risk of Type II Diabetes or existing Type II Diabetes.
        * It's important to emphasize that this tool **DOES NOT OFFER A DIAGNOSIS**. Instead, it serves as a **public awareness tool** to promote healthier lifestyles and encourage individuals at risk of Type II Diabetes or with prediabetes to seek professional medical guidance.
    * **About the Behavioral Risk Factor Surveillance System (BRFSS)**: [Refer to dataset here](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf) 
        * BRFSS conducts health-related telephone surveys to gather data on health behaviors, chronic conditions, and preventive services usage among U.S. residents.
        * Established in 1984, BRFSS now operates in all 50 states, the District of Columbia, and three U.S. territories.
        * It conducts over 400,000 adult interviews annually, making it the world's largest continuously conducted health survey system.
    * **Model Performance**:
        * Based on over 115,000 survey responses used for training and 5-fold cross-validation:
        * Accuracy: 70% (+/- 2%)
        * Recall (Sensitivity): 75% (+/- 2%)
        * Precision: 69% (+/- 2%)
        * The model prioritizes Recall to capture as many potential positives as possible, aligning with the tool's public awareness objective.
    * **Promising Metrics, Keeping in Mind CDC Estimates** - [Source](https://www.cdc.gov/diabetes/library/spotlights/diabetes-facts-stats.html):
        * Approximately 1 in 10 Americans (37.3 million) has diabetes.
            * Alarmingly, around 1 in 5 diabetes cases remain undiagnosed.
        * Over 1 in 3 American adults (96 million) have prediabetes.
            * Shockingly, more than 8 in 10 adults with prediabetes are unaware of their condition.
    """)

with st.expander("Click to see the Decision Tree:"):
    st.write("""This is how the Diabetes risk prediction is made by this app.""")
    st.image('./Decision Tree Rules.png')

st.write("### Answer the following 6 Questions:")

# create the colums to hold user inputs
col1, col2, col3 = st.columns(3)

# gather user inputs

# 1. Weight
weight = col1.number_input(
    '1. Enter your Weight (lbs)', min_value=50, max_value=999, value=190)

# 2. Height
height = col2.number_input(
    '2. Enter your Height (inches): ', min_value=36, max_value=95, value=68)

# 3. Age
age = col3.selectbox(
    '3. Select your Age:', ('Age 18 to 24',
                            'Age 25 to 29',
                            'Age 30 to 34',
                            'Age 35 to 39',
                            'Age 40 to 44',
                            'Age 45 to 49',
                            'Age 50 to 54',
                            'Age 55 to 59',
                            'Age 60 to 64',
                            'Age 65 to 69',
                            'Age 70 to 74',
                            'Age 75 to 79',
                            'Age 80 or older'), index=4)

# 4. HighChol
highchol = col1.selectbox(
    "4. High Cholesterol: Have you EVER been told by a doctor, nurse or other health professional that your Blood Cholesterol is high?",
    ('Yes', 'No'), index=1)

# 5. HighBP
highbp = col2.selectbox(
    "5. High Blood Pressure: Have you EVER been told by a doctor, nurse or other health professional that you have high Blood Pressure?",
    ('Yes', 'No'), index=0)

# 6. Length of time since last routine checkup
checkup = col3.selectbox("6. Health Care Access: About how long has it been since you last visited a doctor for a routine checkup? [A routine checkup is a general physical exam, not an exam for a specific injury, illness, or condition.] scale from 1 = Within past year (anytime less than 12 months ago) to 5 = 5 or more years ago, or Never? ",
                         ('1', '2', '3', '4', '5', 'Never'), index=3)

# Smoker
smoker = col1.selectbox("How regularly do you smoke? ", ('Everyday smoker','Someday smoker','Former smoker','Non-smoker'))
# 8. GenHlth
genhlth = col2.selectbox("6. General Health: How would you rank your General Health on a scale from 1 = Excellent to 5 = Poor? Consider physical and mental health.",
                         ('Excellent', 'Very Good', 'Good', 'Fair', 'Poor'), index=3)


# Create dataframe:
df1 = pd.DataFrame([[round(weight), round(height), age, highchol, highbp, checkup, genhlth]], columns=[
                   'Weight', 'Height', 'Age', 'HighChol', 'HighBP', 'Checkup', 'GenHlth'])


def calculate_bmi(weight, height):
    """
    Calculate BMI from weight in lbs and height in inches.
    Args:
        weight: the weight in lbs
        height: the height in inches

    Returns:
        bmi - the body mass index

    """
    bmi = round((703 * weight)/(height**2))

    return bmi


def prep_df(df):
    """Prepare user .

    Args:
        df: the dataframe containing the 6 user inputs.

    Returns:
        the dataframe with 5 outputs. BMI, Age, HighChol, HighBP, and GenHlth

    """
    # BMI
    df['BMI'] = df.apply(lambda row: calculate_bmi(
        row['Weight'], row['Height']), axis=1)

    # Drop Weight and Height
    df = df.drop(columns=['Weight', 'Height'])

    # Re-Order columns
    df = df[['BMI', 'Age', 'HighChol', 'HighBP', 'GenHlth']]

    # Age
    df['Age'] = df['Age'].replace({'Age 18 to 24': 1, 'Age 25 to 29': 2, 'Age 30 to 34': 3, 'Age 35 to 39': 4, 'Age 40 to 44': 5, 'Age 45 to 49': 6,
                                   'Age 50 to 54': 7, 'Age 55 to 59': 8, 'Age 60 to 64': 9, 'Age 65 to 69': 10, 'Age 70 to 74': 11, 'Age 75 to 79': 12, 'Age 80 or older': 13})
    # HighChol
    df['HighChol'] = df['HighChol'].replace({'Yes': 1, 'No': 0})
    # HighBP
    df['HighBP'] = df['HighBP'].replace({'Yes': 1, 'No': 0})
    # GenHlth
    df['GenHlth'] = df['GenHlth'].replace(
        {'Excellent': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5})

    return df


# prepare the user inputs for the model to accept
df = prep_df(df1)

with st.expander("Click to see user inputs"):
    st.write("**User Inputs** ", df1)
with st.expander("Click to see what goes into the Decision Tree for prediction"):
    st.write("**User Inputs Prepared for Decision Tree** ", df,
             "** Note that BMI is calculated from the Weight and Height you entered. Age has 14 categories from 1 to 13 in steps of 5 years. HighChol and HighBP are 0 for No and 1 for Yes. GenHlth is on a scale from 1=Excellent to 5=Poor. These come directly from BRFSS questions the model learned from.")

# load in the model
model = joblib.load('./dt_model.pkl')

# Make the prediction:
if st.button('Click here to predict your Type II Diabetes Risk'):

    # make the predictions
    prediction = model.predict(df)
    prediction_probability = model.predict_proba(df)
    low_risk_proba = round(prediction_probability[0][0] * 100)
    high_risk_proba = round(prediction_probability[0][1] * 100)

    if(prediction[0] == 0):
        st.write("You are at **low-risk** for Type II Diabetes or prediabetes")
        st.write("Predicted probality of low-risk",
                 low_risk_proba, "%")
        st.write("Predicted probality of high-risk",
                 high_risk_proba, "%")
    else:
        st.write("You are at **high-risk** for Type II Diabetes or prediabetes")
        st.write("Predicted probality of low-risk",
                 low_risk_proba, "%")
        st.write("Predicted probality of high-risk",
                 high_risk_proba, "%")
        st.write(
            "Consider taking the [CDC - Prediabetes Risk Test](https://www.cdc.gov/prediabetes/risktest/)")
        st.write(
            "Get started on your path to preventing type 2 diabetes here: [CDC - Path 2 Prevention](https://diabetespath2prevention.cdc.gov)")
        st.write(
            "Consider enrolling in the National Diabetes Prevention Program, through providers like: [Lark Health](https://www.lark.com).")
