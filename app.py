import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Function to create a download link for a DataFrame as a CSV File
def get_binary_file_downloader_html(df):
    csv=df.to_csv(index=False)
    b64=base64.b64encode(csv.encode()).decode()
    href=f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href

st.title("Heart Disease Predictor")
tab1,tab2,tab3=st.tabs(['Predict','Bulk Predict','Model Information'])

with tab1:
    age=st.number_input("Age(years)",min_value=0,max_value=150)
    sex=st.selectbox("Sex",["Male","Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs to numeric
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
     })

    algonames=['Decision Trees','Logistic Regression','Random Forest','Support Vector Machine']
    modelnames=['decision_tree.pkl','logistic_model.pkl','random_forest.pkl','svm_model.pkl']

    predictions=[]
    def predict_heart_disease(data):
        for modelname in modelnames:
            model=pickle.load(open(modelname,'rb'))
            prediction=model.predict(data)
            predictions.append(prediction)
        return predictions
    # Create a submit button to make predictions
    if st.button("Submit"):
       st.subheader('Results...')
       st.markdown('--------------------------')

       result = predict_heart_disease(input_data)

       for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
            st.markdown('--------------------------')
with tab2:
    st.title("Upload CSV File")

    st.subheader('Instructions to note before uploading the file:')
    st.info("""
        1. No NaN values allowed.
        2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
        'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').
        3. Check the spellings of the feature names.
        4. Feature values conventions:
            - Age: age of the patient [years]
            - Sex: sex of the patient [0: Male, 1: Female]
            - ChestPainType: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic]
            - RestingBP: resting blood pressure [mm Hg]
            - Cholesterol: serum cholesterol [mm/dl]
            - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
            - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2: showing probable or definite left ventricular hypertrophy by Estes' criteria]
            - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
            - ExerciseAngina: exercise-induced angina [1: Yes, 0: No]
            - Oldpeak: oldpeak = ST [Numeric value measured in depression]
            - ST_Slope: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping]
""")
    # Create a file uploader in sidebar
    uploaded_file=st.file_uploader("Upload a CSV file",type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file into a dataframe
        input_data=pd.read_csv(uploaded_file)
        model=pickle.load(open('logistic_model.pkl','rb'))

        # Ensure that the input DataFrame matches the exprected columns and format
        expected_columns=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
        if set(expected_columns).issubset(input_data.columns):

            input_data['Prediction LR']=''

            for i in range(len(input_data)):
                arr=input_data.iloc[i,:-1].values
                input_data['Prediction LR'][i]=model.predict([arr])[0]
            input_data.to_csv('PredictedHeartLR.csv')

            # Display the Predictions
            st.subheader("Predictions:")
            st.write(input_data)

            # Create a button to download the updated CSV file
            st.markdown(get_binary_file_downloader_html(input_data),unsafe_allow_html=True)
        else:
            st.warning("Please make sure that the uploaded CSV file has the correct columns.")

    else:
        st.info("Upload a CSV file to get the predictions")

with tab3:
    import plotly.express as px
    data={'Decision Trees':80.97, 'Logistic Regression':85.86, 'Random Forest':84.24, 'Support Vector Machine':84.23}
    Models=list(data.keys())
    Accuracies=list(data.values())
    df=pd.DataFrame(list(zip(Models,Accuracies)),columns=['Models','Accuracies'])
    fig= px.bar(df,y='Accuracies',x='Models')
    st.plotly_chart(fig)


    








