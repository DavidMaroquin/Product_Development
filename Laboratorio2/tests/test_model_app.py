import streamlit as st
import pandas as pd
from pycaret.regression import load_model
from pycaret.regression import predict_model

data_path = '../data/raw/data_traffic_v1.csv'
dataset=pd.read_csv(data_path)
model = load_model('../models/model_v1')

def main():
    st.title("House Pricess Model")

    Rank=st.slider("Rank",min_value=dataset["Rank"].min(), max_value=dataset['Rank'].max())
    TeamId=st.slider("TeamId",min_value=dataset["TeamId"].min(), max_value=dataset['TeamId'].max())
    Score=st.slider("Score",min_value=dataset["Score"].min(), max_value=dataset['Score'].max())
    SubmissionCount=st.slider("SubmissionCount",min_value=dataset["SubmissionCount"].min(), max_value=dataset['SubmissionCount'].max())

    #Rush_Hour=st.select_slider("Rush Hour", options=list(dataset["Rush Hour"].unique()))

    TeamName = st.selectbox("TeamName?", options=list(dataset['TeamName'].unique()))
    TeamMemberUserNames = st.selectbox("TeamMemberUserNames", options=list(dataset['TeamMemberUserNames'].unique()))

    get_pred = st.button("Predecir")
    if(get_pred):
        data_to_predic = pd.DataFrame({'Rank':[Rank],
                                       'TeamId': [TeamId],
                                       'Score': [Score],
                                       'SubmissionCount':[SubmissionCount],
                                       'TeamName': [TeamName],
                                       'TeamMemberUserNames': [TeamMemberUserNames]})
        predicciones = predict_model(model, data=data_to_predic)
        print(predicciones)
        valor_predicho = round(list(predicciones['prediction_label'])[0],4)
        st.success(f"Valor Predicho: {valor_predicho}")

if(__name__=='__main__'):
    main()
