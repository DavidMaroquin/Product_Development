import streamlit as st
import pandas as pd
from pycaret.regression import load_model
from pycaret.regression import predict_model


data_path = '../data/raw/data_traffic_v1.csv'
dataset=pd.read_csv(data_path)
model = load_model('../models/model_v1')

def main():
    st.title("videojuegos")

    anio=st.slider("anio",min_value=dataset["anio"].min(), max_value=dataset["anio"].max())
    ventasNA=st.slider("ventasNA",min_value=dataset["ventasNA"].min(), max_value=dataset['ventasNA'].max())
    ventasEU=st.slider("ventasEU",min_value=dataset["ventasEU"].min(), max_value=dataset['ventasEU'].max())
    ventasJP=st.slider("ventasJP",min_value=dataset["ventasJP"].min(), max_value=dataset['ventasJP'].max())
    ventasOtros=st.slider("ventasOtros",min_value=dataset["ventasOtros"].min(), max_value=dataset['ventasOtros'].max())

    #Rush_Hour=st.select_slider("Rush Hour", options=list(dataset["Rush Hour"].unique()))

    nombre = st.selectbox("nombre", options=list(dataset['nombre'].unique()))
    plataforma = st.selectbox("plataforma", options=list(dataset['plataforma'].unique()))
    genero = st.selectbox("genero", options=list(dataset['genero'].unique()))
    editorial = st.selectbox("editorial", options=list(dataset['editorial'].unique()))

    get_pred = st.button("Predecir")
    
    if(get_pred):
        data_to_predic = pd.DataFrame({'anio':[anio],
                                       'ventasNA': [ventasNA],
                                       'ventasEU': [ventasEU],
                                       'ventasJP':[ventasJP],
                                       'ventasOtros': [ventasOtros],
                                       'nombre': [nombre],
                                       'plataforma':[plataforma],
                                       'genero':[genero],
                                       'editorial':[editorial]})
        predicciones = predict_model(model, data=data_to_predic)
        print(predicciones)
        valor_predicho = round(list(predicciones['prediction_label'])[0],4)
        st.success(f"Valor Predicho: {valor_predicho}")

if(__name__=='__main__'):
    main()