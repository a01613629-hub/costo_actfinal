import numpy as np
import streamlit as st
import pandas as pd


st.write("# Predicción de costo de una actividad")
st.image("fotocosto.jpg", caption="Predicción del costo de una actividad.")

st.header("Datos de la actividad")

def user_input_features():


    Presupuesto=st.number_input("Presupuesto",min_value=1,max_value=3000,value=1,step=1, )

    Tiempo=st.number_input("Tiempo invertido:",min_value=0,max_value=1440,value=0,step=1,)
    
    Tipo=st.number_input("Tipo",min_value=0,max_value=5,value=1,step=1,)

    Momento=st.number_input("Momento",min_value=0,max_value=2,value=1,step=1,)

    Personas=st.number_input("No. de personas",min_value=0,max_value=20,value=1,step=1,)



    user_input_data = {
        "Presupuesto": Presupuesto,
        "Tiempo invertido": Tiempo,
        "Tipo": Tipo,
        "Momento": Momento,
        "No. de personas": Personas,
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

datos = pd.read_csv("colab_costo.csv", encoding="latin-1")
X= datos.drop("Costo", axis=1)
y= datos["Costo"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613629)

LR=LinearRegression()
LR.fit(X_train,y_train)
modelo = LinearRegression()
modelo.fit(X_train, y_train)

prediccion = b0 + b1[0]*100 + b1[1]*100 + b1[2]*3 + +b1[3]*2+ b1[4]*1

st.subheader("Calculo del costo")
st.write(f"El costo estimado de la actividad es: **${prediccion:,.2f}** pesos")
