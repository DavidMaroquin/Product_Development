import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats #Cientific Python
import numpy as np
from scipy.stats import chi2_contingency

# Página principal

def page_1():
    # Cargar archivo
    file = st.file_uploader("Cargar archivo", type=["csv", "xlsx"])

    # Guardar archivo en la sesión del usuario

    #if file is not None:
        #st.session_state["file"] = file

       
    if file is not None:
        st.session_state["file"] = file


# Página dos

def page_2():
    # Desplegar información cargada
    file = st.session_state["file"]
    
    if file: 
    #Obtener la extensión
        file_extension = file.name.split(".")[-1]

     #Leer el archivo
        if file_extension == "csv":
            data = pd.read_csv(file)
        elif file_extension == "xlsx":
            data = pd.read_excel(file)

    # Mostrar el DataFrame
    st.dataframe(data)

    # Obtener los nombres de las columnas
    columnas = data.columns.tolist()

    # Definir listas para almacenar los nombres de las variables
    fechas = []
    categoricas = []
    discretas = []
    continuas = []

    for columna in columnas:
        if data[columna].dtype == 'datetime64[ns]':
            fechas.append(columna)
        elif data[columna].dtype == 'object':
            categoricas.append(columna)
        elif len(data[columna].unique()) < 10:
            discretas.append(columna)
        else:
            continuas.append(columna)

    # Crear la aplicación web con Streamlit
    st.title("Clasificación de Variables")
    st.subheader("Variables de Fecha")
    st.write(fechas)

    st.subheader("Variables Categóricas")
    st.write(categoricas)

    st.subheader("Variables Discretas")
    st.write(discretas)

    st.subheader("Variables Continuas")
    st.write(continuas)    

    st.title("Gráficos de Variables Continuas") 
    # Iterar sobre las variables continuas
    for variable in continuas:
        st.header(f'Variable: {variable}')

        # Obtener datos de la variable actual
        datos = data[variable]

        # Calcular estadísticas
        media = datos.mean()
        mediana = datos.median()
        desviacion_estandar = datos.std()
        varianza = datos.var()

        # Crear la gráfica de densidad con histograma
        plt.figure(figsize=(10, 6))
        sns.histplot(datos, kde=True, color='skyblue')
        plt.axvline(media, color='red', linestyle='dashed', linewidth=1, label=f'Media: {media:.2f}')
        plt.axvline(mediana, color='green', linestyle='dashed', linewidth=1, label=f'Mediana: {mediana:.2f}')
        plt.title(f'Variable: {variable}')
        plt.xlabel('Valores')
        plt.ylabel('Densidad')
        plt.legend()
        st.pyplot(plt)

        # Imprimir estadísticas
        st.write(f'Media: {media:.2f}')
        st.write(f'Mediana: {mediana:.2f}')
        st.write(f'Desviación Estándar: {desviacion_estandar:.2f}')
        st.write(f'Varianza: {varianza:.2f}')

        
        
    st.title("Gráficos de Variables Discretas")     
        # Iterar sobre las variables discretas
        
    for variable in discretas:
        st.header(f'Variable: {variable}')

        # Obtener datos de la variable actual
        datos = data[variable]

        # Calcular estadísticas
        media = datos.mean()
        mediana = datos.median()
        desviacion_estandar = datos.std()
        varianza = datos.var()
        moda = datos.mode().iloc[0]  # Moda

        # Crear la gráfica de histograma
        plt.figure(figsize=(10, 6))
        sns.histplot(datos, kde=False, color='skyblue')
        plt.axvline(media, color='red', linestyle='dashed', linewidth=1, label=f'Media: {media:.2f}')
        plt.axvline(mediana, color='green', linestyle='dashed', linewidth=1, label=f'Mediana: {mediana:.2f}')
        plt.title(f'Variable: {variable}')
        plt.xlabel('Valores')
        plt.ylabel('Frecuencia')
        plt.legend()
        st.pyplot(plt)

        # Imprimir estadísticas
        st.write(f'Media: {media:.2f}')
        st.write(f'Mediana: {mediana:.2f}')
        st.write(f'Desviación Estándar: {desviacion_estandar:.2f}')
        st.write(f'Varianza: {varianza:.2f}')
        st.write(f'Moda: {moda}')

    st.title("Gráficos de Variables Categóricas") 

    # Iterar sobre las variables categóricas
    for variable in categoricas:
        st.header(f'Variable: {variable}')

        # Contar los valores totales por categoría
        conteo_por_categoria = data[variable].value_counts()

        # Crear la gráfica de barras
        plt.figure(figsize=(10, 6))
        sns.barplot(x=conteo_por_categoria.index, y=conteo_por_categoria.values, palette='viridis')
        plt.title(f'Conteo por Categoría')
        plt.xlabel('Categoría')
        plt.ylabel('Conteo')
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Mostrar la tabla con los valores totales por categoría
        st.write(conteo_por_categoria)
    
    
    st.title("Gráficos de Variables Continuas/Discretas vs Continuas/Discretas") 

    fig1 = plt.figure(figsize=(10, 4))
    variable_x = st.selectbox("Variable Continua/Discretas", continuas+discretas)
    variable_y = st.selectbox("Variables Continuas/Discretas :", continuas+discretas)

        
    plt.scatter(data[variable_x], data[variable_y])
    plt.title(f'Scatter Plot entre {variable_x} y {variable_y}')
    plt.xlabel(variable_x)
    plt.ylabel(variable_y)
    st.pyplot(fig1)
        
     # Calcular la correlación entre las variables seleccionadas
    correlacion = data[variable_x].corr(data[variable_y])

    st.write(f'La correlación entre {variable_x} y {variable_y} es: {correlacion:.2f}')
    
    st.title("Gráficos de Variables Continuas/Discretas vs Temporal") 

         
    fig2 = plt.figure(figsize=(10, 4))
    variableX = st.selectbox("Variable Continua/Discretas", discretas+continuas)
    variableY = None
    if not fechas:
        st.warning("No hay variables disponibles.")
    else:
        variableY = st.selectbox(
            'Variables Fecha:', fechas)
        st.write(f'Has seleccionado: {variableY}')

   # Crear la gráfica de serie de tiempo
    if variableX is not None and variableY is not None:

        if data[variableX].empty or data[variableY].empty:
            st.warning("No hay datos disponibles para crear la serie de tiempo.")
        else:
            # Crear la gráfica de serie de tiempo
            plt.figure(figsize=(10, 6))
            plt.plot(data[variableX], data[variableY])
            plt.title(f'Serie de Tiempo de {variableY} en función de {variableX}')
            plt.xlabel(variableX)
            plt.ylabel(variableY)
            plt.xticks(rotation=45)
            st.pyplot()
    else:
            st.warning("Por favor selecciona ambas variables para crear la serie de tiempo.")
        
    st.title("Gráficos de Variables Continuas vs Categóricas") 

    fig3 = plt.figure(figsize=(10, 4))
    variableA = st.selectbox("Variable Continua", continuas)
    variableB = st.selectbox("Variable Categoricas", categoricas)

    sns.boxplot(data, x=variableB, y=variableA)
    st.pyplot(fig3)

    st.title("Gráficos de Variables Categóricas vs Categóricas") 

    fig4 = plt.figure(figsize=(10, 4))
    variable1 = st.selectbox("Variable Categórica", categoricas)
    variable2 = st.selectbox("Variable Categorica", categoricas)

    #Crear la gráfica de mosaico
    contingency_table = pd.crosstab(data[variable1], data[variable2])
    sns.heatmap(contingency_table, annot=True, cmap='YlGnBu')
    st.pyplot(fig4)

    # Calcular el valor de chi-cuadrado y el p-valor
    chi2, p, _, _ = chi2_contingency(contingency_table)

    # Calcular Cramér V
    n = data.shape[0]
    min_dim = min(contingency_table.shape)
    cramer_v = np.sqrt(chi2 / (n * (min_dim - 1)))


    st.write(f'Chi-cuadrado: {chi2}')
    st.write(f'P-valor: {p}')
    st.write(f'Cramér V: {cramer_v}')

    
    # Generar un gráfico de heatmap para Cramér V
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array([[cramer_v]]), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title('Cramér V Heatmap')
    st.pyplot(plt)
    #st.write(f'Coeficiente de Contingencia de Cramer: {cramer_v:.4f}')

   # print(f"Chi-squared statistic: {chi2_stat}")
    #print(f"P-value: {p_val}")
    #print(f"Degrees of freedom: {dof}")
    #print("Expected frequencies:")
    #print(expected)

   


# Definir páginas

st.title("Aplicación para analizar modelos de análisis de datos")

# Navegación entre páginas

page = st.sidebar.selectbox("Página", ["Principal", "Analisis"])

if page == "Principal":
    page_1()
elif page == "Analisis":
    page_2()

if(__name__=='__page_1__'):
    page_1()