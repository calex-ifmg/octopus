import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Carregar os dados
file_path = "Construtoras_classificado1_atualizado_V2.xlsx"
data = pd.read_excel(file_path)

# Limpeza e preparação dos dados
data = data.dropna(subset=["Equipamento", "Data de coleta", "Modos de falha"])
data = data.sort_values(by=["Equipamento", "Data de coleta"])

# Criar colunas binárias para cada modo de falha
data["Modos de falha"] = data["Modos de falha"].apply(
    lambda x: [mode.strip() for mode in str(x).split('.') if mode.strip()]
)
unique_modes_split = sorted(set([mode for sublist in data["Modos de falha"] for mode in sublist]))
for mode in unique_modes_split:
    data[f"Mode_{mode}"] = data["Modos de falha"].apply(
        lambda x: 1 if isinstance(x, list) and mode in x else 0
    )

# Selecionar características
selected_features = ["Viscosidade a 40ºC [cSt]", "Ponto de fulgor [ºC]", "TBN [mg KOH/g]",
                     "Insolúveis [%]", "Al [ppm]", "Cr [ppm]", "Cu [ppm]", "Fe [ppm]", "Si [ppm]"]

# Treinamento do modelo
def treinar_modelo(data):
    X = data[selected_features]
    y = data[[f"Mode_{mode}" for mode in unique_modes_split]]
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    clf.fit(X, y)
    return clf

multi_model = treinar_modelo(data)

# Interface Streamlit
st.title("Sistema de Análise Preditiva")

# Seleção de equipamento e compartimento
equipamentos = ["Todos"] + list(data["Equipamento"].dropna().unique())
equipamento = st.selectbox("Selecione o Equipamento", equipamentos)

compartimentos = ["Todos"] + list(data["Tipo de Compartimento"].dropna().unique())
compartimento = st.selectbox("Selecione o Tipo de Compartimento", compartimentos)

# Filtrar dados
if equipamento != "Todos":
    data = data[data["Equipamento"] == equipamento]
if compartimento != "Todos":
    data = data[data["Tipo de Compartimento"] == compartimento]

# Exibir dados filtrados
st.subheader("Visualização dos Dados")
st.write(data.head())

# Calcular métricas do modelo
X = data[selected_features]
y = data[[f"Mode_{mode}" for mode in unique_modes_split]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Exibir métricas
st.subheader("Métricas do Modelo")
st.write(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
st.write(f"Precisão: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.2%}")
st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.2%}")
st.write(f"F1-Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.2%}")
