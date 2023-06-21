
import pandas as pd
import streamlit as st
import plotly.express as px


data = pd.read_csv("predictions.csv")


data.rename(columns={"Przewidywana data dla predykcji": "Data_predykcji"}, inplace=True)

# Tytuł raportu
st.title("Raport z predykcji")

# Wyświetlenie danych w tabeli
st.subheader("Przewidywane dane dla predykcji")
st.dataframe(data)

# Wykres predykcji
st.subheader("Wykres predykcji")
fig = px.line(data, x="Data", y=["Predykcja LSTM", "Predykcja GRU"])
st.plotly_chart(fig, use_container_width=True)

# Statystyki podsumowujące
st.subheader("Statystyki podsumowujące")
st.write("Średnia predykcja LSTM:", data["Predykcja LSTM"].mean())
st.write("Średnia predykcja GRU:", data["Predykcja GRU"].mean())