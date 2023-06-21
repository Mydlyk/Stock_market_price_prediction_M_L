import datetime
import requests
from sklearn.linear_model import LinearRegression
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
import csv
import matplotlib.pyplot as plt

import time

def fetch_stock_data():
    # ustawienie danych w try w celu gdy nie ma dostepu do internetu został obsluzony wyjatek
    try:
        start_date = datetime.datetime.now() - datetime.timedelta(days=90)  # Pobieranie danych z 3 miesiecy
        end_date = datetime.datetime.now()

        # Pobieranie ceny ropy 
        oil = yf.download("CL=F", start=start_date, end=end_date)

        # indeksu S&P 500
        sp500 = yf.download("^GSPC", start=start_date, end=end_date)

        #wskaźnika VIX
        
        vix = yf.download("^VIX", start=start_date, end=end_date)
        

        
        sp500_close = sp500["Close"]
        vix_close = vix["Close"]
        
        #Przypisanie danych do indeksów
        data = pd.merge(oil, sp500_close, left_index=True, right_index=True)
        data = pd.merge(data, vix_close, left_index=True, right_index=True)

        #Zwrócenie 3 ostatnich dni dla porównania
        last_three_days = data.tail(3)

        return data, last_three_days
    # nie ma dostepu do internetu ale jest plik zastepczy csv
    except:
        print(f"Błąd: Wystąpił problem podczas pobierania danych z internetu: ")
        try:
            data = pd.read_csv("oil_prices.csv")
            last_three_days = data[-3:]
            return data, last_three_days
       #nie ma internetu oraz pliku csv
        except:
            print(f"Błąd: Wystąpił problem podczas odczytu danych z pliku CSV: ")

    return None, None


def prepare_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 1:])  # Wybrane wskaźniki giełdowe jako cechy
        y.append(data[i+window_size, 0])  # Cena zamknięcia ropy naftowej jako wartość docelowa
    return np.array(X), np.array(y)


def predict_oil_price(data, last_three_days):
    
    # Przygotowanie danych do modelu 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.iloc[:, [0, 2]].values)

    window_size = 10
    X, y = prepare_data(scaled_data, window_size)

    # Podział danych na zestawy treningowe i testowe
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Tworzenie modelu LSTM
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(units=1))

    # Tworzenie modelu GRU
    gru_model = Sequential()
    gru_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    gru_model.add(GRU(units=50))
    gru_model.add(Dense(units=1))

    # Kompilacja i trenowanie modeli
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32)

    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Wykonanie predykcji na danych testowych
    lstm_predictions = lstm_model.predict(X_test)
    gru_predictions = gru_model.predict(X_test)

    # Odwrócenie skalowania danych
    lstm_predictions = scaler.inverse_transform(np.concatenate((X_test[:, -1, :], lstm_predictions), axis=1))[:, -1]
    gru_predictions = scaler.inverse_transform(np.concatenate((X_test[:, -1, :], gru_predictions), axis=1))[:, -1]

    # Wykonanie predykcji na danych dla 90 dni
    lstm_predictions_full = lstm_model.predict(X)
    gru_predictions_full = gru_model.predict(X)

    # Odwrócenie skalowania danych dla 90 dni
    lstm_predictions_full = scaler.inverse_transform(np.concatenate((X[:, -1, :], lstm_predictions_full), axis=1))[:, -1]
    gru_predictions_full = scaler.inverse_transform(np.concatenate((X[:, -1, :], gru_predictions_full), axis=1))[:, -1]

    print("Ceny z trzech ostatnich dni:")
    for i, (_, row) in enumerate(last_three_days.iterrows(), 1):
        print(f"Cena ropy sprzed: {i} dni {row['Close_x']} (Cena ropy), {row['Close_y']} (S&P 500), {row['Close']} (VIX)")

    # Wyświetlenie wyników predykcji konsola
    print("Wyniki predykcji:")
    for i in range(len(lstm_predictions)):
        print(f'Predykcja na dzień {i+1}: LSTM: {lstm_predictions[i]}, GRU: {gru_predictions[i]}')

    return lstm_predictions, gru_predictions, lstm_predictions_full, gru_predictions_full

#zapis do pliku csv w celu stworzenia interaktywnego raportu
def save_to_csv(filename, predictions, mode):
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)

        if mode == 'a' and file.tell() == 0:
            writer.writerow(["Data", "Data wykonania", "Predykcja LSTM", "Predykcja GRU"])  # Nagłówek dla istniejącego pliku

        for i, prediction in enumerate(predictions, 1):
            if mode == 'a':
                marker = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Data i czas wykonania predykcji
            else:
                marker = f"Predykcja {i}"  # Oznaczenie dla nowego pliku

            prediction_date = (datetime.datetime.now() + datetime.timedelta(days=i)).strftime("%Y-%m-%d")

            writer.writerow([prediction_date, marker, prediction[0], prediction[1]])

    if mode == 'a':
        print(f"Dane dopisano do pliku {filename}")
    else:
        print(f"Dane zapisano do nowego pliku {filename}")


# wybor trybow

def main():
    mode = input("Wybierz tryb pracy (1 - Interaktywny, 2 - Automatyczny): \n ")
    if mode == "1":
        interactive_mode()
    elif mode == "2":
        automatic_mode()
    else:
        print("Niepoprawny tryb. Wybierz 1 lub 2.")

#wywolania aplikacji o inych godzinach niz 10
def interactive_mode():
    data, last_three_days = fetch_stock_data()
    lstm_predictions, gru_predictions, lstm_predictions_full, gru_predictions_full = predict_oil_price(data, last_three_days)
    plot_predictions(data["Close_x"].values[-90:], lstm_predictions_full, gru_predictions_full)


    save_to_csv("predictions.csv", zip(lstm_predictions, gru_predictions), mode='a')

#tryb automatyczny wywoluja aplikacje o 10
def automatic_mode():
    while True:
        now = datetime.datetime.now()
        if now.hour == 10 and now.minute == 0:
            data, last_three_days = fetch_stock_data()
            lstm_predictions, gru_predictions, lstm_predictions_full, gru_predictions_full = predict_oil_price(data, last_three_days)
            plot_predictions(data["Close_x"].values[-90:], lstm_predictions_full, gru_predictions_full)


            save_to_csv("predictions.csv", zip(lstm_predictions, gru_predictions), mode='a')
        else:
            current_time = now.strftime("%H:%M:%S")
            print(f"Oczekiwanie na godzinę 10:00... Aktualna godzina: {current_time}")
            time.sleep(60)  # Poczekaj 30 sekund i sprawdź ponownie

# wykresy 
def plot_predictions(true_values, lstm_predictions, gru_predictions):
    days_true = range(1, len(true_values) + 1)
    days_pred = range(len(true_values) + 1, len(true_values) + len(lstm_predictions) + 1)

    # Wykres porównania predykcji do wartości prawdziwych
    plt.figure(figsize=(12, 6))
    plt.plot(days_true, true_values, label='Wartości prawdziwe', color='blue')
    plt.plot(days_pred, lstm_predictions, label='Predykcje LSTM', color='orange')
    plt.plot(days_pred, gru_predictions, label='Predykcje GRU', color='green')

    plt.xlabel('Dzień')
    plt.ylabel('Cena ropy naftowej')
    plt.title('Porównanie predykcji do wartości prawdziwych')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Wykres różnic między predykcjami a wartościami prawdziwymi
    diff_lstm = lstm_predictions - true_values[-len(lstm_predictions):]
    diff_gru = gru_predictions - true_values[-len(gru_predictions):]
    x = range(1, len(lstm_predictions) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(x, diff_lstm, label='Różnice LSTM', color='orange')
    plt.plot(x, diff_gru, label='Różnice GRU', color='green')

    plt.xlabel('Dzień')
    plt.ylabel('Różnica')
    plt.title('Różnice między predykcjami a wartościami prawdziwymi')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except:
        print("Wystapił bład podczas ladowania danych szczegóły powyzej")