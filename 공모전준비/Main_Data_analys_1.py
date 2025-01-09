import requests
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from scipy.stats import chisquare
from datetime import datetime

# Binance API endpoint
BASE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1m"  # Interval for kline data
LIMIT = 1000  # Max number of data points per request

# 날짜와 시간 문자열을 타임스탬프로 변환하는 함수
def datetime_to_timestamp(datetime_str):
    dt = datetime.strptime(datetime_str, "%Y-%m-%d-%H:%M")
    return int(dt.timestamp() * 1000)

# Fetch cryptocurrency data from Binance
def fetch_data(symbol, start_time, end_time):
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": LIMIT,
        "startTime": start_time,
        "endTime": end_time,
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

# Remove 'USDT' from symbol for file naming
def clean_symbol(symbol):
    return symbol.replace('USDT', '')

# Save data to CSV
def save_to_csv(data, symbol, save_directory, start_date, end_date):
    clean_symbol_name = clean_symbol(symbol)
    csv_directory = os.path.join(save_directory, 'csv')
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    file_path = f"{csv_directory}/{clean_symbol_name}_{start_date}_{end_date}_data.csv"
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for kline in data:
            writer.writerow([
                datetime.utcfromtimestamp(kline[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                float(kline[1]), float(kline[2]), float(kline[3]),
                float(kline[4]), float(kline[5])
            ])
    return file_path

# Benford analysis
def first_digit_analysis(data, column):
    data['First_Digit'] = data[column].apply(lambda x: int(str(x)[0]) if pd.notnull(x) and str(x)[0] != '0' else None)
    digit_counts = data['First_Digit'].value_counts().sort_index()
    benford_dist = [np.log10(1 + 1 / d) for d in range(1, 10)]
    actual_frequencies = digit_counts / digit_counts.sum()
    return actual_frequencies, benford_dist

# Plot and save graph for Benford's Law analysis
def plot_benford_graph(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory):
    clean_symbol_name = clean_symbol(symbol)
    graph_directory = os.path.join(save_directory, 'graphs')
    if not os.path.exists(graph_directory):
        os.makedirs(graph_directory)
    
    plt.figure(figsize=(10, 6))
    
    # Actual Distribution as histogram (bar plot)
    plt.bar(actual_frequencies.index, actual_frequencies.values, alpha=0.5, label='Actual Distribution')
    
    # Actual Distribution as line plot (connecting points with a dashed line)
    plt.plot(actual_frequencies.index, actual_frequencies.values, 'b-', marker='o', linestyle='--', color='blue')
    
    # Benford's Distribution as line plot
    plt.plot(range(1, 10), benford_dist, 'ro-', label='Benford Distribution', markerfacecolor='red')
    
    # Graph titles and labels
    plt.title(f'{clean_symbol_name} Volume - Benford\'s Law Analysis ({start_date} to {end_date})')
    plt.xlabel('First Digit')
    plt.ylabel('Frequency (Proportion)')
    plt.xticks(range(1, 10))
    plt.legend()
    
    # Save the graph
    graph_path = f"{graph_directory}/{clean_symbol_name}_{start_date}_{end_date}_benford_graph.png"
    plt.savefig(graph_path)
    plt.close()

# Plot and save table for Benford's Law analysis
def plot_benford_table(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory):
    clean_symbol_name = clean_symbol(symbol)
    table_directory = os.path.join(save_directory, 'tables')
    if not os.path.exists(table_directory):
        os.makedirs(table_directory)
    actual_frequencies_1 = actual_frequencies.round(4)
    benford_dist_1 = [round(f, 4) for f in benford_dist]
    benford_table = pd.DataFrame({
        'First Digit': range(1, 10),
        'Actual Frequency': [actual_frequencies_1.get(i, 0) for i in range(1, 10)],
        'Benford Frequency': benford_dist_1,
    })
    table_path = f"{table_directory}/{clean_symbol_name}_{start_date}_{end_date}_benford_table.png"
    plt.figure(figsize=(8, 4))
    table_ax = plt.gca()
    table_ax.axis('off')
    table_plot = table(table_ax, benford_table, loc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1.2, 1.2)
    plt.title(f'{clean_symbol_name} Volume - Benford\'s Law Analysis ({start_date} to {end_date})')
    plt.savefig(table_path, bbox_inches='tight')
    plt.close()

# Perform and save Chi-Square test results
def perform_chi_square_test(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory):
    clean_symbol_name = clean_symbol(symbol)
    observed = [actual_frequencies.get(i, 0) for i in range(1, 10)]
    expected = benford_dist
    chi2_stat, p_val = chisquare(observed, expected)

    text_directory = os.path.join(save_directory, 'texts')
    if not os.path.exists(text_directory):
        os.makedirs(text_directory)
    chi_square_results_path = f"{text_directory}/{clean_symbol_name}_{start_date}_{end_date}_chi_square_results.txt"
    with open(chi_square_results_path, "w") as file:
        file.write(f"Chi-Square Test Results ({start_date} to {end_date}):\n")
        file.write(f"Symbol: {clean_symbol_name}\n")
        file.write(f"Chi-Square Statistic: {chi2_stat}\n")
        file.write(f"P-value: {p_val}\n")
        if p_val < 0.01:
            file.write(f"Conclusion: {clean_symbol_name} volume does not follow Benford's Law.\n")
        else:
            file.write(f"Conclusion: {clean_symbol_name} volume follows Benford's Law.\n")
    print(f"Chi-Square test results saved to: {chi_square_results_path}")

# Main function
def main():
    # 사용자에게 시작 날짜 및 시간과 종료 날짜 및 시간을 입력 받기
    start_datetime = input("Enter the start date and time (YYYY-MM-DD-HH:MM): ").strip()
    end_datetime = input("Enter the end date and time (YYYY-MM-DD-HH:MM): ").strip()

    try:
        start_time = datetime_to_timestamp(start_datetime)
        end_time = datetime_to_timestamp(end_datetime)
    except ValueError:
        print("Invalid date-time format. Please use the format YYYY-MM-DD-HH:MM.")
        return

    symbols_input = input("Enter the cryptocurrency symbols (separated by commas, e.g., BTCUSDT, ETHUSDT): ").strip().upper()
    symbols = symbols_input.split(",")  # 쉼표로 구분하여 리스트로 저장

    # 최대 50개까지만 입력 받기
    if len(symbols) > 50:
        print("You can only enter up to 50 cryptocurrency symbols.")
        return

    save_directory = './crypto_data'

    # 사용자에게 모든 암호화폐에 대해 카이제곱 검정을 수행할지 여부를 묻기
    perform_all_tests = input("Do you want to perform the Chi-Square test for all cryptocurrencies? (w-yes or w-no): ").strip().lower()

    # 각 암호화폐에 대해 데이터를 받아 분석
    for symbol in symbols:
        symbol = symbol.strip()  # 공백 제거
        print(f"\nFetching data for {symbol} from {start_datetime} to {end_datetime}...")
        file_path = None
        current_time = start_time
        while current_time < end_time:
            try:
                data = fetch_data(symbol, current_time, min(current_time + LIMIT * 60000, end_time))
                if not data:
                    print(f"No data available for {symbol}.")
                    break
                file_path = save_to_csv(data, symbol, save_directory, start_datetime, end_datetime)
                print(f"Saved data for {symbol}.")
                current_time = int(data[-1][0]) + 1
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {symbol}: {e}. Skipping to next cryptocurrency.")
                break

        if file_path:
            print(f"Starting Benford's Law analysis for {symbol}...")
            data = pd.read_csv(file_path, names=["Time", "Open", "High", "Low", "Close", "Volume"])
            actual_frequencies, benford_dist = first_digit_analysis(data, "Volume")
            plot_benford_graph(actual_frequencies, benford_dist, symbol, start_datetime, end_datetime, save_directory)
            plot_benford_table(actual_frequencies, benford_dist, symbol, start_datetime, end_datetime, save_directory)

            # w-yes일 경우 모든 암호화폐에 대해 카이제곱 검정 수행
            if perform_all_tests == "w-yes":
                perform_chi_square_test(actual_frequencies, benford_dist, symbol, start_datetime, end_datetime, save_directory)
            elif perform_all_tests == "w-no":
                print(f"Skipping Chi-Square test for {symbol} as 'w-no' was selected.")
            else:
                # w-no가 아닌 경우, 각 암호화폐에 대해 카이제곱 검정을 묻는 방식
                perform_test = input(f"Do you want to perform the Chi-Square test for {symbol}? (Yes or No): ").strip().lower()
                if perform_test == "yes":
                    perform_chi_square_test(actual_frequencies, benford_dist, symbol, start_datetime, end_datetime, save_directory)
                else:
                    print(f"Skipping Chi-Square test for {symbol}.")
        else:
            print(f"No data to analyze for {symbol}.")
    
if __name__ == "__main__":
    main()
