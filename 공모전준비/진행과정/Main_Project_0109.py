import requests
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from scipy.stats import chisquare
from datetime import datetime
import time

# Binance API endpoint
BASE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1m"  # Interval for kline data
LIMIT = 1000  # Max number of data points per request

# Date to timestamp conversion
def date_to_timestamp(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
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

# Save data to CSV
def save_to_csv(data, symbol, save_directory, start_date, end_date):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    file_path = f"{save_directory}/{symbol}_{start_date}_{end_date}_data.csv"
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
    plt.figure(figsize=(10, 6))
    plt.bar(actual_frequencies.index, actual_frequencies, alpha=0.7, label='Actual Distribution')
    plt.plot(range(1, 10), benford_dist, 'ro-', label='Benford Distribution')
    plt.title(f'{symbol} Volume - Benford\'s Law Analysis ({start_date} to {end_date})')
    plt.xlabel('First Digit')
    plt.ylabel('Frequency (Proportion)')
    plt.xticks(range(1, 10))
    plt.legend()
    graph_path = f"{save_directory}/{symbol}_{start_date}_{end_date}_benford_graph.png"
    plt.savefig(graph_path)
    plt.close()

# Plot and save table for Benford's Law analysis
def plot_benford_table(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory):
    benford_table = pd.DataFrame({
        'First Digit': range(1, 10),
        'Actual Frequency': [actual_frequencies.get(i, 0) for i in range(1, 10)],
        'Benford Frequency': benford_dist,
    })
    table_path = f"{save_directory}/{symbol}_{start_date}_{end_date}_benford_table.png"
    plt.figure(figsize=(8, 4))
    table_ax = plt.gca()
    table_ax.axis('off')
    table_plot = table(table_ax, benford_table, loc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1.2, 1.2)
    plt.title(f'{symbol} Volume - Benford\'s Law Analysis ({start_date} to {end_date})')
    plt.savefig(table_path, bbox_inches='tight')
    plt.close()

# Main function
def main():
    symbol = input("Enter the cryptocurrency symbol (e.g., BTCUSDT): ").strip().upper()
    start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter the end date (YYYY-MM-DD): ").strip()

    start_time = date_to_timestamp(start_date)
    end_time = date_to_timestamp(end_date)
    save_directory = './crypto_data'

    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    file_path = None
    current_time = start_time
    while current_time < end_time:
        try:
            data = fetch_data(symbol, current_time, min(current_time + LIMIT * 60000, end_time))
            if not data:
                print(f"No data available for {symbol}.")
                break
            file_path = save_to_csv(data, symbol, save_directory, start_date, end_date)
            print(f"Saved data for {symbol}.")
            current_time = int(data[-1][0]) + 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}. Retrying in 10 seconds...")
            time.sleep(10)

    if file_path:
        print("Starting Benford's Law analysis...")
        data = pd.read_csv(file_path, names=["Time", "Open", "High", "Low", "Close", "Volume"])
        actual_frequencies, benford_dist = first_digit_analysis(data, "Volume")
        plot_benford_graph(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory)
        plot_benford_table(actual_frequencies, benford_dist, symbol, start_date, end_date, save_directory)

        observed = [actual_frequencies.get(i, 0) for i in range(1, 10)]
        expected = benford_dist
        chi2_stat, p_val = chisquare(observed, expected)
        print(f"Chi-Square Statistic: {chi2_stat}, P-value: {p_val}")
        if p_val < 0.05:
            print(f"{symbol} volume does not follow Benford's Law.")
        else:
            print(f"{symbol} volume follows Benford's Law.")
    else:
        print("No data to analyze.")

if __name__ == "__main__":
    main()
