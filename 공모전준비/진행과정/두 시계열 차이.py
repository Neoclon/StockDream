import os
import pandas as pd
import matplotlib.pyplot as plt

def construct_file_path(exchange, symbol, analysis_target, start_datetime, end_datetime, term_days):
    base_path = "/crypto_data/Timeseries_data/MAC_result/"
    filename = f"{exchange.capitalize()}_{symbol}_{analysis_target}_MAC_Results_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day.csv"
    return os.path.join(base_path, filename)

def load_and_prepare_data(binance_path, upbit_path):
    # Load Binance and Upbit data
    binance_data = pd.read_csv(binance_path)
    upbit_data = pd.read_csv(upbit_path)

    # Convert date columns to datetime
    binance_data["start_date"] = pd.to_datetime(binance_data["start_date"], format="%Y-%m-%d-%H:%M")
    upbit_data["start_date"] = pd.to_datetime(upbit_data["start_date"], format="%Y-%m-%d-%H:%M")

    # Sort Binance data in ascending order
    binance_data = binance_data.sort_values(by="start_date")

    # Sort Upbit data in ascending order
    upbit_data = upbit_data.sort_values(by="start_date")

    return binance_data, upbit_data

def plot_mac_comparison(binance_data, upbit_data):
    plt.figure(figsize=(14, 8))

    # Binance First Digit
    plt.plot(binance_data[binance_data["digit_type"] == "first"]["start_date"],
             binance_data[binance_data["digit_type"] == "first"]["mad"],
             label="Binance (First Digit)", marker='o')

    # Upbit First Digit
    plt.plot(upbit_data[upbit_data["digit_type"] == "first"]["start_date"],
             upbit_data[upbit_data["digit_type"] == "first"]["mad"],
             label="Upbit (First Digit)", marker='o')

    # Binance Second Digit
    plt.plot(binance_data[binance_data["digit_type"] == "second"]["start_date"],
             binance_data[binance_data["digit_type"] == "second"]["mad"],
             label="Binance (Second Digit)", marker='x')

    # Upbit Second Digit
    plt.plot(upbit_data[upbit_data["digit_type"] == "second"]["start_date"],
             upbit_data[upbit_data["digit_type"] == "second"]["mad"],
             label="Upbit (Second Digit)", marker='x')

    # Customize plot
    plt.title("MAC Comparison: Binance vs Upbit")
    plt.xlabel("Date")
    plt.ylabel("Mean Absolute Conformity (MAC)")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    # User inputs
    exchange_binance = "Binance"
    exchange_upbit = "Upbit"
    symbol_binance = input("Enter Binance symbol (e.g., EOSUSDT): ").strip().upper()
    symbol_upbit = input("Enter Upbit symbol (e.g., KRW-EOS): ").strip().upper()
    start_datetime = input("Enter start datetime (YYYY-MM-DD-HH:MM): ").strip()
    end_datetime = input("Enter end datetime (YYYY-MM-DD-HH:MM): ").strip()
    term_days = input("Enter term length in days (e.g., 1): ").strip()
    analysis_target = input("Enter analysis target (TA/TV/VCR/PCR): ").strip().upper()

    # Construct file paths
    binance_path = construct_file_path(exchange_binance, symbol_binance, analysis_target, start_datetime, end_datetime, term_days)
    upbit_path = construct_file_path(exchange_upbit, symbol_upbit, analysis_target, start_datetime, end_datetime, term_days)

    # Load and prepare data
    binance_data, upbit_data = load_and_prepare_data(binance_path, upbit_path)

    # Plot MAC comparison
    plot_mac_comparison(binance_data, upbit_data)

if __name__ == "__main__":
    main()
