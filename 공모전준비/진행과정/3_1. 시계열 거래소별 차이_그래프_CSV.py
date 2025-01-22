import os
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime

#################################################
# 현재 거래소: Binance vs Upbit
# 현재 날짜: 2024-07-01-00:00 부터 2025-01-01-00:00
# 현재 간격: 3일
# 현재 종류:  both
# 현재 target: TA
#################################################

def construct_file_path(exchange, symbol, analysis_target, start_datetime, end_datetime, term_days):
    base_path = f"./crypto_data/Timeseries_data/MAC_result/{term_days}Day_{analysis_target}"
    filename = f"{exchange.capitalize()}_{symbol}_{analysis_target}_MAC_Results_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{term_days}day.csv"
    return os.path.join(base_path, filename)

def fetch_price_data_binance(symbol, start_datetime, end_datetime):
    """Fetch price data from Binance API."""
    base_url = "https://api.binance.com/api/v3/klines"
    interval = "1d"
    start_time = int(datetime.strptime(start_datetime, "%Y-%m-%d-%H:%M").timestamp() * 1000)
    end_time = int(datetime.strptime(end_datetime, "%Y-%m-%d-%H:%M").timestamp() * 1000)

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()

    price_data = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    price_data = price_data.iloc[:, :6]  # Keep only relevant columns
    price_data.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    price_data["timestamp"] = pd.to_datetime(price_data["timestamp"], unit="ms")
    price_data["open"] = price_data["open"].astype(float)
    price_data["close"] = price_data["close"].astype(float)

    return price_data

def load_and_prepare_data(exchange_paths):
    # Load data for multiple exchanges
    data_frames = []
    for path in exchange_paths:
        data = pd.read_csv(path)
        data["start_date"] = pd.to_datetime(data["start_date"], format="%Y-%m-%d-%H:%M")
        data = data.sort_values(by="start_date")
        data_frames.append(data)
    return data_frames

def calculate_mac_difference(exchange_data):
    # Merge the data on start_date and digit_type
    merged_data = pd.merge(exchange_data[0], exchange_data[1], on=["start_date", "digit_type"], suffixes=("_1", "_2"))
    merged_data["mac_difference"] = abs(merged_data["mad_1"] - merged_data["mad_2"])
    return merged_data

def plot_mac_and_price(mac_difference_data, price_data, output_path, digit_selection, styles, exchange_1, exchange_2, symbol):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})

    # MAC Difference Plot
    digit_types = []
    if digit_selection == "both":
        digit_types = ["first", "second"]
    elif digit_selection == "first":
        digit_types = ["first"]
    elif digit_selection == "second":
        digit_types = ["second"]

    for digit_type in digit_types:
        filtered_data = mac_difference_data[mac_difference_data["digit_type"] == digit_type]
        style = styles.get(digit_type, {})
        ax1.plot(
            filtered_data["start_date"],
            filtered_data["mac_difference"],
            label=f"MAC Difference ({digit_type.capitalize()} Digit)",
            color=style.get("color", "blue"),
            marker=style.get("marker", "o"),
            markersize=style.get("markersize", 5),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 1),
        )

    ax1.set_title(f"MAC Difference for {symbol} Between {exchange_1.capitalize()} and {exchange_2.capitalize()}")
    ax1.set_ylabel("MAC Difference")
    ax1.legend()
    ax1.grid(alpha=0.5)

    # Price Movement Plot
    for _, row in price_data.iterrows():
        date = row["timestamp"]
        open_price = row["open"]
        close_price = row["close"]

        if pd.notnull(open_price) and pd.notnull(close_price):
            if open_price == close_price:  # 시가와 종가가 같은 경우
                ax2.plot([date, date], [open_price * 0.95, close_price * 1.05], color='#D55E00', linewidth=3, alpha=1)
            else:
                color = '#2ECC71' if close_price > open_price else '#E74C3C'  # 상승: 초록, 하락: 빨강
                ax2.plot([date, date], [open_price, close_price], color=color, linewidth=3, alpha=1)

    ax2.set_ylabel("Price_Binance")
    ax2.set_xlabel('Date')
    ax2.grid(alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def calculate_statistics(mac_difference_data):
    # Calculate mean and standard deviation for each digit type
    statistics = {}
    for digit_type in ["first", "second"]:
        filtered_data = mac_difference_data[mac_difference_data["digit_type"] == digit_type]
        if not filtered_data.empty:
            mean_value = filtered_data["mac_difference"].mean()
            std_dev = filtered_data["mac_difference"].std()
            statistics[digit_type] = {"mean": mean_value, "std_dev": std_dev}
    return statistics

def save_statistics_to_csv(symbol, statistics, output_file="MAC_Comparison_DATA_누적.csv"):
    # Set the specific path for the output file
    output_dir = "./crypto_data/TS_Difference/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    output_file = os.path.join(output_dir, output_file)  # Set full path for the file

    # Prepare data for saving
    rows = []
    for digit_type, stats in statistics.items():
        rows.append({
            "Symbol": symbol,
            "Digit Type": digit_type.capitalize(),
            "Mean": stats["mean"],
            "Std Dev": stats["std_dev"]
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Append to existing CSV or create a new one
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(output_file, index=False)
    abs_path = os.path.abspath(output_file)  # Get absolute path
    print(f"Statistics saved to {abs_path}")

def main():
    # User inputs
    symbols = input("Enter the symbols (comma-separated, e.g., BTC,ETH,XRP): ").strip().upper().split(",")
    start_datetime = "2024-07-01-00:00"
    end_datetime = "2025-01-01-00:00"
    term_days = "3"
    digit_selection = "both"
    analysis_target = "TA"

    # Define styles for each digit type
    styles = {
        "first": {"color": "blue", "marker": "x", "markersize": 2, "linestyle": "-", "linewidth": 0.7, "label": "FD MAC Differ"},
        "second": {"color": "orange", "marker": "o", "markersize": 2, "linestyle": "-", "linewidth": 0.7, "label": "SD MAC Differ"}
    }

    # Loop through each symbol and perform the same operation
    for symbol in symbols:
        symbol = symbol.strip()  # Remove any extra whitespace
        symbol_1 = symbol + "USDT"
        symbol_2 = "KRW-" + symbol
        exchange_1 = "Binance"
        exchange_2 = "Upbit"

        # Construct file paths
        exchange_paths = [
            construct_file_path(exchange_1.strip(), symbol_1.strip(), analysis_target, start_datetime, end_datetime, term_days),
            construct_file_path(exchange_2.strip(), symbol_2.strip(), analysis_target, start_datetime, end_datetime, term_days)
        ]

        # Load and prepare data
        exchange_data = load_and_prepare_data(exchange_paths)

        # Fetch price data from the first exchange
        price_data = fetch_price_data_binance(symbol_1.strip(), start_datetime, end_datetime)

        # Calculate MAC difference
        mac_difference_data = calculate_mac_difference(exchange_data)

        # Calculate statistics and save to CSV
        statistics = calculate_statistics(mac_difference_data)
        save_statistics_to_csv(symbol, statistics, output_file="MAC_Comparison_DATA_누적.csv")

        # Plot MAC and price and save the output
        output_path = f"./crypto_data/TS_Difference/MAC_Comparison_{exchange_1.capitalize()}_{exchange_2.capitalize()}_{symbol_1[:-4]}_{analysis_target}_{start_datetime.replace(':', '_')}_to_{end_datetime.replace(':', '_')}_{digit_selection}_{term_days}day.png"
        plot_mac_and_price(mac_difference_data, price_data, output_path, digit_selection, styles, exchange_1, exchange_2, symbol)
        print(f"Graph saved to {output_path}")


if __name__ == "__main__":
    main()
