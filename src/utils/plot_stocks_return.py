import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

def plot_daily_returns(stocks_dir, target_companies, start_date, end_date):
    """
    Plot daily returns (Close - Open) for selected stocks within a given date range.

    Args:
        stocks_dir (str): Directory containing stock data files.
        target_companies (list): stock symbols.
        start_date (str): Start date
        end_date (str): End date
    """

    # ===== Font & style settings =====
    rcParams["font.family"] = "Roboto Slab"
    rcParams["axes.unicode_minus"] = False
    sns.set_style("white") 

    returns_data = {}

    for company in target_companies:
        file_path = os.path.join(stocks_dir, f"{company}.csv")
        if os.path.exists(file_path):
            stock_data = pd.read_csv(file_path)
            if {"Date", "Open", "Close"}.issubset(stock_data.columns):
                stock_data["Date"] = pd.to_datetime(stock_data["Date"])
                filtered_data = stock_data[
                    (stock_data["Date"] >= start_date) &
                    (stock_data["Date"] <= end_date)
                ].copy()

                filtered_data["Daily Return"] = filtered_data["Close"] - filtered_data["Open"]
                returns_data[company] = filtered_data.set_index("Date")["Daily Return"]
            else:
                print(f"Missing required columns in {file_path}, skipping.")
        else:
            print(f"File not found: {file_path}, skipping.")

    returns_df = pd.DataFrame(returns_data)

    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    plt.figure(figsize=(14, 8), dpi=300)

    for i, company in enumerate(target_companies):
        if company in returns_df:
            plt.plot(
                returns_df.index,
                returns_df[company],
                label=company,
                linestyle=line_styles[i % len(line_styles)],
                marker=markers[i % len(markers)],
                linewidth=2,
                markersize=5
            )

    plt.title("Daily Returns (Close - Open) for Selected Stocks",
              fontsize=18, fontweight="bold")
    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Daily Return", fontsize=20)
    plt.legend(
        title="Stocks",
        title_fontsize=12,
        fontsize=11,
        loc="upper left",
        frameon=True
    )

    plt.minorticks_on()

    plt.tick_params(
        axis='both',
        which='major',
        direction='in',
        length=8,
        width=1.2,
        bottom=True,
        left=True
    )

    plt.tick_params(
        axis='both',
        which='minor',
        direction='in',
        length=4,
        width=0.8
    )

    import matplotlib.dates as mdates
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    if not returns_df.empty:
        date_range = returns_df.index
        num_dates = len(date_range)
        if num_dates <= 15:
            ax.set_xticks(date_range)
        else:
            step = max(1, num_dates // 10)
            selected_dates = list(date_range[::step])
            if date_range[-1] not in selected_dates:
                selected_dates.append(date_range[-1])
            ax.set_xticks(selected_dates)
    
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    actual_returns = {}
    for company in target_companies:
        if company in returns_df and not returns_df[company].empty:
            actual_returns[company] = returns_df[company].iloc[-1]
    
    return actual_returns


def plot_daily_returns_comparison(stocks_dir, target_companies, start_date1, end_date1, start_date2, end_date2):
    """
    Plot daily returns (Close - Open) for selected stocks within two date ranges side by side.

    Args:
        stocks_dir (str): Directory containing stock data files.
        target_companies (list): stock symbols.
        start_date1 (str): Start date.
        end_date1 (str): End date.
        start_date2 (str): Start date.
        end_date2 (str): End date.
    
    Returns:
        dict: Dictionary with company names as keys and their actual returns as values.
    """

    returns_data1 = {}
    returns_data2 = {}

    try:
        next_day = (pd.to_datetime(start_date2) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        next_day_returns = []
        company_returns = {}
        
        for company in target_companies:
            file_path = os.path.join(stocks_dir, f"{company}.csv")
            if os.path.exists(file_path):
                stock_data = pd.read_csv(file_path)
                if {"Date", "Open", "Close"}.issubset(stock_data.columns):
                    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
                    # Filter for the next day
                    next_day_data = stock_data[stock_data["Date"] == next_day]
                    if not next_day_data.empty:
                        daily_return = next_day_data["Close"].iloc[0] - next_day_data["Open"].iloc[0]
                        next_day_returns.append(daily_return)
                        company_returns[company] = daily_return
        
        if next_day_returns:
            mean_return = pd.Series(next_day_returns).mean()
            std_return = pd.Series(next_day_returns).std()
            print(f"\n{'='*60}")
            print(f"Statistics for {next_day} (day after start_date2):")
            print(f"{'='*60}")
            print(f"\nIndividual Stock/ETF Returns (Close - Open):")
            for company, ret in company_returns.items():
                print(f"  {company:8s}: {ret:8.4f}")
            print(f"\n{'-'*60}")
            print(f"Market Summary:")
            print(f"  Average Return: {mean_return:.4f}")
            print(f"  Std Deviation:  {std_return:.4f}")
            print(f"  Number of stocks: {len(next_day_returns)}")
            print(f"{'='*60}\n")
        else:
            print(f"No data available for {next_day}")
    except Exception as e:
        print(f"Error computing statistics for day after start_date2: {e}")

    def process_data(start_date, end_date, returns_data):
        for company in target_companies:
            file_path = os.path.join(stocks_dir, f"{company}.csv")
            if os.path.exists(file_path):
                stock_data = pd.read_csv(file_path)
                if {"Date", "Open", "Close"}.issubset(stock_data.columns):
                    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
                    # Filter by date range
                    filtered_data = stock_data[(stock_data["Date"] >= start_date) & (stock_data["Date"] <= end_date)].copy()
                    # Calculate daily returns
                    filtered_data.loc[:, "Daily Return"] = filtered_data["Close"] - filtered_data["Open"]
                    returns_data[company] = filtered_data.set_index("Date")["Daily Return"]
                else:
                    print(f"Missing required columns in {file_path}, skipping.")
            else:
                print(f"File not found: {file_path}, skipping.")

    # Process data for both date ranges
    process_data(start_date1, end_date1, returns_data1)
    process_data(start_date2, end_date2, returns_data2)

    returns_df1 = pd.DataFrame(returns_data1)
    returns_df2 = pd.DataFrame(returns_data2)

    sns.set_theme(style="whitegrid")
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    uniform_dates1 = pd.date_range(start=start_date1, end=end_date1, freq='D')
    uniform_dates2 = pd.date_range(start=start_date2, end=end_date2, freq='D')

    tick_interval1 = max(1, len(uniform_dates1) // 10)  # Adjust interval based on date range
    tick_interval2 = max(1, len(uniform_dates2) // 10)

    # Plot first date range
    for i, company in enumerate(target_companies):
        if company in returns_df1:
            axes[0].plot(returns_df1.index, returns_df1[company], label=company, linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)], linewidth=2, markersize=6)
    axes[0].set_title(f"Daily Returns ({start_date1} to {end_date1})", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Date", fontsize=12)
    axes[0].set_ylabel("Actual Daily Return", fontsize=12)
    axes[0].legend(title="Stocks", title_fontsize=10, fontsize=8, loc="lower right", frameon=True)
    axes[0].grid(visible=True, linestyle="--", alpha=0.6)
    axes[0].set_xticks(uniform_dates1[::tick_interval1])  # Set dynamic x-axis ticks
    axes[0].set_xticklabels(uniform_dates1[::tick_interval1].strftime('%Y-%m-%d'), rotation=45, fontsize=10)

    for i, company in enumerate(target_companies):
        if company in returns_df2:
            axes[1].plot(returns_df2.index, returns_df2[company], label=company, linestyle=line_styles[i % len(line_styles)], marker=markers[i % len(markers)], linewidth=2, markersize=6)
    axes[1].set_title(f"Actual Daily Returns on Next Day ({start_date2} to {end_date2})", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].legend(title="Stocks", title_fontsize=10, fontsize=8, loc="lower right", frameon=True)
    axes[1].grid(visible=True, linestyle="--", alpha=0.6)
    axes[1].set_xticks(uniform_dates2[::tick_interval2])  # Set dynamic x-axis ticks
    axes[1].set_xticklabels(uniform_dates2[::tick_interval2].strftime('%Y-%m-%d'), rotation=45, fontsize=10)

    plt.tight_layout()
    plt.show()
    
    return company_returns