import pandas as pd
import os
import torch

def process_data(category_file, stocks_dir, start_date=None, end_date=None):
    """
    Process stock data and generate a PyTorch tensor aligned by date.

    Args:
        category_file (str): Path to the file containing company symbols.
        stocks_dir (str): Directory containing stock data files.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        tensor (torch.Tensor): Tensor of shape (n, m), where n is the number of dates and m is the number of companies.
        dates (list): List of dates.
        retained_symbols (list): Symbols included in the tensor.
        skipped_symbols (list): Symbols skipped during processing.
    """
    category_data = pd.read_csv(category_file)
    symbols = category_data["Symbol"].tolist()

    all_data = []
    retained_symbols = []
    skipped_symbols = []

    for symbol in symbols:
        stock_file = os.path.join(stocks_dir, f"{symbol}.csv")
        
        if os.path.exists(stock_file):
            stock_data = pd.read_csv(stock_file)
            
            if {"Open", "Close", "Date"}.issubset(stock_data.columns):
                stock_data["Date"] = pd.to_datetime(stock_data["Date"])
                
                if start_date and end_date:
                    stock_data = stock_data[(stock_data["Date"] >= start_date) & (stock_data["Date"] <= end_date)]
                
                if not stock_data.empty:
                    stock_data["PriceDiff"] = stock_data["Close"] - stock_data["Open"]
                    stock_data = stock_data[["Date", "PriceDiff"]].set_index("Date")
                    all_data.append(stock_data)
                    retained_symbols.append(symbol)
                else:
                    skipped_symbols.append(symbol)
            else:
                print(f"Missing required columns in {stock_file}, skipping.")
                skipped_symbols.append(symbol)
        else:
            print(f"File not found: {stock_file}, skipping.")
            skipped_symbols.append(symbol)

    if all_data:
        combined_data = pd.concat(all_data, axis=1, join="outer")
        combined_data.columns = retained_symbols

        nan_count = combined_data.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values found.")
            nan_per_company = combined_data.isna().sum()
            print("NaN counts per company:")
            print(nan_per_company[nan_per_company > 0])
            nan_dates = combined_data.index[combined_data.isna().any(axis=1)].tolist()
            print(f"Dates with NaN values: {nan_dates}")
        else:
            print("Data is complete, no NaN values.")

        combined_data = combined_data.sort_index()
        tensor = torch.tensor(combined_data.values, dtype=torch.float32)
        dates = combined_data.index.strftime('%Y-%m-%d').tolist()
        return tensor, dates, retained_symbols, skipped_symbols
    else:
        print("No valid data found, tensor cannot be created.")
        return None, None, [], symbols


def process_multi_industry(industries, categories_dir, stocks_dir, start_date, end_date, n_companies=50):
    """
    Process multiple industries and generate stacked tensors.

    Args:
        industries (list): List of industry names.
        categories_dir (str): Directory containing industry category files.
        stocks_dir (str): Directory containing stock data files.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        n_companies (int): Number of companies to retain per industry.

    Returns:
        stacked_tensor (torch.Tensor): Stacked tensor of shape (k, n, m), where k is the number of industries.
        industry_names (list): List of industry names.
        industry_company_labels (list): List of company symbols for each industry.
    """
    industry_tensors = []
    industry_names = []
    industry_company_labels = []

    for industry in industries:
        category_file = os.path.join(categories_dir, f"{industry}.csv")
        if os.path.exists(category_file):
            tensor, dates, retained_symbols, skipped_symbols = process_data(
                category_file, stocks_dir, start_date, end_date
            )
            if tensor is not None:
                if tensor.shape[1] >= n_companies:
                    tensor = tensor[:, :n_companies]
                    retained_symbols = retained_symbols[:n_companies]
                industry_tensors.append(tensor)
                industry_names.append(industry)
                industry_company_labels.append(retained_symbols)

    if industry_tensors:
        stacked_tensor = torch.stack(industry_tensors, dim=0)
        return stacked_tensor, industry_names, industry_company_labels
    else:
        return None, [], []


def extract_msar_predictions(msar_module, model_info, filtered_symbols, print_analysis=True):
    """
    Extract MSAR model predictions for each company.
    
    Args:
        msar_module: The imported msar module (src.models.msar)
        model_info (dict): MSAR model output dictionary containing 'summary' with 'exp_next'
        filtered_symbols (list): List of company symbols
        print_analysis (bool): Whether to print detailed analysis for each company
    
    Returns:
        dict: Dictionary mapping company symbol to predicted return (float)
    """
    predictions = {}
    
    for i, company in enumerate(filtered_symbols):
        try:
            if print_analysis:
                print(f"\n{'='*60}")
                print(f"Analysis for {company} (index {i}):")
                print(f"{'='*60}")
                msar_module.analyze_info(model_info, c=0, i=i)
            
            # Extract the prediction value (E[rₜ₊₁ | Y₁:ₜ])
            exp_next = model_info["summary"]["exp_next"]
            
            # Handle different data structures
            if hasattr(exp_next, 'shape'):  # numpy array or torch tensor
                if len(exp_next.shape) == 2:
                    prediction = exp_next[0, i]
                else:
                    prediction = exp_next[i]
            else:  # list or other sequence
                prediction = exp_next[i]
            
            # Convert to scalar
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]
            if hasattr(prediction, 'item'):
                prediction = prediction.item()
            prediction = float(prediction)
            
            predictions[company] = prediction
            
        except Exception as e:
            print(f"Error analyzing {company}: {e}")
            import traceback
            traceback.print_exc()
    
    if print_analysis:
        print("\n" + "="*60)
        print("MSAR Predictions saved:")
        print(predictions)
    
    return predictions


# Example usage
if __name__ == "__main__":
    category_file = r"d:\ada\stocks_categories\health_care.csv"
    stocks_dir = r"d:\ada\archive\stocks"
    start_date = "1997-08-08"
    end_date = "1997-09-08"

    tensor, dates, retained_symbols, skipped_symbols = process_data(category_file, stocks_dir, start_date, end_date)

    if tensor is not None:
        print(f"Tensor shape: {tensor.shape}")
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"Retained symbols: {retained_symbols[:3]}")
    else:
        print("Tensor creation failed.")

    print(f"Skipped symbols: {skipped_symbols}")