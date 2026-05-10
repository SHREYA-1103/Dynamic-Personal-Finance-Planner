import pandas as pd
import os


# LOAD & CLEAN EQUITY DATA
def load_equity_data(path):
    df = pd.read_csv(path)

    # RENAME COLUMNS
    df = df.rename(columns={"Price": "Close"})

    # DATE PARSING
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

    # CLEAN CLOSE
    df['Close'] = df['Close'].astype(str).str.replace(',', '')
    df['Close'] = df['Close'].astype(float)

    # RETURNS
    df['equity_return'] = df['Change %'].astype(str).str.replace('%', '')
    df['equity_return'] = df['equity_return'].astype(float) / 100

    df = df[['Date', 'Close', 'equity_return']]
    df = df.sort_values('Date')

    return df


def classify_market_type(row, vol_threshold):
    # CRISIS
    if row['drawdown'] <= -0.20 or row['volatility'] > vol_threshold:
        return 'Crisis'

    # BEAR
    elif row['equity_return'] < 0 or row['drawdown'] <= -0.10:
        return 'Bear'

    # BULL
    else:
        return 'Bull'


# ADD MARKET FEATURES
def add_market_features(df, window=24):

    # VOLATILITY
    df['volatility'] = df['equity_return'].rolling(window).std()

    # DRAWDOWN
    df['peak'] = df['Close'].cummax()
    df['drawdown'] = (df['Close'] - df['peak']) / df['peak']

    # DROP NaNs
    df = df.dropna().reset_index(drop=True)

    # REGIME CLASSIFICATION
    vol_threshold = df['volatility'].quantile(0.95)

    df['market_type'] = df.apply(
        lambda x: classify_market_type(x, vol_threshold),
        axis=1
    )

    df = df[['Date', 'equity_return', 'volatility', 'drawdown', 'market_type']]

    return df


# LOAD & CLEAN DEBT DATA
def load_debt_data(path):
    df = pd.read_csv(path)

    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

    df['debt_return'] = df['Change %'].astype(str).str.replace('%', '')
    df['debt_return'] = df['debt_return'].astype(float) / 100

    df = df[['Date', 'debt_return']]
    df = df.sort_values('Date')

    return df


# MERGE DATA
def merge_data(eq_df, debt_df):
    df = pd.merge(eq_df, debt_df, on="Date", how="inner")
    return df


# ADD CASH RETURN
def add_cash_return(df):
    df['cash_return'] = 0.03 / 12  # monthly
    df = df[[
        'Date',
        'equity_return',
        'debt_return',
        'cash_return',
        'volatility',
        'drawdown',
        'market_type'
    ]]
    return df


# FINAL PIPELINE
def data_process_pipeline(equity_path, debt_path):

    # LOAD DATA
    equity_df = load_equity_data(equity_path)
    debt_df = load_debt_data(debt_path)

    # ADD FEATURES
    equity_market_df = add_market_features(equity_df)

    # MERGE
    merged_df = merge_data(equity_market_df, debt_df)

    # ADD CASH
    final_df = add_cash_return(merged_df)

    # ENCODE MARKET REGIME
    mapping = {'Bull': 2, 'Bear': 1, 'Crisis': 0}
    final_df['market_indicator'] = final_df['market_type'].map(mapping)

    # FINAL CLEANUP
    final_df = final_df[[
        'Date',
        'equity_return',
        'debt_return',
        'cash_return',
        'volatility',
        'drawdown',
        'market_indicator'
    ]]

    final_df = final_df.sort_values('Date').reset_index(drop=True)

    # FINAL CHECK
    assert final_df.isna().sum().sum() == 0, "NaNs detected!"

    return final_df


if __name__ == "__main__":
    file_path = "../data/processed/market_data.csv"

    if os.path.exists(file_path):
        print("Processed data already exists")

    else:
        df = data_process_pipeline(
            "../data/raw/EQUITY_nifty_50_historical_data_jan_2005_to_jan_2026.csv",
            "../data/raw/DEBT_indian_bonds_historical_data_jan_2007_to_jan_2026.csv"
        )

        df.to_csv("../data/processed/market_data.csv", index=False)

        print("Processed dataset saved at data/processed/market_data.csv")