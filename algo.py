# Trading algorithm dependencies
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting import Strategy, Backtest
from tqdm import tqdm

# Note: If you're running this for the first time, install the required packages with:
# pip install -r requirements.txt

# -------------------------------------------------------
# LOADING UP THE MARKET DATA
# -------------------------------------------------------
def read_csv_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        # Fixing the time format so pandas can read it properly
        if "Gmt time" in df.columns:
            df["Gmt time"] = df["Gmt time"].astype(str).str.replace(".000", "", regex=False)
            df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S')
            df.set_index("Gmt time", inplace=True)
        
        # Getting rid of candles where price didn't move (no point in keeping these)
        df = df[df.High != df.Low]
        
        # Making sure we have all the price data we need
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {file_path}: Missing OHLC columns.")
            return None
            
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def get_data_from_folder():
    """
    Just grabs all the CSV files from the data folder
    """
    folder_path = "."
    dataframes = []
    file_names = []

    # Create the data folder if it doesn't exist yet
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"--- Created '{folder_path}' folder for you ---")
        print("Just drop your CSV files in there and run this again!")
        return [], []

    # Find all the CSV files
    files_found = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    if not files_found:
        print(f"No CSV files in '{folder_path}'. Add some price data and try again.")
        return [], []

    print(f"Found {len(files_found)} files. Let's load them up...")
    for file_name in tqdm(files_found):
        file_path = os.path.join(folder_path, file_name)
        df = read_csv_to_dataframe(file_path)
        if df is not None:
            dataframes.append(df)
            file_names.append(file_name)
    
    return dataframes, file_names

# -------------------------------------------------------
# THE ACTUAL TRADING STRATEGY
# -------------------------------------------------------
def SMA(series, period):
    return pd.Series(series).rolling(window=period).mean().values

def RSI(series, period=14):
    delta = pd.Series(series).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

class ProfessionalStrategy(Strategy):
    # These settings seem to work pretty well
    slperc = 0.01      # Stop loss at 1%
    tpperc = 0.02      # Take profit at 2%
    ma_period = 20
    rsi_period = 14
    rsi_upper = 70
    rsi_lower = 30
    risk_pct = 0.01    # Risk 1% on each trade

    def init(self):
        self.sma = self.I(SMA, self.data.Close, period=self.ma_period)
        self.rsi = self.I(RSI, self.data.Close, period=self.rsi_period)

    def next(self):
        price = self.data.Close[-1]
        ma = self.sma[-1]
        rsi = self.rsi[-1]
        
        # Time to enter a trade?
        if not self.position:
            # Looking to buy: Price > MA (Trend) AND RSI < 30 (Oversold Pullback)
            if price > ma and rsi < self.rsi_lower:
                sl = price * (1 - self.slperc)
                tp = price * (1 + self.tpperc)
                self.calculate_entry('long', sl, tp)
                
            # Looking to sell: Price < MA (Trend) AND RSI > 70 (Overbought Pullback)
            elif price < ma and rsi > self.rsi_upper:
                sl = price * (1 + self.slperc)
                tp = price * (1 - self.tpperc)
                self.calculate_entry('short', sl, tp)

    def calculate_entry(self, direction, sl, tp):
        price = self.data.Close[-1]
        risk_per_share = abs(price - sl)
        
        if risk_per_share == 0: return

        # Figure out how big our position should be
        risk_amount = self.equity * self.risk_pct
        size = risk_amount / risk_per_share
        
        size = int(size) if size >= 1 else 0
        
        if size > 0:
            if direction == 'long':
                self.buy(size=size, sl=sl, tp=tp)
            else:
                self.sell(size=size, sl=sl, tp=tp)

# -------------------------------------------------------
# LET'S SEE HOW THIS PERFORMS
# -------------------------------------------------------
def run_backtest(dataframes, file_names):
    results = []
    for i, df in enumerate(dataframes):
        print(f"\n--- Testing {file_names[i]} ---")
        bt = Backtest(df, ProfessionalStrategy, cash=10000, commission=0.0002)
        
        # Try to find the best settings
        try:
            stats = bt.optimize(
                slperc=[0.01, 0.015, 0.02],
                tpperc=[0.01, 0.02, 0.03, 0.04],
                ma_period=[20, 50, 100, 200],
                maximize='SQN',
                max_tries=50,
                random_state=0
            )
            results.append(stats)
            print(stats)
            
            # Show the interactive chart for the first dataset as an example
            if i == 0:
                print("\n--- Opening interactive dashboard in browser ---")
                bt.plot(open_browser=True)
                
        except Exception as e:
            print(f"Optimization didn't work for {file_names[i]}: {e}")
            print("Just running the basic test instead...")
            stats = bt.run()
            print(stats)
        
    return results

# --- LET'S GET STARTED ---
if __name__ == "__main__":
    # 1. Load up the data
    dataframes, file_names = get_data_from_folder()
    
    if dataframes:
        # 2. Run the backtest
        run_backtest(dataframes, file_names)
    else:
        # No data found, nothing to do
        pass