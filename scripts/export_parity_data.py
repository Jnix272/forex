import numpy as np
import pandas as pd


def generate_parity_data(n_ticks=100):
    """
    Generates dummy ticks and calculates Python features, then exports 
    them for C++ parity testing.
    """
    np.random.seed(42)
    # Generate random walk prices around 1.1000
    prices = 1.1000 + np.cumsum(np.random.normal(0, 0.0001, n_ticks))

    df = pd.DataFrame({'price': prices})

    # Python Feature Calculations (matching the C++ mock logic)
    # 1. Price
    # 2. SMA(14)
    # 3. EMA(14)

    df['SMA_14'] = df['price'].rolling(window=14).mean()
    df['EMA_14'] = df['price'].ewm(span=14, adjust=False).mean()

    # Fill NAs
    df.fillna(method='bfill', inplace=True)

    # Export raw ticks for C++ to read
    df[['price']].to_csv('parity_ticks.csv', index=False, header=False)

    # Export Python calculated features for C++ to compare against
    df[['price', 'SMA_14', 'EMA_14']].to_csv('parity_features.csv', index=False, header=False)

    print("Exported parity_ticks.csv and parity_features.csv successfully.")

if __name__ == "__main__":
    generate_parity_data()
