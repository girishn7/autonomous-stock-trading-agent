import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cluster_market_regimes(df, n_clusters=3):
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['regime'] = kmeans.fit_predict(X_scaled)

    return df, kmeans

def plot_regimes(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df['Close'], label='Close Price')
    for r in df['regime'].unique():
        plt.scatter(df[df['regime'] == r].index,
                    df[df['regime'] == r]['Close'],
                    label=f'Regime {r}', s=10)
    plt.legend()
    plt.title("Market Regimes by KMeans")
    plt.show()
