import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
df=pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
df["date_time"]=pd.to_datetime(df["date_time"])
df.set_index("date_time", inplace=True)
df["hour"]=df.index.hour
df["dayofweek"]=df.index.dayofweek
df["month"]=df.index.month
features=["temp","rain_1h","snow_1h","clouds_all","hour","dayofweek","month"]
target="traffic_volume"
df=df[features+[target]].dropna()
scaler_X=MinMaxScaler()
scaler_y=MinMaxScaler()
X_scaled=scaler_X.fit_transform(df[features])
y_scaled=scaler_y.fit_transform(df[[target]])
joblib.dump(scaler_X, "scaler_X.save")
joblib.dump(scaler_y, "scaler_y.save")
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)
SEQ_LEN = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)
split_idx = int(0.8 * len(X_seq))
X_train= X_seq[:split_idx]
y_train= y_seq[:split_idx]
X_test= X_seq[split_idx:]
y_test= y_seq[split_idx:]
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
