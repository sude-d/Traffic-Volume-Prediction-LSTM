import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
df["date_time"]=pd.to_datetime(df["date_time"])
df.set_index("date_time", inplace=True)
#aylık
plt.figure()
plt.plot( df["traffic_volume"],label="Traffic Volume",color="steelblue")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.title("Traffic Volume Over Time")
plt.legend()
plt.show()
#saatlik
df["hour"]=df.index.hour
hourly_avg=df.groupby("hour")["traffic_volume"].mean()
plt.figure()
sns.barplot(x=hourly_avg.index, y=hourly_avg.values, palette="viridis")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Traffic Volume")
plt.title("Average Traffic Volume by Hour of the Day")
plt.show()