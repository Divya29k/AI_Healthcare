import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Sample time-series data
np.random.seed(42)
data = np.random.normal(size=100)
data[50:55] = np.random.normal(loc=5.0, size=5)  # Adding anomalies

# Create a DataFrame
df = pd.DataFrame(data, columns=['value'])
df['time'] = pd.date_range(start='1/1/2023', periods=len(df), freq='D')

# Isolation Forest model
model = IsolationForest(contamination=0.05)
df['anomaly'] = model.fit_predict(df[['value']])

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['value'], label='Value')
plt.scatter(df[df['anomaly'] == -1]['time'], df[df['anomaly'] == -1]['value'], color='red', label='Anomaly')
plt.title('Time-Series Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
