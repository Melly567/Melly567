import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the datasets
vixcls_data = pd.read_csv("C:/Users/onwuk/.ipython/VIXCLS.csv")
dexuseu_data = pd.read_csv("C:/Users/onwuk/.ipython/DEXUSEU.csv")
m1sl_data = pd.read_csv("C:/Users/onwuk/.ipython/M1SL.csv")
rbusbis_data = pd.read_csv("C:/Users/onwuk/.ipython/RBUSBIS.csv")
t10yie_data = pd.read_csv("C:/Users/onwuk/.ipython/T10YIE.csv")

#Inspect column names
print("VIXCLS Data Columns:", vixcls_data.columns)
print("DEXUSEU Data Columns:", dexuseu_data.columns)
print("M1SL Data Columns:", m1sl_data.columns)
print("RBUSBIS Data Columns:", rbusbis_data.columns)
print("T10YIE Data Columns:", t10yie_data.columns)

# Inspect the first few rows of each dataset
print("VIXCLS Data Sample:\n", vixcls_data.head())
print("DEXUSEU Data Sample:\n", dexuseu_data.head())
print("M1SL Data Sample:\n", m1sl_data.head())
print("RBUSBIS Data Sample:\n", rbusbis_data.head())
print("T10YIE Data Sample:\n", t10yie_data.head())


# Convert date columns to datetime
vixcls_data['DATE'] = pd.to_datetime(vixcls_data['DATE'])
dexuseu_data['DATE'] = pd.to_datetime(dexuseu_data['DATE'])
m1sl_data['DATE'] = pd.to_datetime(m1sl_data['DATE'])
rbusbis_data['DATE'] = pd.to_datetime(rbusbis_data['DATE'])
t10yie_data['DATE'] = pd.to_datetime(t10yie_data['DATE'])

# Ensure column names are consistent and informative
vixcls_data.rename(columns={'DATE': 'Date', 'VIXCLS': 'VIX'}, inplace=True)
dexuseu_data.rename(columns={'DATE': 'Date', 'DEXUSEU': 'Exchange_Rate'}, inplace=True)
m1sl_data.rename(columns={'DATE': 'Date', 'M1SL': 'M1_Money_Supply'}, inplace=True)
rbusbis_data.rename(columns={'DATE': 'Date', rbusbis_data.columns[1]: 'CPI'}, inplace=True)
t10yie_data.rename(columns={'DATE': 'Date', 'T10YIE': 'TIPS'}, inplace=True)

# Convert columns to numeric, handling non-numeric values
vixcls_data['VIX'] = pd.to_numeric(vixcls_data['VIX'], errors='coerce')
dexuseu_data['Exchange_Rate'] = pd.to_numeric(dexuseu_data['Exchange_Rate'], errors='coerce')
m1sl_data['M1_Money_Supply'] = pd.to_numeric(m1sl_data['M1_Money_Supply'], errors='coerce')
rbusbis_data['CPI'] = pd.to_numeric(rbusbis_data['CPI'], errors='coerce')
t10yie_data['TIPS'] = pd.to_numeric(t10yie_data['TIPS'], errors='coerce')

# Drop rows with NaN values (optional, based on your data cleaning strategy)
vixcls_data.dropna(inplace=True)
dexuseu_data.dropna(inplace=True)
m1sl_data.dropna(inplace=True)
rbusbis_data.dropna(inplace=True)
t10yie_data.dropna(inplace=True)

# Recheck the data
print("Cleaned VIXCLS Data Sample:\n", vixcls_data.head())
print("Cleaned DEXUSEU Data Sample:\n", dexuseu_data.head())
print("Cleaned M1SL Data Sample:\n", m1sl_data.head())
print("Cleaned RBUSBIS Data Sample:\n", rbusbis_data.head())
print("Cleaned T10YIE Data Sample:\n", t10yie_data.head())

# Plot M1 Money Supply
plt.figure(figsize=(10, 6))
plt.plot(m1sl_data['Date'], m1sl_data['M1_Money_Supply'], label='M1 Money Supply')
plt.title('M1 Money Supply Over Time')
plt.xlabel('Date')
plt.ylabel('M1 Money Supply')
plt.legend()
plt.savefig(r'C:\Users\onwuk\.ipython\M1_Money_Supply.png')
plt.show()

# Plot CPI (Inflation)
plt.figure(figsize=(10, 6))
plt.plot(rbusbis_data['Date'], rbusbis_data['CPI'], label='CPI (Inflation)', color='orange')
plt.title('CPI (Inflation) Over Time')
plt.xlabel('Date')
plt.ylabel('CPI')
plt.legend()
plt.savefig(r'C:\Users\onwuk\.ipython\CPI_Inflation.png')
plt.show()

dexuseu_data = pd.read_csv("C:/Users/onwuk/.ipython/DEXUSEU.csv")

# Convert date columns to datetime
dexuseu_data['DATE'] = pd.to_datetime(dexuseu_data['DATE'])

# Ensure column names are consistent and informative
dexuseu_data.rename(columns={'DATE': 'Date', 'DEXUSEU': 'Exchange_Rate'}, inplace=True)

# Check the data for any inconsistencies
print(dexuseu_data.head())
print(dexuseu_data.describe())

# Plot Exchange Rate with Improved Clarity
plt.figure(figsize=(10, 6))
plt.plot(dexuseu_data['Date'], dexuseu_data['Exchange_Rate'], label='Exchange Rate (USD to EUR)', color='green')
plt.title('Exchange Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Exchange Rate (USD to EUR)')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\onwuk\.ipython\Exchange_Rate.png')
plt.show()

# Plot Stock Market Volatility Index (VIX)
plt.figure(figsize=(10, 6))
plt.plot(vixcls_data['Date'], vixcls_data['VIX'], label='VIX', color='red')
plt.title('Stock Market Volatility Index (VIX) Over Time')
plt.xlabel('Date')
plt.ylabel('VIX')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\onwuk\.ipython\VIX.png')
plt.show()

# Inspect TIPS data columns to find the correct column name
print("T10YIE Data Columns:", t10yie_data.columns)

# Assuming the correct column name for TIPS is now known, replace 'T10YIE' with the correct column name if needed.
# Plot Treasury Inflation-Protected Securities (TIPS)
plt.figure(figsize=(10, 6))
plt.plot(t10yie_data['Date'], t10yie_data['TIPS'], label='TIPS (10 Year)', color='purple')
plt.title('TIPS (10 Year) Over Time')
plt.xlabel('Date')
plt.ylabel('TIPS')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\onwuk\.ipython\TIPS.png')
plt.show()

# Merge datasets on Date
merged_data = pd.merge(m1sl_data, dexuseu_data, on='Date', how='inner')
merged_data = pd.merge(merged_data, rbusbis_data, on='Date', how='inner')
merged_data = pd.merge(merged_data, vixcls_data, on='Date', how='inner')
merged_data = pd.merge(merged_data, t10yie_data, on='Date', how='inner')

# Calculate correlation matrix
correlation_matrix = merged_data.corr()

# Display the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Economic Variables')
plt.savefig(r'C:\Users\onwuk\.ipython\Correlation_Matrix.png')
plt.show()

# Shift M1 Money Supply by one period forward and backward
merged_data['M1_Money_Supply_Lagged_1'] = merged_data['M1_Money_Supply'].shift(1)
merged_data['M1_Money_Supply_Lead_1'] = merged_data['M1_Money_Supply'].shift(-1)

# Calculate correlations
lagged_correlation = merged_data[['M1_Money_Supply_Lagged_1', 'Exchange_Rate', 'VIX', 'CPI', 'TIPS']].corr()
lead_correlation = merged_data[['M1_Money_Supply_Lead_1', 'Exchange_Rate', 'VIX', 'CPI', 'TIPS']].corr()

# Display lagged correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(lagged_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Lagged Correlation Matrix (M1 Money Supply Lagged by 1 Period)')
plt.savefig(r'C:\Users\onwuk\.ipython\Lagged_Correlation_Matrix.png')
plt.show()

# Display lead correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(lead_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Lead Correlation Matrix (M1 Money Supply Leading by 1 Period)')
plt.savefig(r'C:\Users\onwuk\.ipython\Lead_Correlation_Matrix.png')
plt.show()

# Calculate productivity for the merged dataset
merged_data['Productivity'] = merged_data['VIX'] / (merged_data['M1_Money_Supply'] * merged_data['CPI'])

# Plot the productivity over time
plt.figure(figsize=(14, 7))
plt.plot(merged_data['Date'], merged_data['Productivity'], label='Productivity')
plt.title('Productivity Over Time')
plt.xlabel('Date')
plt.ylabel('Productivity')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\onwuk\.ipython\Productivity.png')
plt.show()

