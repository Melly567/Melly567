#!/usr/bin/env python
# coding: utf-8

# # MELISSA ONWUKA

# # HNG TASK 2 DATA ANALYSIS FLOOD PREDICTION

# In[37]:


import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Rainfall Data in Lagos.csv')

# Display the first few rows to inspect the columns
print(data.head())

# Check for 'date' column presence
print(data.columns)

# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Ensure 'date' column exists and is properly formatted
if 'date' not in data.columns:
    raise KeyError("'date' column is missing in the dataset")


# In[38]:


# Set the date as index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Explicitly set the frequency to daily
data = data.asfreq('D')

# Handle missing values
data.fillna(method='ffill', inplace=True)


# In[39]:


# Time-Series Forecasting using SARIMA for precipitation
sarima_model = SARIMAX(data['prcp'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

# Forecast for the next 30 days
forecast_steps = 30
forecast = sarima_model.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='D')[1:]
forecast_data = forecast.predicted_mean


# In[40]:


# Create a DataFrame for the forecasted data
future_data = pd.DataFrame({'date': forecast_index, 'prcp': forecast_data})
future_data.set_index('date', inplace=True)

# Define a threshold for flood prediction
future_data['flood'] = (future_data['prcp'] > 10).astype(int)


# In[41]:


# Create a DataFrame for the forecasted data
future_data = pd.DataFrame({'date': forecast_index, 'prcp': forecast_data})
future_data.set_index('date', inplace=True)

# Define a threshold for flood prediction
future_data['flood'] = (future_data['prcp'] > 10).astype(int)

# Find the date with the highest flood probability
if future_data['flood'].sum() > 0:
    predicted_flood_date = future_data[future_data['flood'] == 1].index.min()
else:
    predicted_flood_date = "None"

predicted_flood_date, future_data['prcp'].max()


# In[42]:


# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(data['prcp'], label='Historical Precipitation')
plt.plot(future_data['prcp'], label='Forecasted Precipitation')
plt.axhline(y=10, color='r', linestyle='--', label='Flood Threshold')
plt.legend()
plt.title('Precipitation Forecast')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.tight_layout()
plt.savefig('Rainfall Data in Lagos.png')
plt.close()


# In[45]:


from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 16)
        self.cell(0, 10, 'HNG Task 2 : Flood Prediction Report for Lagos', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# Create a PDF document
pdf = PDF()
pdf.add_page()

# Add content
pdf.set_font("Times", size=12)

# General Introduction
pdf.multi_cell(0, 10, txt="""
The model was trained using historical weather data for Lagos. The following features were considered:
- Average Temperature (tavg)
- Minimum Temperature (tmin)
- Maximum Temperature (tmax)
- Precipitation (prcp)
- Wind Direction (wdir)
- Wind Speed (wspd)
- Air Pressure (pres)

The target variable was defined as a binary indicator of flood occurrence, with a threshold of precipitation greater than 10mm indicating a flood.
""")

# Analysis of Results
pdf.multi_cell(0, 10, txt="""
After training the model on the entire dataset, predictions were made for the next 30 days using the SARIMA model.
The model's predictions indicate the likelihood of a flood based on forecasted precipitation.

Detailed results and analysis:

1. Historical Precipitation: Shows the historical data used to train the model.
2. Forecasted Precipitation: Predicts the next 30 days based on historical trends.
3. Flood Threshold: Indicates the level above which a flood is likely.

Summary of Predictions:
""")

if predicted_flood_date:
    pdf.multi_cell(0, 10, txt=f"""
- The date with the highest predicted probability of a flood is: {predicted_flood_date.date()}
- The forecasted precipitation on this date is: {future_data['prcp'][predicted_flood_date]:.2f} mm
- The maximum forecasted precipitation in the next 30 days is: {future_data['prcp'].max():.2f} mm
    """)
else:
    pdf.multi_cell(0, 10, txt="No flood predicted in the next 30 days.")

# Add the forecast plot to the PDF
pdf.image('Rainfall Data in Lagos.png', x=10, y=140, w=190)

# Add an appendix section with code
pdf.add_page()
pdf.set_font("Times", size=16)
pdf.cell(0, 10, 'Appendix: Code Used', 0, 1, 'C')

# Add the code used in the report
code_str = """
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from fpdf import FPDF

# Load the dataset
file_path = 'Rainfall Data in Lagos.csv'
data = pd.read_csv(file_path)

# Ensure 'date' column exists and is properly formatted
if 'date' not in data.columns:
    raise KeyError("'date' column is missing in the dataset")

data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Explicitly set the frequency to daily
data = data.asfreq('D')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Time-Series Forecasting using SARIMA for precipitation
sarima_model = SARIMAX(data['prcp'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

# Forecast for the next 30 days
forecast_steps = 30
forecast = sarima_model.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast_data = forecast.predicted_mean

# Create a DataFrame for the forecasted data
future_data = pd.DataFrame({'date': forecast_index, 'prcp': forecast_data})
future_data.set_index('date', inplace=True)

# Define a threshold for flood prediction
future_data['flood'] = (future_data['prcp'] > 10).astype(int)

# Find the date with the highest flood probability
if future_data['flood'].sum() > 0:
    predicted_flood_date = future_data[future_data['flood'] == 1].index.min()
else:
    predicted_flood_date = None

predicted_flood_date, future_data['prcp'].max()

# Plot the forecast with figure size 12,6 for better clarity
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['prcp'], label='Historical Precipitation')
plt.plot(future_data.index, future_data['prcp'], label='Forecasted Precipitation', linestyle='dashed')
plt.axhline(y=10, color='r', linestyle='--', label='Flood Threshold')
plt.legend()
plt.title('Precipitation Forecast')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.tight_layout()
plt.savefig('Rainfall Data in Lagos.png')
plt.close()
"""
pdf.set_font("Times", size=10)
pdf.multi_cell(0, 10, txt=code_str)

# Save the PDF
pdf_output_path = 'Rainfall Data in Lagos.pdf'
pdf.output(pdf_output_path)

pdf_output_path


# In[ ]:




