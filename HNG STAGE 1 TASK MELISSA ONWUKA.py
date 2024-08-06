#!/usr/bin/env python
# coding: utf-8

# # Melissa Onwuka
# HNG INTERNSHIP DATA ANALYSIS STAGE 1 TASK

# # Load and Filter Datasets

# In[3]:


import pandas as pd

# Load the datasets
election_results_path = 'ABIA_crosschecked.csv'
polling_units_path = 'polling-units.csv'

election_df = pd.read_csv(election_results_path)
polling_units_df = pd.read_csv(polling_units_path)

# Filter polling units to only include those from Abia state
polling_units_df = polling_units_df[polling_units_df['state_name'] == 'ABIA']

# Display the first few rows to understand their structure
election_df.head(), polling_units_df.head()


# # Merge Datasets

# In[4]:


# Rename columns in polling_units_df to match with election_df
polling_units_df = polling_units_df.rename(columns={
    'name': 'PU-Name',
    'location.latitude': 'Latitude',
    'location.longitude': 'Longitude'
})

# Merge the datasets on the common column 'PU-Name'
merged_df = pd.merge(election_df, polling_units_df, on='PU-Name')

# Display the merged dataframe
merged_df.head()


# # Use Neighbors using KDTree

# In[5]:


import numpy as np
from scipy.spatial import KDTree

# Extract latitude and longitude values
coords = merged_df[['Latitude', 'Longitude']].values

# Create a KDTree for fast spatial queries
tree = KDTree(coords)

# Function to find neighbors within a radius using KDTree
def find_neighbors_kdtree(tree, coords, radius_km):
    neighbors = {}
    for i, coord in enumerate(coords):
        # Use query_ball_point to find all points within the specified radius
        indices = tree.query_ball_point(coord, radius_km / 6371.0)  # Earth's radius in km
        # Store the neighbors, excluding the point itself
        neighbors[merged_df.iloc[i]['PU-Code']] = [merged_df.iloc[j]['PU-Code'] for j in indices if j != i]
    return neighbors

# Define a 1 km radius for neighbor identification
radius_km = 1
neighbors_dict = find_neighbors_kdtree(tree, coords, radius_km)

# Display the neighbors dictionary for verification
list(neighbors_dict.items())[:5]


# # Calculate Outlier Scores

# In[18]:


# Function to calculate outlier scores
def calculate_outlier_scores(df, neighbors_dict):
    outlier_scores = []
    for polling_unit, neighbors in neighbors_dict.items():
        for party in ['APC', 'LP', 'PDP', 'NNPP']:  # Replace with actual party names if different
            try:
                # Get the votes for the current polling unit
                votes = df.loc[df['PU-Code'] == polling_unit, party].values[0]
                if len(neighbors) > 0:
                    # Calculate the mean votes for the neighbors
                    neighbor_votes = df[df['PU-Code'].isin(neighbors)][party].mean()
                    # Calculate the outlier score as the absolute difference
                    outlier_score = abs(votes - neighbor_votes)
                    # Append the results to the list
                    outlier_scores.append({'PollingUnit': polling_unit, 'Party': party, 'OutlierScore': outlier_score})
            except Exception as e:
                print(f"Error processing {polling_unit} for {party}: {e}")
    return pd.DataFrame(outlier_scores)

# Calculate outlier scores
outlier_scores_df = calculate_outlier_scores(merged_df, neighbors_dict)
outlier_scores_df.head()



# # Sort and Save results

# In[11]:


pip install openpyxl


# In[12]:


from openpyxl.workbook import Workbook


# In[13]:


# Sort by outlier scores
sorted_outliers = outlier_scores_df.sort_values(by='OutlierScore', ascending=False)

# Save the results to CSV for reporting
sorted_outliers.to_csv('sorted_outliers_abia.csv', index=False)

# Save to Excel for further reporting
sorted_outliers.to_excel('sorted_outliers_abia.xlsx', index=False)

# Display the top 3 outliers
top_3_outliers = sorted_outliers.head(3)
print(top_3_outliers)


# # Visualization

# In[20]:


pip install geopandas


# In[21]:


import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point


# In[22]:


# Convert the DataFrame to a GeoDataFrame
geometry = [Point(xy) for xy in zip(merged_df['Longitude'], merged_df['Latitude'])]
gdf = gpd.GeoDataFrame(merged_df, geometry=geometry)

# Plotting all polling units
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
gdf.plot(ax=ax, color='blue', markersize=5, label='Polling Units')

# Highlight the top 3 outliers
top_3_outliers_gdf = gdf[gdf['PU-Code'].isin(top_3_outliers['PollingUnit'])]
top_3_outliers_gdf.plot(ax=ax, color='red', markersize=20, label='Top 3 Outliers')

plt.legend()
plt.title('Polling Units and Top 3 Outliers in Abia State')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[23]:


# Calculate outlier scores
outlier_scores_df = calculate_outlier_scores(merged_df, neighbors_dict)

# Save the results to CSV
outlier_scores_df.to_csv('outlier_scores_with_coordinates.csv', index=False)

# Display the first few rows of the resulting DataFrame
outlier_scores_df.head()


# In[25]:


import pandas as pd
import numpy as np
from scipy.spatial import KDTree

# Assuming merged_df and neighbors_dict have been defined in the previous steps

# Function to calculate outlier scores
def calculate_outlier_scores_with_coords(df, neighbors_dict):
    outlier_scores = []
    for polling_unit, neighbors in neighbors_dict.items():
        for party in ['APC', 'LP', 'PDP', 'NNPP']:  # Replace with actual party names if different
            try:
                votes = df.loc[df['PU-Code'] == polling_unit, party].values[0]
                if len(neighbors) > 0:
                    neighbor_votes = df[df['PU-Code'].isin(neighbors)][party].mean()
                    outlier_score = abs(votes - neighbor_votes)
                    outlier_scores.append({
                        'PollingUnit': polling_unit,
                        'Party': party,
                        'OutlierScore': outlier_score,
                        'Latitude': df.loc[df['PU-Code'] == polling_unit, 'Latitude'].values[0],
                        'Longitude': df.loc[df['PU-Code'] == polling_unit, 'Longitude'].values[0]
                    })
            except Exception as e:
                print(f"Error processing {polling_unit} for {party}: {e}")
    return pd.DataFrame(outlier_scores)

# Calculate outlier scores
outlier_scores_with_coords_df = calculate_outlier_scores_with_coords(merged_df, neighbors_dict)

# Save the results to CSV with added longitude and latitude values
outlier_scores_with_coords_df.to_csv('outlier_scores_with_coordinates.csv', index=False)

# Sort the outlier scores
sorted_outliers_df = outlier_scores_with_coords_df.sort_values(by='OutlierScore', ascending=False)

# Save the sorted outlier scores to Excel
sorted_outliers_df.to_excel('sorted_outliers.xlsx', index=False)

# Display the first few rows of the resulting DataFrames
print(outlier_scores_with_coords_df.head())
print(sorted_outliers_df.head())


# In[26]:


import pandas as pd

# Load the sorted outlier scores
sorted_outliers_df = pd.read_excel('sorted_outliers.xlsx')

# Display the first few rows
print(sorted_outliers_df.head())


# In[ ]:




