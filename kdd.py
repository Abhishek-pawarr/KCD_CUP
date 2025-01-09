import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("C:\\Kdd cup 2010\\train.csv", sep='\t')  # Replace with your dataset path

# Convert 'Step Start Time' and 'Step End Time' to datetime
data['Step Start Time'] = pd.to_datetime(data['Step Start Time'], format='%Y-%m-%d %H:%M:%S')
data['Step End Time'] = pd.to_datetime(data['Step End Time'], format='%Y-%m-%d %H:%M:%S')

# Create a new column for the time taken to complete a step
data['Step Duration'] = (data['Step End Time'] - data['Step Start Time']).dt.total_seconds()

# Fill any missing values if necessary (e.g., Step Duration)
data.fillna({'Step Duration': 0}, inplace=True)

# Preview the dataset
print(data.head())