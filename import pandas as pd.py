import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Read the spreadsheet
file_path = "C:/Users/nparr/OneDrive/Desktop/ITSCM 180/air traffic-1.xlsx"
data = pd.read_excel(file_path)

# Filter data for years 2003-2022
data_filtered = data[(data['Year'] >= 2003) & (data['Year'] <= 2022)]

# Select relevant columns for regression
X = data_filtered[['Dom_Pax', 'Dom_Flt', 'Dom_RPM', 'Dom_ASM']]
y = data_filtered['Dom_LF']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform multiple regression fit
model = LinearRegression()
model.fit(X_scaled, y)

# Use the model to predict load factor for 2023
data_2023 = data[data['Year'] == 2023][['Dom_Pax', 'Dom_Flt', 'Dom_RPM', 'Dom_ASM']]
data_2023_scaled = scaler.transform(data_2023)
predicted_load_factor = model.predict(data_2023_scaled)

# Extract actual load factor for 2023
actual_load_factor = data[data['Year'] == 2023]['Dom_LF']

# Calculate error statistic (mean absolute percentage error)
mape = mean_absolute_error(actual_load_factor, predicted_load_factor) / actual_load_factor.mean() * 100

# Print the results
print("Predicted Load Factor for 2023:", predicted_load_factor)
print("Actual Load Factor for 2023:", actual_load_factor.values)
print("Mean Absolute Percentage Error: {:.2f}%".format(mape))  # Print MAPE as a percentage
print("Brewers")