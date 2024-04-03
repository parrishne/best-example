import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Read the spreadsheet
file_path = "C:/Users/nparr/OneDrive/Desktop/ITSCM 180/air traffic-1.xlsx"
df = pd.read_excel(file_path)

# Data preprocessing
# Select relevant columns
df = df[['Year', 'Month', 'Dom_Pax', 'Dom_Flt', 'Dom_RPM', 'Dom_ASM', 'Dom_LF']]
# Filter data for years 2003-2019 and 2021-2022
df = df[((df['Year'] >= 2003) & (df['Year'] <= 2019)) | ((df['Year'] >= 2021) & (df['Year'] <= 2022))]
# Convert month and year columns to numeric
df['Year'] = pd.to_numeric(df['Year'])
df['Month'] = pd.to_numeric(df['Month'])
# Create a combined date column
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
# Drop original year and month columns
df.drop(['Year', 'Month'], axis=1, inplace=True)

# Store feature names
feature_names = df.drop(['Dom_LF', 'Date'], axis=1).columns.tolist()

# Splitting data for training and testing
# We'll use 80% of the data for training and 20% for testing
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_data.drop(['Dom_LF', 'Date'], axis=1))
scaled_test_data = scaler.transform(test_data.drop(['Dom_LF', 'Date'], axis=1))

# Fitting the multiple regression model
regressor = LinearRegression()
regressor.fit(scaled_train_data, train_data['Dom_LF'])

# Forecasting domestic load factor for 9 months in 2023
# Create data for 2023
months_2023 = pd.date_range(start='2023-01-01', end='2023-09-01', freq='MS')
df_2023 = pd.DataFrame({'Date': months_2023})
# Scale the data
scaled_2023_data = scaler.transform(df_2023[feature_names])
# Predict domestic load factor for 2023
df_2023['Predicted_LF'] = regressor.predict(scaled_2023_data)

# Merge with actual data for comparison
actual_2023_data = df[df['Date'].dt.year == 2023]
merged_data = pd.merge(df_2023, actual_2023_data, on='Date', how='left')

# Calculating percentage error
merged_data['Percentage_Error'] = abs(merged_data['Dom_LF'] - merged_data['Predicted_LF']) / merged_data['Dom_LF'] * 100

# Displaying the results
print("Forecasted and Actual Domestic Load Factor for 2023 (Excluding 2020):")
print(merged_data[['Date', 'Dom_LF', 'Predicted_LF', 'Percentage_Error']])
print("\nMean Absolute Percentage Error:", merged_data['Percentage_Error'].mean())
