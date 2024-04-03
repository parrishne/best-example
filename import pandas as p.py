import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Load the data
df = pd.read_excel(r'C:\Users\Owner\OneDrive\Data\Whitewater\Spring 2024\ITSCM 180\
Assignments\5\sales_data.xlsx') # Update this to your spreadsheet's actual path
# Ensure the date column is in datetime format
df['date'] = pd.to_datetime(df['date'])
# Extract year, month, and other relevant information
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
# Filter for 2023 data
sales_2023 = df[df['year'] == 2023]
# Get unique product categories
product_categories = sales_2023['product_category'].unique()
# Initialize a dictionary to store percentage errors for each product category
percentage_errors = {}
# Loop through each product category
for category in product_categories:
# Filter the dataset for the current category
category_data = sales_2023[sales_2023['product_category'] == category]
# Aggregate sales by month
monthly_sales =
category_data.groupby(['month']).agg(total_sales=('sales_amount',
'sum')).reset_index()
# Prepare features and target variable
X = monthly_sales[['month']]
y = monthly_sales['total_sales']
# Since we're only using month as a feature, we only need to scale this feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
test_size=0.25, random_state=42)
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Calculate the mean absolute error and percentage error
error = mean_absolute_error(y_test, y_pred)
percentage_error = (error / y_test.mean()) * 100
# Store the percentage error for the current product category
percentage_errors[category] = percentage_error
# Print the percentage errors for each product category
for category, error in percentage_errors.items():
print(f'Percentage Error for {category}: {error:.2f}%')
