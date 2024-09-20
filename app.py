import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Function to create lag features
def create_lag_features(df, target_col, lags=3):
    for lag in range(1, lags + 1):
        df[f'{target_col}_Lag_{lag}'] = df[target_col].shift(lag)
    df = df.dropna()
    return df


# Load dataset (for demo, you would get real-time data in an actual scenario)
@st.cache_data
def load_data():
    # Simulating loaded data, use real-time data in production
    data = pd.read_csv('dataset.csv')  # Ensure CSV has the correct format and columns
    return data


# Get the input data
df = load_data()

# Feature engineering to generate lag features
df = create_lag_features(df, 'Amazon_Price')

# Select the features that were used during model training
selected_features = [
    'Amazon_Price_Lag_1',
    'Amazon_Price_Lag_2',
    'Amazon_Price_Lag_3',
    'Meta_Price',
    'Netflix_Price',
    'Nasdaq_100_Price'
]

# App Title and Description
st.title('Amazon Stock Price Prediction App')
st.write("""
This application uses historical stock prices and market trends to predict the future price of Amazon stock.
Input the latest prices below to get your prediction!
""")

# Sidebar for inputs
st.sidebar.header('User Input Features')

input_data = {}
for feature in selected_features:
    last_value = df[feature].iloc[-1]
    if isinstance(last_value, str):
        last_value = float(last_value.replace(',', ''))  # Remove commas if present
    else:
        last_value = float(last_value)

    # Use a slider instead of a number input
    input_data[feature] = st.sidebar.slider(
        f'Enter value for {feature}',
        min_value=float(last_value) * 0.8,  # Set min value to 80% of last value
        max_value=float(last_value) * 1.2,  # Set max value to 120% of last value
        value=last_value,  # Default to last value
        step=0.01  # Adjust the step for more precision
    )

# Create a DataFrame for the input
input_df = pd.DataFrame([input_data])

# Visualize historical data
st.subheader('Historical Amazon Stock Prices')
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Amazon_Price'], label='Historical Prices', color='blue')
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Amazon Stock Prices Over Time')

# Format x-axis to show only the year
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to every year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format to show only the year

plt.xticks(rotation=45)
plt.legend()
st.pyplot(plt)

# # Perform prediction
# if st.button('Predict Amazon Price'):
#     with st.spinner('Predicting...'):
#         prediction = model.predict(input_df)
#         st.success(f"Predicted Amazon Price: ${prediction[0]:,.2f}")

# Perform prediction
# Perform prediction
if st.button('Predict Amazon Price', key='predict_button'):
    with st.spinner('Predicting...'):
        prediction = model.predict(input_df)
        st.success(f"Predicted Amazon Price: ${prediction[0]:,.2f}")
