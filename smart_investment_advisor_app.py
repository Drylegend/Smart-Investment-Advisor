#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Smart Inverstment Advisor
#Utsav Chatterjee
#Version:v1.0


# In[2]:


#Installing Lybraries
#get_ipython().system('pip install yfinance pandas numpy matplotlib seaborn ta scikit-learn')


# In[3]:


#Importing lybraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ta import add_all_ta_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[4]:


# Downloading data for a stock, e.g., Apple (AAPL)
df = yf.download("AAPL", start="2020-01-01", end="2024-12-31")
df = df.dropna()
df.head()


# In[10]:


import yfinance as yf
import pandas as pd
from ta import add_all_ta_features

# Download multi-index data with group_by
df = yf.download("AAPL", start="2020-01-01", end="2024-12-31", group_by='ticker')

# Flatten the multi-level columns: ('AAPL', 'Open') → 'Open'
df.columns = df.columns.droplevel(0)

# Drop NaN values if any
df = df.dropna()

# Reset index to get 'Date' as a column
df.reset_index(inplace=True)

# Now df.columns are: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

# Keep only the required columns
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Add technical indicators
df = add_all_ta_features(
    df,
    open="Open",
    high="High",
    low="Low",
    close="Close",
    volume="Volume",
    fillna=True
)

# Set 'Date' back as index
df.set_index('Date', inplace=True)

# Preview the data
df.head()


# In[11]:


# Define future prediction window (e.g., 5 days ahead)
future_days = 5

# Shift the closing price by -5 days to get the future price
df['Future_Close'] = df['Close'].shift(-future_days)

# Calculate percentage change from current close to future close
df['Price_Change_%'] = ((df['Future_Close'] - df['Close']) / df['Close']) * 100

# Create label column
def get_label(change):
    if change > 2:
        return 1    # Buy
    elif change < -2:
        return -1   # Sell
    else:
        return 0    # Hold

df['Signal'] = df['Price_Change_%'].apply(get_label)

# Drop rows with NaN in Future_Close (at the end)
df.dropna(inplace=True)

# Preview
df[['Close', 'Future_Close', 'Price_Change_%', 'Signal']].tail(10)


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Feature columns: all the technical indicators (except 'Signal')
X = df.drop(columns=['Signal', 'Future_Close', 'Price_Change_%'])

# Target: 'Signal' column (Buy, Hold, Sell)
y = df['Signal']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[14]:


#Visulization
import matplotlib.pyplot as plt

# Create a new column for predicted signals
df['Predicted_Signal'] = model.predict(X)

# Plot Actual vs Predicted signals
plt.figure(figsize=(10, 6))

# Plotting the Actual Signal
plt.plot(df.index, df['Signal'], label='Actual Signal', linestyle='--', color='blue', alpha=0.7)

# Plotting the Predicted Signal
plt.plot(df.index, df['Predicted_Signal'], label='Predicted Signal', linestyle='-', color='orange', alpha=0.7)

plt.title('Actual vs Predicted Signals')
plt.xlabel('Date')
plt.ylabel('Signal')
plt.legend()

plt.show()


# In[15]:


# Plot the actual price and the signals (buy/sell)
plt.figure(figsize=(10, 6))

# Plotting the actual close price
plt.plot(df.index, df['Close'], label='Close Price', color='black', alpha=0.7)

# Mark Buy signals (1) and Sell signals (-1) on the price chart
buy_signals = df[df['Signal'] == 1]
sell_signals = df[df['Signal'] == -1]

plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', alpha=1)
plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', alpha=1)

plt.title('Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.show()


# In[16]:


from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and best model
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
print("Optimized Accuracy:", accuracy_score(y_test, y_pred))


# In[ ]:


#Deployment


# In[21]:


#get_ipython().system('pip install streamlit')


# In[18]:


import streamlit as st
import yfinance as yf

# Streamlit app
st.title('Smart Investment Advisor')

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.date_input("End Date", value=pd.to_datetime('2024-12-31'))

if st.button('Get Predictions'):
    df = yf.download(ticker, start=start_date, end=end_date)
    # Add your model prediction code here
    st.write(df)
    # Display signals and plots


# In[ ]:





# In[ ]:




