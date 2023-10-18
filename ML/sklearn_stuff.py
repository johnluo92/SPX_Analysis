import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data into a Pandas dataframe
df = pd.read_csv('SPX_Data.csv')

# Extract the features and target variables
X = df.drop(['Adj Close**'], axis=1)
y = df['Adj Close**']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = model.score(X_test, y_test)

# Print the model's accuracy
print("Accuracy:", accuracy)
