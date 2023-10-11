# Here is a revamped version of the original code that uses Random Forest instead of Linear Regression.
# I've also included feature importance visualization for Random Forest.
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Filter out FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Function to calculate Gini Impurity
def gini_impurity(labels):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    proportions = label_counts / np.sum(label_counts)
    gini = 1 - np.sum(np.square(proportions))
    return gini


# Data Loading and Preprocessing


# Load data from the specified directory
input_dir = "C:/Users/Karlo/PycharmProjects/ML_project/input"

# Load data from 2015, 2016, 2017, 2018 and 2019 CSV files
happiness_2015 = pd.read_csv(f"{input_dir}/2015.csv")
happiness_2016 = pd.read_csv(f"{input_dir}/2016.csv")
happiness_2017 = pd.read_csv(f"{input_dir}/2017.csv")
happiness_2018 = pd.read_csv(f"{input_dir}/2018.csv")
happiness_2019 = pd.read_csv(f"{input_dir}/2019.csv")

# years 2015 - 2017 have more overlapping columns than 2015 - 2019

datasheets = [happiness_2015, happiness_2016, happiness_2017, happiness_2018, happiness_2019]

#  Rename columns for consistency
happiness_2015.columns = ['Country', 'Region', 'Happiness_Rank', 'Score',
                          'Standard Error', 'Economy', 'Family',
                          'Health', 'Freedom', 'Trust',
                          'Generosity', 'Dystopia_Residual']

happiness_2016.columns = ['Country', 'region', 'Happiness_Rank', 'Score', 'Lower Confidence Interval'
    , 'Upper Confidence Interval', 'Economy', 'Family',
                          'Health', 'Freedom', 'Trust',
                          'Generosity', 'Dystopia_Residual']

happiness_2017.columns = ['Country', 'Happiness_Rank', 'Score', 'Whisker.high', 'Whisker.low', 'Economy', 'Family',
                          'Health', 'Freedom', 'Trust',
                          'Generosity', 'Dystopia_Residual']

happiness_2018.columns = ['Overall rank', 'Country or region', 'Score', 'Economy', 'Social support',
                          'Health', 'Freedom', 'Generosity',
                          'Perceptions of corruption']

happiness_2019.columns = ['Overall rank', 'Country or region', 'Score', 'Economy', 'Social support',
                          'Health', 'Freedom', 'Generosity',
                          'Perceptions of corruption']

happiness_overlap = ['Score', 'Economy', 'Health', 'Freedom', 'Generosity']

for datasheet in datasheets:
    for column in datasheet.columns.tolist():
        if column not in happiness_overlap:
            datasheet.drop(column, axis=1, inplace=True)  # Add inplace=True to modify the DataFrame in place

frames = [happiness_2015, happiness_2016, happiness_2017, happiness_2018, happiness_2019]

happiness = pd.concat(frames)

# Separate the target variable (Score) and the features
X = happiness.drop("Score", axis=1)
y = happiness["Score"]

# Split the data into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Make predictions using the trained model on the testing set
y_pred_test = rf.predict(X_test)
y_pred_val = rf.predict(X_val)

# Gini Impurity Calculations
gini_test = gini_impurity(y_test)
gini_pred_test = gini_impurity(y_pred_test)
gini_val = gini_impurity(y_val)
gini_pred_val = gini_impurity(y_pred_val)

# Print Gini Impurity
print(f"Gini Impurity on Testing Set (Actual vs Predicted): {gini_test} vs {gini_pred_test}.")
print(f"Gini Impurity on Validation Set (Actual vs Predicted): {gini_val} vs {gini_pred_val}.")

