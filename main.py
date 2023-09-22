import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split  # Import train_test_split
import matplotlib.pyplot as plt

# todo overall: visualise data.
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


# Separate the target variable (Happiness_Score) and the features
X = happiness.drop("Score", axis=1)
y = happiness["Score"]


# Create a 1x5 grid for 5 subplots
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

# Plot the scatter plots
axes[0].scatter(X.Economy, y)
axes[0].set_title("Economy")

axes[1].scatter(X.Health, y)
axes[1].set_title("Health")

axes[2].scatter(X.Freedom, y)
axes[2].set_title("Freedom")

axes[3].scatter(X.Generosity, y)
axes[3].set_title("Generosity")

# Show the plots
plt.show()

# Split the data into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create a Linear Regression model
lm = LinearRegression()

# Fit the model to your training data
lm.fit(X_train, y_train)

# Make predictions using the trained model on the testing set
y_pred_test = lm.predict(X_test)

# Calculate Mean Squared Error (MSE) on the testing set to evaluate model performance
mse_test = mean_squared_error(y_test, y_pred_test)

# Make predictions using the trained model on the validation set
y_pred_val = lm.predict(X_val)

# Calculate Mean Squared Error (MSE) on the validation set to evaluate model performance
mse_val = mean_squared_error(y_val, y_pred_val)

# Print the MSE values for testing and validation
print("Mean Squared Error (MSE) on Testing Set:", mse_test)
print("Mean Squared Error (MSE) on Validation Set:", mse_val)
