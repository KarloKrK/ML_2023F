import warnings
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Data Loading and Preprocessing

# Filter out FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data from the specified directory
input_dir = "C:/Users/Karlo/PycharmProjects/ML_project/input"

# Load data from CSV files
happiness_2015 = pd.read_csv(f"{input_dir}/2015.csv")
happiness_2016 = pd.read_csv(f"{input_dir}/2016.csv")
happiness_2017 = pd.read_csv(f"{input_dir}/2017.csv")
happiness_2018 = pd.read_csv(f"{input_dir}/2018.csv")
happiness_2019 = pd.read_csv(f"{input_dir}/2019.csv")

datasheets = [happiness_2015, happiness_2016, happiness_2017, happiness_2018, happiness_2019]

# Rename columns for consistency and filter columns
happiness_overlap = ['Score', 'Economy', 'Health', 'Freedom', 'Generosity']
for datasheet in datasheets:
    for column in datasheet.columns.tolist():
        if column not in happiness_overlap:
            datasheet.drop(column, axis=1, inplace=True)

# Concatenate all years
happiness = pd.concat(datasheets)

# Split the data into training, testing, and validation sets
X = happiness.drop("Score", axis=1)
y = happiness["Score"]
X.dropna(inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled_random = scaler.fit_transform(X)

# Perform KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled_random)

# Compute the inertia for 3 clusters
inertia_3_clusters = kmeans.inertia_

# Compute the silhouette score for 3 clusters
silhouette_score_3_clusters = silhouette_score(X_scaled_random, kmeans.labels_)

print(inertia_3_clusters, silhouette_score_3_clusters)