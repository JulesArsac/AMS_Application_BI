import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Function to prepare the data (separate X and y for train/test)
"""
    Generates the preference table for the problem.

    Args:
        attributes: The attributes on which the data is valued.
        min_max: Array containing either 1 for a maximization criterion or -1 for a minimization criterion.
        weights: Weight of each attribute.

    Returns:
        pref_table: The preference table of the given values.
"""
def prepare_data(train, test, target_column):
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]
    return X_train, y_train, X_test, y_test

# Define parameters and columns to process
# (sqrt(n))
n_neighbors = 230

# Target column
target_column = "Discount Availed"

# Columns to drop
columns_to_drop = ["Purchase Date", "Purchase Method", "Location"]

# Load and clean the data
data = pd.read_csv("dataset/E-commerce-data-cleaned.csv")
data = data.drop(columns=columns_to_drop)

# Split the data into numerical and categorical columns
numerical_cols = []
categorical_cols = ["Married", "Age Group", "Product Category"]

only_numerical = False
only_categorical = False

if not(numerical_cols == []):
    # Standardize the numerical columns
    scaler = StandardScaler()
    numerical_data = pd.DataFrame(scaler.fit_transform(data[numerical_cols]), columns=numerical_cols)
else:
    numerical_data = []
    only_categorical = True

if not(categorical_cols == []):
    # Encode the categorical columns
    encoder = OneHotEncoder()
    categorical_data = pd.DataFrame(
        encoder.fit_transform(data[categorical_cols]).toarray(),
        columns=encoder.get_feature_names_out(categorical_cols)
    )
else:
    categorical_data = []
    only_categorical = True

if only_numerical:
    X = numerical_data
elif only_categorical:
    X = categorical_data
else:
    # Concatenate the transformed columns
    X = pd.concat([numerical_data, categorical_data], axis=1)

X[target_column] = data[target_column]

# Split the data into training and testing sets
train, test = train_test_split(X, test_size=0.2, random_state=1)
X_train, y_train, X_test, y_test = prepare_data(train, test, target_column)

# Initialize the models
models = {
    "Random Forest": RandomForestClassifier(random_state=69),
    "SVC": SVC(random_state=69),
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=n_neighbors),
    "Dummy": DummyClassifier(strategy='most_frequent')
}

# Train and evaluate the models
print(f"Predicting '{target_column}' from the transformed data")
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{model_name} Accuracy: {score:.2f}")