import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Function to prepare the data (separate X and y for train/test)
def prepare_data(train, test, target_column):
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]
    return X_train, y_train, X_test, y_test

# Default number of neighbors for KNN
DEFAULT_N_NEIGHBORS = 234

# Load the data
data = pd.read_csv("dataset/E-commerce-data-cleaned.csv")

# Ask the user for inputs
target_column = input("Entrez la colonne cible à prédire : ")

if target_column not in data.columns:
    raise ValueError(f"La colonne cible '{target_column}' n'existe pas dans le dataset.")

columns_to_use = input("Entrez les colonnes à utiliser pour la prédiction (séparées par des virgules) : ")
columns_to_use = [col.strip() for col in columns_to_use.split(",")]

for col in columns_to_use:
    if col not in data.columns:
        raise ValueError(f"La colonne '{col}' n'existe pas dans le dataset.")

try:
    n_neighbors = int(input(f"Entrez le nombre de voisins pour KNN (par défaut n = sqrt(nombre de lignes de data) = {DEFAULT_N_NEIGHBORS}): "))
except ValueError:
    n_neighbors = DEFAULT_N_NEIGHBORS

# Split the data into numerical and categorical columns
numerical_cols = [col for col in columns_to_use if data[col].dtype in ['int64', 'float64']]
categorical_cols = [col for col in columns_to_use if col not in numerical_cols]

# Process numerical columns
if numerical_cols:
    scaler = StandardScaler()
    numerical_data = pd.DataFrame(scaler.fit_transform(data[numerical_cols]), columns=numerical_cols)
else:
    numerical_data = pd.DataFrame()

# Process categorical columns
if categorical_cols:
    encoder = OneHotEncoder()
    categorical_data = pd.DataFrame(
        encoder.fit_transform(data[categorical_cols]).toarray(),
        columns=encoder.get_feature_names_out(categorical_cols)
    )
else:
    categorical_data = pd.DataFrame()

# Concatenate numerical and categorical data
X = pd.concat([numerical_data, categorical_data], axis=1)

# Add the target column
X[target_column] = data[target_column]

# Split the data into training and testing sets
train, test = train_test_split(X, test_size=0.2, random_state=1)
X_train, y_train, X_test, y_test = prepare_data(train, test, target_column)

# Initialize the models
models = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Random Forest": RandomForestClassifier(random_state=1),
    "SVC": SVC(random_state=1),
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=n_neighbors),
}

# Display correlation matrix for the training set
correlation = train.corr()
if target_column in correlation:
    print("\nMatrice de corrélation avec la colonne cible :")
    print(correlation[target_column])

# Train and evaluate the models
print(f"\nPrédiction pour '{target_column}' à partir des colonnes sélectionnées.")
for model_name, model in models.items():
    print(f"\nEntraînement du modèle {model_name}...")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{model_name} - Précision : {score:.2f}")