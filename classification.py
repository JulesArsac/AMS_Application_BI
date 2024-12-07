import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def prepare_data(train, test, target_column):
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]
    return X_train, y_train, X_test, y_test

n_neighbors = 232 # sqrt of n

# Chargement et préparation des données
data = pd.read_csv("dataset/E-commerce-data-cleaned.csv")
target_column = "Married"
columns_to_drop = ["Purchase Date", "Product Category", "Purchase Method", "Location"]

data = data.drop(columns=columns_to_drop)
target = data[target_column]
data = data.drop(columns=[target_column])

only_num = data.drop(columns=["Gender", "Age Group", "Discount Availed"])
scaler = StandardScaler()
only_num_scaled = scaler.fit_transform(only_num)

only_cat = data.drop(columns=["Gross Amount", "Net Amount"])
encoder = OneHotEncoder()
encoder.fit(only_cat)
only_cat_encoded = encoder.transform(only_cat).toarray()

X = pd.concat([pd.DataFrame(only_num_scaled), pd.DataFrame(only_cat_encoded), pd.DataFrame(target)], axis=1)

train, test = train_test_split(X, test_size=0.2, random_state=69)
X_train, y_train, X_test, y_test = prepare_data(train, test, target_column)

# Initialisation des modèles
rdforest = RandomForestClassifier(random_state=69)
svc = SVC(random_state=69)
gmb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# Entraînement des modèles
print("Training Random Forest")
rdforest.fit(X_train, y_train)
print("Training SVC")
svc.fit(X_train, y_train)
print("Training Gaussian Naive Bayes")
gmb.fit(X_train, y_train)
print("Training K-Nearest Neighbors")
knn.fit(X_train, y_train)

# Évaluation des modèles
print(f"Random Forest score: {rdforest.score(X_test, y_test)}")
print(f"SVC score: {svc.score(X_test, y_test)}")
print(f"Gaussian Naive Bayes score: {gmb.score(X_test, y_test)}")
print(f"KNN score: {knn.score(X_test, y_test)}")