import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


# Function to prepare the data (separate X and y for train/test)

def prepare_data(train, test, target_column):
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]
    return X_train, y_train, X_test, y_test

def accuracy_score(lst_classif, lst_classif_names, X, y):
    for clf, name_clf in zip(lst_classif, lst_classif_names):
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        scores = cross_val_score(clf, X, y, cv=skf)
        print("Accuracy of " + name_clf + " classifier on cross-validation: %0.2f (+/- %0.2f)" % (
        scores.mean(), scores.std() * 2))

# Define parameters and columns to process
# (sqrt(n))
n_neighbors = 7



# Load and clean the data
data = pd.read_csv("dataset/E-commerce-data-cleaned.csv")

data["Purchase Date"] = pd.to_datetime(data["Purchase Date"], format="%d/%m/%Y %H:%M:%S")
data["Purchase Date"] = data["Purchase Date"].dt.dayofweek

target_column = "Discount Availed"

# Split the data into numerical and categorical columns
numerical_cols = ["Gross Amount", "Net Amount"]
categorical_cols = ["Gender", "Age Group", "Purchase Date"]

only_numerical = False
only_categorical = False

if not (numerical_cols == []):
    # Standardize the numerical columns
    scaler = StandardScaler()
    numerical_data = pd.DataFrame(scaler.fit_transform(data[numerical_cols]), columns=numerical_cols)
else:
    numerical_data = []
    only_categorical = True

if not (categorical_cols == []):
    # Encode the categorical columns
    encoder = OneHotEncoder()
    categorical_data = pd.DataFrame(
        encoder.fit_transform(data[categorical_cols]).toarray(),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # categorical_data = pd.get_dummies(data[categorical_cols])
else:
    categorical_data = []
    only_numerical = True

if only_numerical:
    X = numerical_data
elif only_categorical:
    X = categorical_data
else:
    # Concatenate the transformed columns
    X = pd.concat([numerical_data, categorical_data], axis=1)


X[target_column] = data[target_column]

# X[target_column] = data[target_column]
# mean = X[target_column].mean()
#
# # Utilisation de .loc pour éviter les erreurs de chaîne d'affectation
# X.loc[X[target_column] < mean, target_column] = 0
# X.loc[X[target_column] >= mean, target_column] = 1




# Split the data into training and testing sets
train, test = train_test_split(X, test_size=0.2, random_state=1)
X_train, y_train, X_test, y_test = prepare_data(train, test, target_column)


# Initialize the models
models = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Random Forest": RandomForestClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=n_neighbors),
    "SVC": SVC(),
}

# Train and evaluate the models
print(f"Predicting '{target_column}' from the transformed data")
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{model_name} Accuracy: {score:.2f}")

# names = []
# classifiers = []
# for name, model in models.items():
#     names.append(name)
#     classifiers.append(model)
#
# target = X[target_column]
# X = X.drop(columns=[target_column])
# accuracy_score(classifiers, names, X, target)


# correlation = train.corr()
# print(correlation[target_column])
