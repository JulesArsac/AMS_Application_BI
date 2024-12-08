import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = pd.read_csv("dataset/E-commerce-data-cleaned.csv")

data["Purchase Date"] = pd.to_datetime(data["Purchase Date"], format="%d/%m/%Y %H:%M:%S")
data["Purchase Date"] = data["Purchase Date"].dt.dayofweek
# Split the data into numerical and categorical columns
numerical_cols = ["Gross Amount", "Net Amount"]
categorical_cols = ["Gender", "Age Group","Purchase Date", "Product Category", "Discount Availed", "Purchase Method", "Location", "Married"]

scaler = StandardScaler()
numerical_data = pd.DataFrame(scaler.fit_transform(data[numerical_cols]), columns=numerical_cols)

encoder = OneHotEncoder()
categorical_data = pd.DataFrame(
    encoder.fit_transform(data[categorical_cols]).toarray(),
    columns=encoder.get_feature_names_out(categorical_cols)
)

X = pd.concat([numerical_data, categorical_data], axis=1)

for target in X.columns:
    correlation = X.corr(method="spearman")
    target_correlation = correlation[target]
    threshold = 0.2
    print(target_correlation[(target_correlation > threshold) | (target_correlation < -threshold) & (target_correlation < 0.98)])




