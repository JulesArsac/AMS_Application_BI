import math

import pandas as pd

data = pd.read_csv("dataset/E-commerce-data.csv")

# Drop rows by name
data = data.drop(columns=["Discount Name", "Discount Amount (INR)", "CID", "TID"])

# Update the "Discount Availed" column
data["Discount Availed"] = data["Discount Availed"].map({
    "Yes": 1,
    "No": 0
})


# Update the "Married" column
data["Married"] = data["Married"].map({
    "yes": 1,
    "no": 0
})

data["Gender"] = data["Gender"].map({
    "Male": 0,
    "Female": 1,
    "Other": 2
})

# Update the "Age Group" column
dico_age_group = {"under 18": 0, "18-25": 1, "25-45": 2, "45-60": 3, "60 and above": 4}
data["Age Group"] = data["Age Group"].map(dico_age_group)

# Update the "Gross Amount and Net Amount" columns
print("Updating Gross Amount and Net Amount columns")
for i in range(len(data)):
    if i % 1000 == 0:
        print(f"Progress: {i}/{len(data)}")
    gross_amount = data.loc[i, "Gross Amount"]
    net_amount = data.loc[i, "Net Amount"]

    if not math.isnan(gross_amount):
        gross_amount = round(gross_amount, 2)
        data.loc[i, "Gross Amount"] = gross_amount

    if not math.isnan(net_amount):
        net_amount = round(net_amount, 2)
        data.loc[i, "Net Amount"] = net_amount
print("Gross Amount and Net Amount columns updated")

# Save the cleaned data
data.to_csv("dataset/E-commerce-data-cleaned.csv", index=False)


data = pd.read_csv("dataset/E-commerce-data-cleaned.csv")

nb_rows = len(data)

# Remove row with missing values
data = data.dropna()

nb_rows_after = len(data)

print(f"Number of rows before removing rows with missing values: {nb_rows}")
print(f"Number of rows after removing rows with missing values: {nb_rows_after}")

# Drop rows where net amount is negative
data = data[data["Net Amount"] >= 0]

print(f"Number of rows after removing rows with negative net amount: {len(data)}")

data.to_csv("dataset/E-commerce-data-cleaned.csv", index=False)