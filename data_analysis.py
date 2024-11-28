import pandas as pd
import matplotlib.pyplot as plt
import math
import mplfinance as mpf

# Load the data
data = pd.read_csv("dataset/E-commerce-data.csv")




count_married = 0
count_not_married = 0
# amount_married = 0
# amount_not_married = 0
already_seen_married = []
already_seen_not_married = []
for i in range(len(data)):
    net_amount = data["Net Amount"][i]
    customer_id = data["CID"][i]
    if math.isnan(net_amount):
        continue
    if data["Married"][i] == "yes":
        # amount_married += net_amount
        count_married += 1
        if customer_id not in already_seen_married:
            already_seen_married.append(customer_id)
    else:
        # amount_not_married += net_amount
        count_not_married += 1
        if customer_id not in already_seen_not_married:
            already_seen_not_married.append(customer_id)

# amount_married = count_married / count_married
# amount_not_married = amount_not_married / count_not_married
# amount_married = round(amount_married, 2)
# amount_not_married = round(amount_not_married, 2)

nb_transaction_per_person_married = count_married / len(already_seen_married)
nb_transaction_per_person_not_married = count_not_married / len(already_seen_not_married)

plt.figure(figsize=(10, 6))
plt.bar(['Married', 'Not Married'], [nb_transaction_per_person_married, nb_transaction_per_person_not_married])
plt.title('Amount of transactions by Person and Marital Status')
plt.ylabel('Net Amount')
plt.savefig('output/amount_of_transaction_by_person_by_marital_status.png')


# dico_gender = {}
# for i in range(len(data)):
#     value = data["Location"][i]
#     if str(value) == 'nan':
#         continue
#     if not dico_gender.get(value):
#         dico_gender[value] = 1
#     else:
#         dico_gender[value] += 1
#
#
# plt.figure(figsize=(10, 6))
# plt.pie(list(dico_gender.values()), labels=list(dico_gender.keys()), autopct='%1.1f%%')
# plt.title('Distribution by Location')
# plt.savefig('output/distribution_by_location.png')


