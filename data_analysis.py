import pandas as pd
import matplotlib.pyplot as plt
import math
import mplfinance as mpf

# Load the data
data = pd.read_csv("dataset/E-commerce-data.csv")




# amount_married = 0
# amount_not_married = 0
# for i in range(len(data)):
#     net_amount = data["Net Amount"][i]
#     if math.isnan(net_amount):
#         continue
#     if data["Married"][i] == "yes":
#         amount_married += net_amount
#     else:
#         amount_not_married += net_amount
#
# amount_married = amount_married / 1000000
# amount_not_married = amount_not_married / 1000000
# amount_married = round(amount_married, 2)
# amount_not_married = round(amount_not_married, 2)
# plt.figure(figsize=(10, 6))
# plt.bar(['Married', 'Not Married'], [amount_married, amount_not_married])
# plt.title('Net Amount by Marital Status (in million)')
# plt.ylabel('Net Amount')
# plt.savefig('output/net_amount_by_marital_status.png')


dico_gender = {}
for i in range(len(data)):
    value = data["Location"][i]
    if str(value) == 'nan':
        continue
    if not dico_gender.get(value):
        dico_gender[value] = 1
    else:
        dico_gender[value] += 1


plt.figure(figsize=(10, 6))
plt.pie(list(dico_gender.values()), labels=list(dico_gender.keys()), autopct='%1.1f%%')
plt.title('Distribution by Location')
plt.savefig('output/distribution_by_location.png')


