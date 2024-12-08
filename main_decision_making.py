from Electre import electre_s,electre_v
from promethee import promethee_I,promethee_II
import pandas as pd

#Prix, vitesse_max, conso_moyenne, distance_frein, confort, volume coffre, accèlération
weights_per_profile = {
    "student" : [0.5,0.05,0.2,0.075,0.05,0.05,0.075],
    "family" : [0.25,0.05,0.2,0.1,0.2,0.1,0.1]
}

min_max = [-1,1,-1,-1,1,1,1]
veto = [3000,10,3.5,5,4,90,3]
thresholds = [2000,3,2,3,2,20,2]
cars = ["Alfa_156","Audi_A4","Cit_Xantia","Peugeot_406","Saab_TID","Rnlt_Laguna","VW_Passat","BMW_320d","Cit_Xara","Rnlt_Safrane"]

for profile in weights_per_profile.keys():
    print(f"====================== {profile} ======================")
    weights = weights_per_profile[profile]
    attributes = pd.read_csv("data/donnees.csv", header=None).values
    promethee_I(attributes,min_max,weights,cars,f"graphs/prometheeI_{profile}")
    promethee_II(attributes,min_max,weights,cars,f"graphs/prometheeII_{profile}")
    promethee_I(attributes,min_max,weights,cars,f"graphs/prometheeIs_{profile}",thresholds)
    promethee_II(attributes,min_max,weights,cars,f"graphs/prometheeIIs_{profile}",thresholds)
    electre_v(attributes,min_max,weights,veto,cars,f"graphs/ElectreIv_{profile}")
    electre_s(attributes,min_max,weights,veto,thresholds,cars,f"graphs/ElectreIs_{profile}")

