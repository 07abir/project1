import os
import sys
# ============================================================
#   PlantVeda_User_Routes_Mk2.py
#   Collects user input, calls all 5 ML modules,
#   sends their answers to PlantVeda_Voting for final result
# ============================================================

import PlantVeda_SVM_Mk2
import PlantVeda_NB_Mk2
import PlantVeda_KNN_Mk2
import PlantVeda_MLP_Mk2
import PlantVeda_LR_Mk2
import PlantVeda_Voting_Mk2

# ------------------ STEP 1: COLLECT USER INPUT ------------------
print("=" * 50)
print("       Welcome to PlantVeda - Plant Advisor")
print("=" * 50)

print("\nAvailable Soil Types:")
print("  Loamy / Sandy / Sandy soil / Moist loam / Wet soil")
soil_input = input("\nEnter Soil Type: ").strip()

print("\nAvailable Sunlight Types:")
print("  Full sun / Partial sun / Partial")
sun_input = input("\nEnter Sunlight Type: ").strip()

print("\nEnter Temperature Range (e.g. 20-35):")
temp_input = input("Temperature: ").strip()

# Pack into dictionary
user_input = {
    "Soil": soil_input,
    "Sunlight": sun_input,
    "Temperature": temp_input
}

# ------------------ STEP 2: CALL ALL 5 MODULES ------------------
print("\n" + "=" * 50)
print("   Consulting all 5 models...")
print("=" * 50)

knn_result = PlantVeda_KNN_Mk2.predict(user_input)
lr_result  = PlantVeda_LR_Mk2.predict(user_input)
mlp_result = PlantVeda_MLP_Mk2.predict(user_input)
svm_result = PlantVeda_SVM_Mk2.predict(user_input)
nb_result  = PlantVeda_NB_Mk2.predict(user_input)

print(f"\n  KNN          predicted : {knn_result}")
print(f"  Lin.Reg      predicted : {lr_result}")
print(f"  Neural Net   predicted : {mlp_result}")
print(f"  SVM          predicted : {svm_result}")
print(f"  Naive Bayes  predicted : {nb_result}")

# ------------------ STEP 3: SEND TO VOTING ------------------
all_predictions = {
    "KNN"        : knn_result,
    "LR"         : lr_result,
    "MLP"        : mlp_result,
    "SVM"        : svm_result,
    "NaiveBayes" : nb_result
}

final_answer = PlantVeda_Voting_Mk2.vote(all_predictions)

# ------------------ STEP 4: DISPLAY FINAL RESULT ------------------
print("\n" + "=" * 50)
print("         PlantVeda Final Recommendation")
print("=" * 50)
print(f"\n  Recommended Plant Growth Type : {final_answer}")
print("\n" + "=" * 50)