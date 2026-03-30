import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ------------------ STEP 1: DATA ------------------
soil = [
"Loamy","Loamy","Loamy","Loamy","Sandy","Sandy soil","Moist loam","Sandy",
"Sandy","Loamy","Loamy","Moist loam","Loamy","Loamy","Sandy","Loamy","Sandy",
"Wet soil","Sandy","Moist loam","Moist loam","Moist loam","Sandy","Sandy",
"Loamy","Loamy","Loamy","Sandy","Loamy","Loamy","Loamy","Moist loam","Sandy",
"Sandy","Moist loam","Moist loam","Sandy","Moist loam"
]
sunlight = [
"Partial sun","Full sun","Full sun","Full sun","Full sun","Full sun","Full sun",
"Full sun","Full sun","Partial","Full sun","Partial","Full sun","Full sun",
"Full sun","Full sun","Partial","Full sun","Full sun","Full sun","Full sun",
"Full sun","Full sun","Full sun","Full sun","Full sun","Full sun","Full sun",
"Full sun","Full sun","Full sun","Partial","Full sun","Partial","Partial",
"Partial","Full sun","Full sun"
]
temperature = [
"20-35","20-40","20-35","20-35","20-35","20-40","20-35","25-35","25-40","18-35",
"20-35","20-35","20-40","24-35","24-35","20-40","22-35","20-35","20-35","20-35",
"18-30","18-35","20-35","20-38","20-35","25-35","22-35","20-35","22-35","20-35",
"20-35","18-30","25-40","20-35","18-30","24-35","20-35","20-35"
]
growth = [
"Tree","Tree","Tree","Tree","Tree","Tree","Tree","Palm","Palm","Grass","Tree",
"Tree","Tree","Tree","Tree","Tree","Tree","Tree","Tree","Tree","Aquatic herb",
"Aquatic herb","Herb","Vine","Shrub","Herb","Tree","Herb","Tree","Tree","Tree",
"Palm","Palm","Shrub","Shrub","Tree","Tree","Tree"
]

# Create DataFrame
df = pd.DataFrame({
    "Soil": soil,
    "Sunlight": sunlight,
    "Temperature": temperature,
    "Growth": growth
})

# ------------------ STEP 2: TEMPERATURE → NUMERIC ------------------
df["Temperature"] = df["Temperature"].apply(
    lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2
)

# ------------------ STEP 3: ENCODE FEATURES ------------------
le_soil = LabelEncoder()
le_sun = LabelEncoder()
df["Soil"] = le_soil.fit_transform(df["Soil"])
df["Sunlight"] = le_sun.fit_transform(df["Sunlight"])

# ------------------ STEP 4: NORMALIZATION ------------------
scaler = MinMaxScaler()
df[["Soil","Sunlight","Temperature"]] = scaler.fit_transform(
    df[["Soil","Sunlight","Temperature"]]
)

# ------------------ STEP 5: TARGET ENCODING (g4) ------------------
le_target = LabelEncoder()
g4 = le_target.fit_transform(df["Growth"])
print("Encoded Target (g4):")
print(g4)

# ------------------ STEP 6: SVM MODEL ------------------
X = df[["Soil","Sunlight","Temperature"]]
y = g4

svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X, y)

# ------------------ STEP 7: PREDICTION ------------------
pred = svm.predict(X)
print("\nPredictions:")
print(pred[:10])

# ------------------ STEP 8: ACCURACY ------------------
accuracy = accuracy_score(y, pred)
print("\nAccuracy:", accuracy)