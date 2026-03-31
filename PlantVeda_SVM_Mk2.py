import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC

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

def predict(input_data):
    df = pd.DataFrame({
        "Soil": soil, "Sunlight": sunlight,
        "Temperature": temperature, "Growth": growth
    })

    df["Temperature"] = df["Temperature"].apply(
        lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2
    )

    le_soil = LabelEncoder()
    le_sun = LabelEncoder()
    df["Soil"] = le_soil.fit_transform(df["Soil"])
    df["Sunlight"] = le_sun.fit_transform(df["Sunlight"])

    scaler = MinMaxScaler()
    df[["Soil","Sunlight","Temperature"]] = scaler.fit_transform(
        df[["Soil","Sunlight","Temperature"]]
    )

    le_target = LabelEncoder()
    df["Growth_enc"] = le_target.fit_transform(df["Growth"])

    X = df[["Soil","Sunlight","Temperature"]]
    y = df["Growth_enc"]

    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X, y)

    # ---------- transform user input ----------
    input_soil = le_soil.transform([input_data["Soil"]])[0]
    input_sun  = le_sun.transform([input_data["Sunlight"]])[0]
    temp_str   = input_data["Temperature"]
    input_temp = (int(temp_str.split('-')[0]) + int(temp_str.split('-')[1])) / 2

    user_row = pd.DataFrame([[input_soil, input_sun, input_temp]],
                         columns=["Soil","Sunlight","Temperature"])
    user_row = scaler.transform(user_row)
    user_row = pd.DataFrame(user_row,
                         columns=["Soil","Sunlight","Temperature"])

   

    pred_enc = svm.predict(user_row)[0]
    return le_target.inverse_transform([pred_enc])[0]