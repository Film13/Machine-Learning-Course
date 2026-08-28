import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
df = pd.read_csv("cats(1).csv", header=None)

df.columns = [
    "ID",
    "PetID",
    "URL",
    "Animal",
    "Age",
    "Sex",
    "Size",
    "Coat",
    "Breed",
    "Images",
    "ImageURLs"
]
features = [
    "Age",
    "Size",
    "Coat",
    "Breed"
]

X = df[features].copy()

y = df["Sex"].copy()

X = pd.get_dummies(
    X,
    columns=features
)
encoder = LabelEncoder()

y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model = MLPClassifier(

    hidden_layer_sizes=(16, 8),

    activation="relu",

    solver="adam",

    max_iter=300,

    random_state=42
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

actual = encoder.inverse_transform(
    y_test
)

predicted = encoder.inverse_transform(
    y_pred
)

result = pd.DataFrame({

    "Actual": actual,

    "Predicted": predicted

})


print("\nPrediction Results:")

print(
    result.head(30)
)