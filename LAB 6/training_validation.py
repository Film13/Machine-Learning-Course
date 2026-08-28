import pandas as pd
import matplotlib.pyplot as plt

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

features = ["Age", "Size", "Coat", "Breed"]

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
    max_iter=1,
    warm_start=True,
    random_state=42
)

epochs = 100

train_accuracy = []
validation_accuracy = []
train_loss = []


for epoch in range(epochs):

    model.fit(X_train, y_train)

    # Training prediction
    train_pred = model.predict(X_train)

    # Validation/Test prediction
    test_pred = model.predict(X_test)

    # Accuracy
    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()

    train_accuracy.append(train_acc)
    validation_accuracy.append(test_acc)

    # Loss
    train_loss.append(model.loss_)

    print(
        f"Epoch {epoch + 1:3d} | "
        f"Train Accuracy: {train_acc:.4f} | "
        f"Validation Accuracy: {test_acc:.4f} | "
        f"Loss: {model.loss_:.4f}"
    )

plt.figure(figsize=(8, 5))

plt.plot(
    train_accuracy,
    label="Training Accuracy"
)

plt.plot(
    validation_accuracy,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    train_loss,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training Loss"
)

plt.legend()
plt.grid(True)

plt.show()
print("\n==============================")
print("Final Result")
print("==============================")

print(
    f"Training Accuracy: "
    f"{train_accuracy[-1] * 100:.2f}%"
)

print(
    f"Validation Accuracy: "
    f"{validation_accuracy[-1] * 100:.2f}%"
)

print(
    f"Final Loss: "
    f"{train_loss[-1]:.4f}"
)