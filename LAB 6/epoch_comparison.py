import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR,
    "cats(1).csv"
)

OUTPUT_TXT = os.path.join(
    BASE_DIR,
    "output_epoch_comparison.txt"
)

OUTPUT_PNG = os.path.join(
    BASE_DIR,
    "epoch_comparison.png"
)

if not os.path.exists(CSV_PATH):
    print("ERROR: cats(1).csv was not found.")
    sys.exit()

df = pd.read_csv(
    CSV_PATH,
    header=None
)

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

target = "Sex"

X = df[features].copy()
y = df[target].copy()


# Convert categorical features
X = pd.get_dummies(
    X,
    columns=features
)


# Encode target
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
epoch_values = [
    10,
    25,
    50,
    100,
    200
]
accuracies = []
# เก็บ Output ทั้งหมดไว้ใน list
output = []

output.append("========================================")
output.append("       NEURAL NETWORK LAB 6")
output.append("       EPOCH COMPARISON")
output.append("========================================")

output.append("")
output.append(f"Dataset: cats(1).csv")
output.append(f"Dataset Shape: {df.shape}")
output.append(f"Training Samples: {len(X_train)}")
output.append(f"Testing Samples: {len(X_test)}")

output.append("")
output.append("Neural Network Configuration:")
output.append("Hidden Layers: 2")
output.append("Neurons: 16, 8")
output.append("Activation: ReLU")
output.append("Solver: Adam")

output.append("")
output.append("========================================")
output.append("       EPOCH COMPARISON RESULTS")
output.append("========================================")


for epochs in epoch_values:

    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=epochs,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    accuracies.append(accuracy)

    result = (
        f"Epochs: {epochs:3d} | "
        f"Accuracy: {accuracy:.4f} | "
        f"{accuracy * 100:.2f}%"
    )

    output.append(result)

    print(result)
results = pd.DataFrame({
    "Epochs": epoch_values,
    "Accuracy": accuracies
})

output.append("")
output.append("========================================")
output.append("           RESULTS TABLE")
output.append("========================================")

output.append(
    results.to_string(index=False)
)
best_index = accuracies.index(
    max(accuracies)
)

best_epoch = epoch_values[best_index]

best_accuracy = accuracies[best_index]


output.append("")
output.append("========================================")
output.append("           BEST RESULT")
output.append("========================================")

output.append(
    f"Best Epochs: {best_epoch}"
)

output.append(
    f"Best Accuracy: {best_accuracy * 100:.2f}%"
)
with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(output)
    )
plt.figure(figsize=(8, 5))

plt.plot(
    epoch_values,
    accuracies,
    marker="o"
)

plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")

plt.title(
    "Neural Network Accuracy vs Number of Epochs"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300
)

plt.show()
print("")
print("========================================")
print("OUTPUT FILES CREATED")
print("========================================")

print("Text Output:")
print(OUTPUT_TXT)

print("")
print("Graph:")
print(OUTPUT_PNG)

print("")
print("Done!")