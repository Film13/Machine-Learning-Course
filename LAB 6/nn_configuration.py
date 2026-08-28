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
    "output_nn_configuration.txt"
)

OUTPUT_PNG = os.path.join(
    BASE_DIR,
    "nn_configuration.png"
)
if not os.path.exists(CSV_PATH):

    print("ERROR: cats(1).csv was not found.")

    print("Expected location:")
    print(CSV_PATH)

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
configurations = {

    "1 Hidden Layer - 8 Neurons":
        (8,),

    "1 Hidden Layer - 16 Neurons":
        (16,),

    "2 Hidden Layers - 16, 8 Neurons":
        (16, 8),

    "2 Hidden Layers - 32, 16 Neurons":
        (32, 16),

    "3 Hidden Layers - 32, 16, 8 Neurons":
        (32, 16, 8)
}
results = []

output = []

output.append("========================================")
output.append("       NEURAL NETWORK LAB 6")
output.append("       NN CONFIGURATION")
output.append("========================================")

output.append("")

output.append(
    f"Dataset: cats(1).csv"
)

output.append(
    f"Dataset Shape: {df.shape}"
)

output.append(
    f"Training Samples: {len(X_train)}"
)

output.append(
    f"Testing Samples: {len(X_test)}"
)

output.append("")

output.append(
    "Epochs used: 100"
)

output.append("")

output.append("========================================")
output.append("       CONFIGURATION RESULTS")
output.append("========================================")
for name, hidden_layers in configurations.items():

    model = MLPClassifier(

        hidden_layer_sizes=hidden_layers,

        activation="relu",

        solver="adam",

        max_iter=100,

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
    results.append({

        "Configuration": name,

        "Hidden Layers": len(hidden_layers),

        "Neurons": str(hidden_layers),

        "Accuracy": accuracy,

        "Accuracy (%)": accuracy * 100
    })
    result_text = (

        f"{name} | "

        f"Accuracy: {accuracy:.4f} "

        f"({accuracy * 100:.2f}%)"

    )


    print(result_text)

    output.append(result_text)

results_df = pd.DataFrame(
    results
)
output.append("")
output.append("========================================")
output.append("           RESULTS TABLE")
output.append("========================================")

output.append("")

output.append(
    results_df.to_string(index=False)
)
best_index = results_df[
    "Accuracy"
].idxmax()


best_config = results_df.loc[
    best_index
]


output.append("")

output.append("========================================")
output.append("           BEST CONFIGURATION")
output.append("========================================")

output.append("")

output.append(
    f"Configuration: "
    f"{best_config['Configuration']}"
)

output.append(
    f"Hidden Layers: "
    f"{best_config['Hidden Layers']}"
)

output.append(
    f"Neurons: "
    f"{best_config['Neurons']}"
)

output.append(
    f"Accuracy: "
    f"{best_config['Accuracy (%)']:.2f}%"
)
with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(output)
    )
plt.figure(
    figsize=(10, 6)
)


plt.bar(

    results_df["Configuration"],

    results_df["Accuracy (%)"]

)


plt.xlabel(
    "Neural Network Configuration"
)


plt.ylabel(
    "Accuracy (%)"
)


plt.title(
    "Comparison of Neural Network Configurations"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.grid(
    axis="y"
)


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

print("Best Configuration:")
print(
    best_config["Configuration"]
)

print(
    f"Best Accuracy: "
    f"{best_config['Accuracy (%)']:.2f}%"
)

print("")

print("Done!")