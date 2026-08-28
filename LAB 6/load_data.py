import pandas as pd

# Load dataset
df = pd.read_csv("cats(1).csv", header=None)

# Set column names
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

# Display dataset
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nSex distribution:")
print(df["Sex"].value_counts())

print("\nAge distribution:")
print(df["Age"].value_counts())

print("\nSize distribution:")
print(df["Size"].value_counts())

print("\nCoat distribution:")
print(df["Coat"].value_counts())

print("\nBreed distribution:")
print(df["Breed"].value_counts())