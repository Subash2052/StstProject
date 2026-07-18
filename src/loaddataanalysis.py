# Import required libraries
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("../data/Churn.csv")
# Clean column names
# Clean column names
df.columns = (
    df.columns
    .str.strip()          # Remove leading/trailing spaces
    .str.replace(r"\s+", " ", regex=True)  # Replace multiple spaces with one
)
print(df.columns.tolist())

# Display the first five rows
print(df.head())
# Display the dimensions of the dataset
print("\nDataset Shape:")
print(df.shape)

# Display the column names
print("\nColumn Names:")
print(df.columns)

# Display information about the dataset
print("\nDataset Information:")
print(df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())
# -----------------------------
# Check for Missing Values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Check for Duplicate Records
# -----------------------------
print("\nDuplicate Records:")
print(df.duplicated().sum())
# -----------------------------
# Remove Duplicate Records
# -----------------------------
df = df.drop_duplicates()

# Verify duplicates have been removed
print("\nDuplicate Records After Cleaning:")
print(df.duplicated().sum())

# Display the new dataset shape
print("\nDataset Shape After Removing Duplicates:")
print(df.shape)
# -----------------------------
# Save the cleaned dataset
# -----------------------------
df.to_csv("../data/Churn_Cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
# -----------------------------
# Histogram: Subscription Length
# -----------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["Subscription Length"],
         bins=20,
         edgecolor="black")

plt.title("Distribution of Subscription Length")
plt.xlabel("Subscription Length")
plt.ylabel("Number of Customers")

plt.savefig("../figures/subscription_length_histogram.png")

plt.close()

print("Subscription Length histogram saved successfully!")

# ==========================================================
# Histogram: Charge Amount
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(df["Charge Amount"],
         bins=20,
         edgecolor="black")

plt.title("Distribution of Charge Amount")
plt.xlabel("Charge Amount")
plt.ylabel("Number of Customers")

plt.savefig("../figures/charge_amount_histogram.png")

plt.close()

print("✓ Charge Amount histogram saved successfully!")


# ==========================================================
# Histogram: Frequency of Use
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(df["Frequency of use"],
         bins=20,
         edgecolor="black")

plt.title("Distribution of Frequency of Use")
plt.xlabel("Frequency of Use")
plt.ylabel("Number of Customers")

plt.savefig("../figures/frequency_of_use_histogram.png")

plt.close()

print("✓ Frequency of Use histogram saved successfully!")
# ==========================================================
# Boxplot: Customer Value by Churn
# ==========================================================

plt.figure(figsize=(7,5))

sns.boxplot(x="Churn", y="Customer Value", data=df)

plt.title("Customer Value by Churn")
plt.xlabel("Churn")
plt.ylabel("Customer Value")

plt.savefig("../figures/customer_value_boxplot.png")

plt.close()

print("✓ Customer Value boxplot saved successfully!")


# ==========================================================
# Histogram: Customer Value
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(df["Customer Value"],
         bins=20,
         edgecolor="black")

plt.title("Distribution of Customer Value")
plt.xlabel("Customer Value")
plt.ylabel("Number of Customers")

plt.savefig("../figures/customer_value_histogram.png")

plt.close()

print("✓ Customer Value histogram saved successfully!")

# ==========================================================
# Scatter Plot: Subscription Length vs Customer Value
# ==========================================================

plt.figure(figsize=(8,5))

plt.scatter(df["Subscription Length"],
            df["Customer Value"],
            alpha=0.6)

plt.title("Subscription Length vs Customer Value")
plt.xlabel("Subscription Length")
plt.ylabel("Customer Value")

plt.savefig("../figures/subscription_vs_customer_value.png")

plt.close()

print("✓ Scatter plot saved successfully!")
# ==========================================================
# Correlation Heatmap
# ==========================================================

plt.figure(figsize=(12,8))

correlation = df.corr(numeric_only=True)

sns.heatmap(correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f")

plt.title("Correlation Matrix")

plt.savefig("../figures/correlation_heatmap.png")

plt.close()

print("✓ Correlation heatmap saved successfully!")
# ==========================================================
# Churn Distribution
# ==========================================================

plt.figure(figsize=(6,4))

sns.countplot(x="Churn", data=df)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.savefig("../figures/churn_distribution.png")

plt.close()

print("✓ Churn distribution chart saved successfully!")
# ==========================================================
# Correlation with Churn
# ==========================================================

print("\nCorrelation with Churn:")
print(df.corr(numeric_only=True)["Churn"].sort_values(ascending=False))