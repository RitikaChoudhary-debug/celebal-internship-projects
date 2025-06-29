import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy.stats import zscore

# -------------------------------------
# Setup
# -------------------------------------

# Create directory for saving visuals
IMAGE_DIR = "titanic_visuals"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
sns.set_theme(style="whitegrid")

# -------------------------------------
# Load & Clean Data
# -------------------------------------

def load_dataset():
    logging.info("Loading Titanic dataset...")
    df = sns.load_dataset('titanic')
    logging.info(f"Dataset shape: {df.shape}")
    return df

def clean_data(df):
    df = df.copy()
    df['age'].fillna(df['age'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df.drop(columns=['deck'], inplace=True)
    return df

# -------------------------------------
# Outlier Detection using Z-Score
# -------------------------------------

def detect_outliers(df):
    df = df.copy()
    df['fare_z'] = np.abs(zscore(df['fare'].fillna(0)))
    df['is_outlier'] = df['fare_z'] > 3
    logging.info(f"Detected {df['is_outlier'].sum()} outliers in 'fare'")
    return df

def plot_outliers(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x='age', y='fare',
        hue='is_outlier',
        data=df,
        palette={True: 'red', False: 'blue'},
        alpha=0.6
    )
    plt.title("Fare Outliers (Z-score > 3)")
    plt.savefig(f"{IMAGE_DIR}/scatter_outliers.png")
    plt.close()
    logging.info("Saved scatterplot: fare outliers")

# -------------------------------------
# Plotting Functions
# -------------------------------------

def plot_histograms(df):
    for col in ['age', 'fare']:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"{col.capitalize()} Distribution")
        plt.savefig(f"{IMAGE_DIR}/hist_{col}.png")
        plt.close()

def plot_boxplots(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['fare'])
    plt.title("Fare Boxplot")
    plt.savefig(f"{IMAGE_DIR}/box_fare.png")
    plt.close()

def plot_counts(df):
    for col in ['sex', 'class', 'embarked']:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df)
        plt.title(f"{col.capitalize()} Count")
        plt.savefig(f"{IMAGE_DIR}/count_{col}.png")
        plt.close()

def plot_survival_comparisons(df):
    for col in ['class', 'sex', 'embarked']:
        plt.figure(figsize=(6, 4))
        sns.barplot(x=col, y='survived', data=df)
        plt.title(f"Survival Rate by {col.capitalize()}")
        plt.savefig(f"{IMAGE_DIR}/bar_survival_{col}.png")
        plt.close()

def plot_relationships(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='age', y='fare', hue='survived', data=df, alpha=0.6)
    plt.title("Age vs Fare by Survival")
    plt.savefig(f"{IMAGE_DIR}/scatter_age_fare.png")
    plt.close()

def plot_corr_heatmap(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(f"{IMAGE_DIR}/heatmap_corr.png")
    plt.close()

# -------------------------------------
# Save Cleaned Data
# -------------------------------------

def save_cleaned_data(df, filename="titanic_cleaned.csv"):
    df.to_csv(filename, index=False)
    logging.info(f"Cleaned dataset saved as: {filename}")

# -------------------------------------
# Main EDA Runner
# -------------------------------------

def main():
    df = load_dataset()
    df = clean_data(df)
    df = detect_outliers(df)

    plot_histograms(df)
    plot_boxplots(df)
    plot_counts(df)
    plot_corr_heatmap(df)
    plot_relationships(df)
    plot_survival_comparisons(df)
    plot_outliers(df)

    save_cleaned_data(df)

    # Print sample of outliers
    outliers_df = df[df['is_outlier'] == True][['age', 'fare', 'fare_z', 'class', 'sex', 'survived']]
    logging.info("Sample fare outliers (Z-score > 3):")
    print(outliers_df.head())

    logging.info(f"EDA complete. Visuals saved in: {IMAGE_DIR}")

# -------------------------------------
# Run Script
# -------------------------------------

if __name__ == "__main__":
    main()
