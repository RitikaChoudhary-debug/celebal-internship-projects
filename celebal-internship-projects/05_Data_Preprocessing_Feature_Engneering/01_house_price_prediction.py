# 01_house_price_prediction.py
# Internship Project - Celebal Technologies
# Author: Ritika Choudhary | B.Tech Bioinformatics (Prefinal Year)
# Objective: Preprocess and prepare house price data for machine learning models by cleaning, handling missing values, encoding, and visualization.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
from sklearn.preprocessing import LabelEncoder

# Setup logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load training and testing datasets
def load_data(train_path, test_path):
    logging.info("Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logging.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test

# Display missing values in the dataset
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table = mis_val_table[mis_val_table['Missing Values'] > 0]
    logging.info(f"Missing values:\n{mis_val_table.head()}\n")
    return mis_val_table

# Heatmap to visualize missing values
def plot_missing_data(df):
    logging.info("Plotting missing data heatmap...")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.tight_layout()
    os.makedirs('visuals', exist_ok=True)
    plt.savefig('visuals/missing_data.png')
    plt.close()
    logging.info("Missing data heatmap saved to 'visuals/missing_data.png'")

# Barplot of missing value counts
def plot_missing_bar(df):
    mis_val = df.isnull().sum()
    mis_val = mis_val[mis_val > 0].sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=mis_val.values, y=mis_val.index, palette='magma')
    plt.title('Missing Values Count Per Column')
    plt.xlabel('Number of Missing Values')
    plt.tight_layout()
    plt.savefig('visuals/missing_values_barplot.png')
    plt.close()
    logging.info("Missing value barplot saved to 'visuals/missing_values_barplot.png'")

# Plot distribution of the target variable SalePrice
def plot_target_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['SalePrice'], kde=True, color='skyblue', bins=30)
    plt.title('Distribution of SalePrice')
    plt.xlabel('SalePrice')
    plt.ylabel('Frequency')
    os.makedirs('visuals', exist_ok=True)
    plt.savefig('visuals/target_distribution.png')
    plt.close()
    logging.info("Target distribution plot saved to 'visuals/target_distribution.png'")

# Correlation heatmap of numerical features
def plot_correlation_heatmap(df):
    plt.figure(figsize=(14, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('visuals/correlation_heatmap.png')
    plt.close()
    logging.info("Correlation heatmap saved to 'visuals/correlation_heatmap.png'")

# Top 10 correlated features with SalePrice
def plot_top10_correlations(df):
    corr = df.corr()
    top_corr = corr['SalePrice'].abs().sort_values(ascending=False)[1:11]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_corr.values, y=top_corr.index, palette='crest')
    plt.title('Top 10 Features Correlated with SalePrice')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('visuals/top10_corr_features.png')
    plt.close()
    logging.info("Top 10 correlation plot saved to 'visuals/top10_corr_features.png'")

# Fill missing values: median for numerical, mode for categorical
def handle_missing_values(df):
    logging.info("Handling missing values...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=[object]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

# Encode categorical columns using LabelEncoder
def encode_categorical(df):
    logging.info("Encoding categorical variables...")
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = label_enc.fit_transform(df[col])
    return df

# Save preprocessed data to disk
def save_clean_data(train, test, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, 'clean_train.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'clean_test.csv'), index=False)
    logging.info("Cleaned datasets saved to 'outputs/' folder.")

if __name__ == '__main__':
    TRAIN_PATH = r'C:\Users\lenovo\Desktop\celebal-internship-projects\celebal-internship-projects\05_Data_Preprocessing_Feature_Engneering\data\train.csv'
    TEST_PATH = r'C:\Users\lenovo\Desktop\celebal-internship-projects\celebal-internship-projects\05_Data_Preprocessing_Feature_Engneering\data\test.csv'

    train, test = load_data(TRAIN_PATH, TEST_PATH)

    # Visual insights before cleaning
    missing_values_table(train)
    plot_missing_data(train)
    plot_missing_bar(train)
    plot_target_distribution(train)
    encoded_train = encode_categorical(train.copy())
    plot_correlation_heatmap(encoded_train)
    plot_top10_correlations(encoded_train)


    # Combine, clean, encode, and split
    combined = pd.concat([train.drop('SalePrice', axis=1), test])
    combined = handle_missing_values(combined)
    combined = encode_categorical(combined)

    train_clean = combined.iloc[:len(train), :]
    train_clean['SalePrice'] = train['SalePrice']
    test_clean = combined.iloc[len(train):, :]

    save_clean_data(train_clean, test_clean)
    logging.info("\u2705 Data preprocessing pipeline completed successfully.")