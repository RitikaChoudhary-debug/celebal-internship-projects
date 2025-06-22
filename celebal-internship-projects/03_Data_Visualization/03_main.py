# Python script for Data Visualization

"""
Title: Mental Health in Tech Survey
Source: Open Sourcing Mental Illness (OSMI) via Kaggle
Description: Visualizes mental health support, awareness, and patterns among tech workers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(script_dir, "02_images")
os.makedirs(images_path, exist_ok=True)

# Load dataset
data_path = os.path.join(script_dir, "01_mental_health_survey.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)

# Clean gender column
df['Gender'] = df['Gender'].str.lower()
df['Gender'] = df['Gender'].replace({
    'male': 'Male', 'm': 'Male', 'man': 'Male', 'cis male': 'Male',
    'female': 'Female', 'f': 'Female', 'woman': 'Female', 'cis female': 'Female',
    'trans-female': 'Trans', 'trans woman': 'Trans', 'transgender': 'Trans',
    'genderqueer': 'Other', 'non-binary': 'Other', 'agender': 'Other',
})
df['Gender'] = df['Gender'].apply(lambda x: 'Other' if x not in ['Male', 'Female', 'Trans'] else x)

# Filter out unreasonable age values
df = df[(df['Age'] > 10) & (df['Age'] < 100)]

# Gender Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df, palette='pastel')
plt.title("Gender Distribution")
plt.savefig(os.path.join(images_path, "gender_distribution.png"))
plt.close()

# Remote Work vs Benefits
pd.crosstab(df['remote_work'], df['benefits']).plot(kind='bar', stacked=True, colormap='viridis')
plt.title("Remote Work vs Mental Health Benefits")
plt.xlabel("Remote Work Allowed")
plt.ylabel("Employee Count")
plt.tight_layout()
plt.savefig(os.path.join(images_path, "remote_vs_benefits.png"))
plt.close()

# Work Interference vs Family History
plt.figure(figsize=(8, 5))
sns.countplot(x='work_interfere', hue='family_history', data=df, palette='coolwarm')
plt.title("Work Interference vs Family History")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(images_path, "work_interfere_family_history.png"))
plt.close()

# Mental Health Consequences
plt.figure(figsize=(6, 4))
sns.countplot(x='mental_health_consequence', data=df, palette='mako')
plt.title("Mental Health Consequences at Workplace")
plt.tight_layout()
plt.savefig(os.path.join(images_path, "mental_health_consequence.png"))
plt.close()

# Supervisor Support vs Seeking Help
plt.figure(figsize=(7, 5))
sns.countplot(x='supervisor', hue='seek_help', data=df, palette='Set2')
plt.title("Supervisor Support vs Seeking Help")
plt.tight_layout()
plt.savefig(os.path.join(images_path, "supervisor_support_seek_help.png"))
plt.close()

# Correlation Heatmap
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

plt.figure(figsize=(12, 10))
sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Mental Health Features")
plt.tight_layout()
plt.savefig(os.path.join(images_path, "correlation_heatmap.png"))
plt.close()

# Pie Chart: Mental Health Benefits
benefits_counts = df['benefits'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(benefits_counts, labels=benefits_counts.index, autopct='%1.1f%%', startangle=140,
        colors=sns.color_palette('pastel'))
plt.title("Mental Health Benefits Provided")
plt.tight_layout()
plt.savefig(os.path.join(images_path, "pie_mental_health_benefits.png"))
plt.close()

# Boxplot: Age by Gender
plt.figure(figsize=(8, 5))
sns.boxplot(x='Gender', y='Age', data=df, palette='pastel')
plt.title("Age Distribution by Gender")
plt.tight_layout()
plt.savefig(os.path.join(images_path, "boxplot_age_by_gender.png"))
plt.close()

# Histogram: Age Distribution
plt.figure(figsize=(7, 5))
sns.histplot(df['Age'], bins=15, kde=True, color='skyblue')
plt.title("Age Distribution of Respondents")
plt.tight_layout()
plt.savefig(os.path.join(images_path, "histogram_age_distribution.png"))
plt.close()

# Save cleaned dataset
cleaned_path = os.path.join(script_dir, "04_cleaned_mental_health_data.csv")
df.to_csv(cleaned_path, index=False)

print(f"Cleaned data saved as: {cleaned_path}")
print(f"All visualizations saved in: {images_path}")
