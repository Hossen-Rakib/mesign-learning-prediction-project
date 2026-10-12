import pandas as pd
import numpy as np

def generate_heart_disease_dataset(file_path="heart_disease.csv", num_samples=300):
    """
    Generates a realistic Heart Disease dataset with intentionally added
    missing values (NaN) and duplicate rows for demonstrating data preprocessing.
    """
    np.random.seed(42)

    # Generate synthetic feature data with realistic medical ranges
    age = np.random.randint(30, 75, size=num_samples)
    sex = np.random.choice([0, 1], size=num_samples, p=[0.35, 0.65])  # 0: Female, 1: Male
    cp = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.45, 0.20, 0.25, 0.10])  # Chest Pain Type
    trestbps = np.random.randint(95, 180, size=num_samples).astype(float)  # Resting Blood Pressure
    chol = np.random.randint(150, 360, size=num_samples).astype(float)  # Serum Cholesterol
    fbs = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])  # Fasting Blood Sugar > 120 mg/dl
    thalach = np.random.randint(90, 200, size=num_samples).astype(float)  # Max Heart Rate
    exang = np.random.choice([0, 1], size=num_samples, p=[0.65, 0.35])  # Exercise Induced Angina

    # Target calculation based on risk factors
    risk_score = (
        (age > 50).astype(int) * 1.2 +
        (sex == 1).astype(int) * 0.8 +
        (cp > 0).astype(int) * 1.8 +
        (trestbps > 135).astype(int) * 1.1 +
        (chol > 240).astype(int) * 1.0 +
        (thalach < 140).astype(int) * 1.5 +
        (exang == 1).astype(int) * 1.6
    )

    probability = 1 / (1 + np.exp(-(risk_score - 4.2)))
    target = (np.random.rand(num_samples) < probability).astype(int)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'thalach': thalach,
        'exang': exang,
        'target': target
    })

    # Intentionally insert a few missing values (NaN) for preprocessing demo
    nan_indices_chol = np.random.choice(df.index, size=6, replace=False)
    nan_indices_bp = np.random.choice(df.index, size=4, replace=False)
    df.loc[nan_indices_chol, 'chol'] = np.nan
    df.loc[nan_indices_bp, 'trestbps'] = np.nan

    # Intentionally insert duplicate rows for preprocessing demo
    duplicates = df.iloc[:5].copy()  # Duplicate first 5 rows
    df = pd.concat([df, duplicates], ignore_index=True)

    df.to_csv(file_path, index=False)
    print(f"Dataset successfully created at '{file_path}' with {len(df)} samples (including NaN & duplicates).")
    return df

if __name__ == "__main__":
    generate_heart_disease_dataset()
