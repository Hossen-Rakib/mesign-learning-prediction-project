import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# Rich UI setup
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

class HeartDiseasePredictor:
    def __init__(self, csv_file="heart_disease.csv"):
        self.csv_file = csv_file
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def load_and_preprocess_data(self, test_size=0.2, random_state=42):
        """
        Loads dataset and performs standard data cleaning & preprocessing:
        1. Duplicate Removal
        2. Missing Value Imputation (Median)
        3. Train / Test Splitting
        4. Feature Scaling
        """
        self.df = pd.read_csv(self.csv_file)
        
        # 1. Duplicate Handling
        initial_count = len(self.df)
        duplicate_count = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        
        # 2. Missing Value (NaN) Handling - Median Imputation
        missing_count = self.df.isnull().sum().sum()
        for col in ['trestbps', 'chol', 'thalach']:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())

        X = self.df.drop('target', axis=1)
        y = self.df['target']

        # Train/Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale numerical features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        if HAS_RICH:
            prep_info = (
                f"[bold white]• Raw Dataset Records :[/] [cyan]{initial_count}[/]\n"
                f"[bold white]• Duplicate Rows Removed:[/] [green]{duplicate_count}[/]\n"
                f"[bold white]• Missing Values Fixed  :[/] [green]{missing_count} (Median Imputed)[/]\n"
                f"[bold white]• Cleaned Dataset Size  :[/] [bold cyan]{len(self.df)}[/]\n"
                f"[bold white]• Train / Test Split    :[/] [bold green]{len(self.X_train)} Train (80%) | {len(self.X_test)} Test (20%)[/]"
            )
            console.print(Panel(prep_info, title="[bold cyan]Data Preprocessing Pipeline Report[/bold cyan]", box=box.ROUNDED))
        else:
            print(f"[+] Data preprocessed: {len(self.df)} records ({len(self.X_train)} Train / {len(self.X_test)} Test)")

    def train_all_models(self):
        """
        Trains 4 Classification Models:
        - Decision Tree Classifier
        - Logistic Regression
        - Random Forest Classifier
        - K-Nearest Neighbors (KNN)
        """
        if self.df is None:
            self.load_and_preprocess_data()

        self.models = {
            "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
        }

        for name, model in self.models.items():
            if name in ["Logistic Regression", "K-Nearest Neighbors (KNN)"]:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            cm = confusion_matrix(self.y_test, y_pred)

            self.results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'confusion_matrix': cm,
                'predictions': y_pred
            }

    def display_model_evaluation(self):
        """
        Displays professional evaluation metrics table and Decision Tree breakdown.
        """
        if not self.results:
            self.train_all_models()

        if HAS_RICH:
            table = Table(title="[bold yellow]ML Model Classification Performance Comparison[/bold yellow]", box=box.ROUNDED, expand=True)
            table.add_column("Model Algorithm", style="bold white")
            table.add_column("Accuracy", justify="right", style="bold green")
            table.add_column("Precision", justify="right", style="cyan")
            table.add_column("Recall", justify="right", style="yellow")
            table.add_column("F1-Score", justify="right", style="magenta")
            
            # Find best model
            best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]

            for name, m in self.results.items():
                is_best = (name == best_model_name)
                badge = " [bold green][BEST][/bold green]" if is_best else ""
                acc_str = f"{m['accuracy']*100:.2f}%"
                prec_str = f"{m['precision']*100:.2f}%"
                rec_str = f"{m['recall']*100:.2f}%"
                f1_str = f"{m['f1']*100:.2f}%"
                
                table.add_row(f"{name}{badge}", acc_str, prec_str, rec_str, f1_str)
            
            console.print(table)
            
            # Decision Tree Breakdown Card
            dt_model = self.models["Decision Tree Classifier"]
            dt_metrics = self.results["Decision Tree Classifier"]
            cm = dt_metrics['confusion_matrix']

            # Confusion Matrix Table inside panel
            cm_text = (
                f"[bold white]Actual Healthy  :[/] [green]{cm[0][0]:3d}[/green] True Negatives    | [red]{cm[0][1]:3d}[/red] False Positives\n"
                f"[bold white]Actual Disease  :[/] [yellow]{cm[1][0]:3d}[/yellow] False Negatives   | [green]{cm[1][1]:3d}[/green] True Positives"
            )
            
            # Feature Importance Visual Meters
            importances = dt_model.feature_importances_
            feature_names = self.df.drop('target', axis=1).columns
            sorted_feats = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            
            feat_meters = []
            for feat, imp in sorted_feats:
                if imp > 0.001:
                    filled = int(imp * 30)
                    bar = f"[cyan]{'■' * filled}[/cyan][dim grey]{'░' * (30 - filled)}[/dim grey]"
                    feat_meters.append(f"[bold white]{feat:<10}[/] {bar} [bold yellow]{imp*100:5.1f}%[/bold yellow]")
            
            dt_card_content = (
                f"[bold cyan]Overall Accuracy:[/] [bold green]{dt_metrics['accuracy']*100:.2f}%[/bold green]  |  "
                f"[bold cyan]F1-Score:[/] [bold green]{dt_metrics['f1']*100:.2f}%[/bold green]\n\n"
                f"[bold yellow]• Confusion Matrix Breakdown:[/bold yellow]\n{cm_text}\n\n"
                f"[bold yellow]• Top Feature Importance Impact:[/bold yellow]\n" + "\n".join(feat_meters)
            )
            
            console.print(Panel(dt_card_content, title="[bold green]Decision Tree Classifier Deep Breakdown[/bold green]", box=box.ROUNDED))

        elif HAS_TABULATE:
            table_data = []
            for name, m in self.results.items():
                table_data.append([
                    name, 
                    f"{m['accuracy']*100:.2f}%", 
                    f"{m['precision']*100:.2f}%", 
                    f"{m['recall']*100:.2f}%", 
                    f"{m['f1']*100:.2f}%"
                ])
            print(tabulate(table_data, headers=["Model Name", "Accuracy", "Precision", "Recall", "F1 Score"], tablefmt="grid"))
        else:
            for name, m in self.results.items():
                print(f"{name:<26} -> Accuracy: {m['accuracy']*100:6.2f}%")

    def predict_patient(self, patient_dict, algorithm="Decision Tree Classifier"):
        """
        Predicts heart disease for a patient input dictionary.
        """
        if algorithm not in self.models:
            self.train_all_models()

        model = self.models[algorithm]
        feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang']
        patient_df = pd.DataFrame([patient_dict])[feature_order]

        if algorithm in ["Logistic Regression", "K-Nearest Neighbors (KNN)"]:
            patient_data = self.scaler.transform(patient_df)
        else:
            patient_data = patient_df

        prediction = model.predict(patient_data)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(patient_data)[0]

        return prediction, proba

if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    predictor.load_and_preprocess_data()
    predictor.train_all_models()
    predictor.display_model_evaluation()
