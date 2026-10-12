import os
import sys
import warnings
import pandas as pd
from dataset_generator import generate_heart_disease_dataset
from eda import perform_eda, print_stat_table
from models import HeartDiseasePredictor

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Suppress warnings
warnings.filterwarnings('ignore')
DATASET_PATH = "heart_disease.csv"

# Rich UI setup
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from rich.prompt import Prompt, IntPrompt, FloatPrompt
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

def clear_screen():
    """Clears the console screen for Windows and Unix systems."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    if HAS_RICH:
        title_panel = Panel(
            "[bold white]HEART DISEASE PREDICTION SYSTEM[/bold white]\n"
            "[dim cyan]Machine Learning Classification & Medical Data Analytics CLI[/dim cyan]",
            style="bold blue",
            box=box.DOUBLE,
            expand=True
        )
        console.print(title_panel)
    else:
        print("\n" + "=" * 70)
        print("         HEART DISEASE PREDICTION SYSTEM (ML CLASSIFICATION)")
        print("=" * 70)

def ensure_dataset():
    if not os.path.exists(DATASET_PATH):
        if HAS_RICH:
            console.print(f"[bold yellow]Dataset '{DATASET_PATH}' missing. Generating initial synthetic medical dataset...[/bold yellow]")
        else:
            print(f"[!] Dataset '{DATASET_PATH}' missing. Generating dataset...")
        generate_heart_disease_dataset(DATASET_PATH)

def prompt_patient_input():
    if HAS_RICH:
        console.print(Panel("[bold cyan]PATIENT MEDICAL PARAMETERS INPUT FORM[/bold cyan]", box=box.ROUNDED))
    else:
        print("\n+--- PATIENT MEDICAL INFORMATION ENTRY ---+")

    try:
        if HAS_RICH:
            age = IntPrompt.ask("[bold white]1. Age in years[/bold white] (e.g. 52)")
            sex_str = Prompt.ask("[bold white]2. Sex[/bold white] (1 = Male, 0 = Female)", choices=["0", "1"], default="1")
            sex = int(sex_str)
            
            console.print("\n[bold white]3. Chest Pain Type (cp):[/bold white]")
            console.print("   [cyan][0][/cyan] Typical Angina    [cyan][1][/cyan] Atypical Angina")
            console.print("   [cyan][2][/cyan] Non-anginal Pain  [cyan][3][/cyan] Asymptomatic")
            cp_str = Prompt.ask("   Select Chest Pain Type", choices=["0", "1", "2", "3"], default="0")
            cp = int(cp_str)
            
            trestbps = FloatPrompt.ask("\n[bold white]4. Resting Blood Pressure in mm Hg[/bold white] (e.g. 130)")
            chol = FloatPrompt.ask("[bold white]5. Serum Cholesterol in mg/dl[/bold white] (e.g. 240)")
            fbs_str = Prompt.ask("[bold white]6. Fasting Blood Sugar > 120 mg/dl[/bold white] (1 = Yes, 0 = No)", choices=["0", "1"], default="0")
            fbs = int(fbs_str)
            thalach = FloatPrompt.ask("[bold white]7. Maximum Heart Rate Achieved (thalach)[/bold white] (e.g. 150)")
            exang_str = Prompt.ask("[bold white]8. Exercise Induced Angina[/bold white] (1 = Yes, 0 = No)", choices=["0", "1"], default="0")
            exang = int(exang_str)
        else:
            age = int(input("1. Age (e.g. 52): "))
            sex = int(input("2. Sex (1=Male, 0=Female): "))
            cp = int(input("3. Chest Pain (0-3): "))
            trestbps = float(input("4. Resting BP (e.g. 130): "))
            chol = float(input("5. Cholesterol (e.g. 240): "))
            fbs = int(input("6. Fasting BS > 120 (1=Yes, 0=No): "))
            thalach = float(input("7. Max HR (e.g. 150): "))
            exang = int(input("8. Exercise Angina (1=Yes, 0=No): "))

        patient = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'thalach': thalach,
            'exang': exang
        }
        return patient
    except ValueError:
        if HAS_RICH:
            console.print("[bold red]Invalid numerical input detected! Please enter standard values.[/bold red]")
        else:
            print("[!] Invalid input entered.")
        return None

def main():
    ensure_dataset()
    predictor = HeartDiseasePredictor(DATASET_PATH)

    while True:
        clear_screen()
        print_banner()

        if HAS_RICH:
            menu_table = Table(show_header=False, box=box.ROUNDED, expand=True)
            menu_table.add_column("Option", style="bold cyan", width=8, justify="center")
            menu_table.add_column("Action Description", style="bold white")
            
            menu_table.add_row("[1]", "View Dataset Preview & Statistical Summary")
            menu_table.add_row("[2]", "Run Exploratory Data Analysis (EDA) & Render Plots")
            menu_table.add_row("[3]", "Train & Compare ML Models (Decision Tree, LR, RF, KNN)")
            menu_table.add_row("[4]", "Predict Patient Heart Disease (Interactive Diagnosis)")
            menu_table.add_row("[5]", "Reset & Regenerate Heart Disease Dataset CSV")
            menu_table.add_row("[6]", "Exit Application")
            
            console.print(menu_table)
            choice = Prompt.ask("\n[bold yellow]Select Menu Option[/bold yellow]", choices=["1", "2", "3", "4", "5", "6"], default="1")
        else:
            print("  1. View Dataset Summary")
            print("  2. Run EDA")
            print("  3. Train Models")
            print("  4. Predict Patient Heart Disease")
            print("  5. Reset Dataset")
            print("  6. Exit")
            choice = input("\nSelect Option (1-6): ").strip()

        if choice == '1':
            clear_screen()
            df = pd.read_csv(DATASET_PATH)
            if HAS_RICH:
                console.print(Panel(f"[bold cyan]DATASET PREVIEW & METRICS ({len(df)} Patients)[/bold cyan]", box=box.ROUNDED))
                
                # Show first 5 rows in Rich Table
                head_table = Table(title="[bold yellow]Dataset Sample Preview (First 5 Patients)[/bold yellow]", box=box.ROUNDED)
                for col in df.columns:
                    head_table.add_column(col, style="cyan", justify="right")
                for _, row in df.head().iterrows():
                    head_table.add_row(*[str(val) for val in row])
                console.print(head_table)
                
                # Print stats table
                print_stat_table(df)
            else:
                print(df.head())
                print_stat_table(df)
            
            input("\nPress Enter to return to main menu...")

        elif choice == '2':
            clear_screen()
            df = pd.read_csv(DATASET_PATH)
            perform_eda(df, show_interactive=True)
            input("\nPress Enter to return to main menu...")

        elif choice == '3':
            clear_screen()
            predictor.load_and_preprocess_data()
            predictor.train_all_models()
            predictor.display_model_evaluation()
            input("\nPress Enter to return to main menu...")

        elif choice == '4':
            clear_screen()
            patient = prompt_patient_input()
            if patient:
                if HAS_RICH:
                    console.print("\n[bold cyan]Select Machine Learning Model for Prediction:[/bold cyan]")
                    console.print("  [1] [bold green]Decision Tree Classifier[/bold green] (Recommended / Highest Interpretability)")
                    console.print("  [2] [bold yellow]Random Forest Classifier[/bold yellow]")
                    console.print("  [3] [bold magenta]Logistic Regression[/bold magenta]")
                    console.print("  [4] [bold cyan]K-Nearest Neighbors (KNN)[/bold cyan]")
                    alg_choice = Prompt.ask("Choose Algorithm", choices=["1", "2", "3", "4"], default="1")
                else:
                    alg_choice = input("Choose Algorithm (1-4): ").strip()

                alg_map = {
                    '1': "Decision Tree Classifier",
                    '2': "Random Forest Classifier",
                    '3': "Logistic Regression",
                    '4': "K-Nearest Neighbors (KNN)"
                }
                selected_alg = alg_map.get(alg_choice, "Decision Tree Classifier")

                prediction, proba = predictor.predict_patient(patient, algorithm=selected_alg)

                if HAS_RICH:
                    if prediction == 1:
                        conf_str = f"Confidence: [bold red]{proba[1]*100:.1f}%[/bold red]" if proba is not None else ""
                        
                        # Probability Meter
                        bar_len = 25
                        filled = int(proba[1] * bar_len) if proba is not None else bar_len
                        prob_bar = f"[red]{'■' * filled}[/red][dim grey]{'░' * (bar_len - filled)}[/dim grey]"
                        
                        report_content = (
                            f"[bold red][!] HIGH RISK DETECTED — POSITIVE FOR HEART DISEASE[/bold red]\n"
                            f"[bold white]Model Used  :[/] [yellow]{selected_alg}[/yellow]\n"
                            f"[bold white]Risk Gauge  :[/] {prob_bar} ({conf_str})\n\n"
                            f"[bold red]Recommended Clinical Action:[/bold red]\n"
                            f"  • Immediate consultation with a certified cardiologist.\n"
                            f"  • Perform detailed ECG, Stress Test, and Coronary Angiogram."
                        )
                        console.print(Panel(report_content, title="[bold red]PATIENT DIAGNOSIS REPORT[/bold red]", border_style="red", box=box.HEAVY))
                    else:
                        conf_str = f"Confidence: [bold green]{proba[0]*100:.1f}%[/bold green]" if proba is not None else ""
                        
                        # Probability Meter
                        bar_len = 25
                        filled = int(proba[0] * bar_len) if proba is not None else bar_len
                        prob_bar = f"[green]{'■' * filled}[/green][dim grey]{'░' * (bar_len - filled)}[/dim grey]"
                        
                        report_content = (
                            f"[bold green][✓] LOW RISK DETECTED — HEALTHY CARDIAC PROFILE[/bold green]\n"
                            f"[bold white]Model Used  :[/] [cyan]{selected_alg}[/cyan]\n"
                            f"[bold white]Health Gauge:[/] {prob_bar} ({conf_str})\n\n"
                            f"[bold green]Recommended Lifestyle Advice:[/bold green]\n"
                            f"  • Patient parameters align with normal healthy cardiac ranges.\n"
                            f"  • Maintain balanced diet, regular exercise, and annual health checkups."
                        )
                        console.print(Panel(report_content, title="[bold green]PATIENT DIAGNOSIS REPORT[/bold green]", border_style="green", box=box.HEAVY))
                else:
                    if prediction == 1:
                        print("\n[!] STATUS: HEART DISEASE DETECTED (HIGH RISK)")
                    else:
                        print("\n[+] STATUS: NO HEART DISEASE DETECTED (LOW RISK)")
            input("\nPress Enter to return to main menu...")

        elif choice == '5':
            clear_screen()
            generate_heart_disease_dataset(DATASET_PATH)
            predictor = HeartDiseasePredictor(DATASET_PATH)
            if HAS_RICH:
                console.print(Panel("[bold green]Synthetic heart disease dataset CSV reset and regenerated successfully![/bold green]", box=box.ROUNDED))
            else:
                print("[+] Dataset reset successfully.")
            input("\nPress Enter to return to main menu...")

        elif choice == '6':
            clear_screen()
            if HAS_RICH:
                console.print(Panel("[bold cyan]Thank you for using Heart Disease Prediction System. Goodbye![/bold cyan]", box=box.ROUNDED))
            else:
                print("Exiting. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()
