import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Suppress visual library warnings
warnings.filterwarnings('ignore')

# Rich library imports for clean executive terminal UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.text import Text
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

# Set pandas display options
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

def print_stat_table(df):
    """Prints a beautifully styled table for dataset statistical summary."""
    stats = df.describe().T[['mean', '50%', 'std', 'min', 'max']].rename(columns={'50%': 'median'}).reset_index()
    stats.columns = ['Feature Name', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
    
    if HAS_RICH:
        table = Table(title="[bold cyan]Feature Statistical Summary[/bold cyan]", box=box.ROUNDED, header_style="bold magenta", expand=True)
        table.add_column("Feature Name", style="bold white", justify="left")
        table.add_column("Mean", style="cyan", justify="right")
        table.add_column("Median", style="green", justify="right")
        table.add_column("Std Dev", style="yellow", justify="right")
        table.add_column("Min", style="blue", justify="right")
        table.add_column("Max", style="bright_red", justify="right")
        
        for _, row in stats.iterrows():
            table.add_row(
                str(row['Feature Name']),
                f"{row['Mean']:.2f}",
                f"{row['Median']:.2f}",
                f"{row['Std Dev']:.2f}",
                f"{row['Min']:.2f}",
                f"{row['Max']:.2f}"
            )
        console.print(table)
    elif HAS_TABULATE:
        print(tabulate(stats, headers='keys', tablefmt='grid', showindex=False, floatfmt=".2f"))
    else:
        print(stats.to_string(index=False))

def render_terminal_bar_chart(df):
    """Draws a clean, styled Heart Disease Target Distribution chart."""
    disease_counts = df['target'].value_counts()
    no_disease = disease_counts.get(0, 0)
    has_disease = disease_counts.get(1, 0)
    total = len(df)
    
    no_disease_pct = (no_disease / total) * 100 if total > 0 else 0
    has_disease_pct = (has_disease / total) * 100 if total > 0 else 0

    if HAS_RICH:
        table = Table(title="[bold cyan]Heart Disease Target Distribution[/bold cyan]", box=box.ROUNDED, expand=True)
        table.add_column("Category", style="bold white", width=22)
        table.add_column("Visual Distribution Meter", width=38)
        table.add_column("Patients", style="bold yellow", justify="right", width=12)
        table.add_column("Percentage", style="bold green", justify="right", width=12)
        
        # Clean progress bar fill
        bar_len = 25
        b1_filled = int((no_disease_pct / 100) * bar_len)
        b2_filled = int((has_disease_pct / 100) * bar_len)
        
        meter1 = f"[bold green]{'■' * b1_filled}[/bold green][dim grey]{'░' * (bar_len - b1_filled)}[/dim grey]"
        meter2 = f"[bold bright_red]{'■' * b2_filled}[/bold bright_red][dim grey]{'░' * (bar_len - b2_filled)}[/dim grey]"
        
        table.add_row("[green]💚 Healthy (0)[/green]", meter1, str(no_disease), f"{no_disease_pct:.1f}%")
        table.add_row("[bright_red]❤️ Heart Disease (1)[/bright_red]", meter2, str(has_disease), f"{has_disease_pct:.1f}%")
        
        console.print(table)
    else:
        print(f"Healthy (0)     : {no_disease} ({no_disease_pct:.1f}%)")
        print(f"Heart Disease (1): {has_disease} ({has_disease_pct:.1f}%)")

def render_terminal_scatter_plot(df):
    """Draws a clean Age vs Heart Rate (thalach) distribution breakdown."""
    age_groups = [
        ("30-40 Yrs", df[(df['age'] >= 30) & (df['age'] < 40)]['thalach'].mean()),
        ("40-50 Yrs", df[(df['age'] >= 40) & (df['age'] < 50)]['thalach'].mean()),
        ("50-60 Yrs", df[(df['age'] >= 50) & (df['age'] < 60)]['thalach'].mean()),
        ("60-70 Yrs", df[(df['age'] >= 60) & (df['age'] < 70)]['thalach'].mean()),
        ("70+   Yrs", df[df['age'] >= 70]['thalach'].mean())
    ]
    
    if HAS_RICH:
        table = Table(title="[bold cyan]Age Group vs Max Heart Rate (thalach)[/bold cyan]", box=box.ROUNDED, expand=True)
        table.add_column("Age Group", style="bold white", width=16)
        table.add_column("Max HR Metric Gauge", width=38)
        table.add_column("Avg Heart Rate", style="bold cyan", justify="right", width=16)
        
        max_possible_hr = 220.0
        for group_name, avg_hr in age_groups:
            if pd.isna(avg_hr):
                avg_hr = 0.0
            
            # Meter percentage relative to 200 bpm max limit
            pct = min(avg_hr / 200.0, 1.0)
            bar_len = 25
            filled = int(pct * bar_len)
            
            # Gradient style according to rate level
            color = "green" if avg_hr > 145 else "yellow" if avg_hr > 135 else "bright_red"
            meter = f"[{color}]{'■' * filled}[/{color}][dim grey]{'░' * (bar_len - filled)}[/dim grey]"
            
            table.add_row(group_name, meter, f"{avg_hr:.1f} bpm")
        
        console.print(table)
    else:
        for group_name, avg_hr in age_groups:
            print(f"{group_name} : {avg_hr:.1f} bpm")

def perform_eda(df, save_dir="eda_plots", show_interactive=False):
    """
    Performs Exploratory Data Analysis (EDA) on the Heart Disease dataset
    with clean, beautiful visual rich outputs and saved high-res PNG plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if HAS_RICH:
        console.print(Panel.fit("[bold yellow]📊 EXPLORATORY DATA ANALYSIS (EDA) DASHBOARD[/bold yellow]", border_style="bold cyan"))
        
        # 1. Dataset Overview Panel
        overview_text = (
            f"[bold white]• Total Patient Records:[/] [bold cyan]{df.shape[0]}[/]\n"
            f"[bold white]• Attributes (Features):[/] [bold cyan]{df.shape[1]}[/]\n"
            f"[bold white]• Feature Names:[/] [dim]{', '.join(df.columns)}[/]"
        )
        console.print(Panel(overview_text, title="[bold green]1. Dataset Overview[/bold green]", box=box.ROUNDED))
        
        # 2. Data Quality Panel
        missing = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        status_missing = "[green]✓ Clean (0 missing)[/green]" if missing == 0 else f"[red]⚠ {missing} Missing Values[/red]"
        status_dup = "[green]✓ Clean (0 duplicates)[/green]" if duplicates == 0 else f"[yellow]⚠ {duplicates} Duplicates[/yellow]"
        
        quality_text = f"[bold white]• Missing Values Status:[/] {status_missing}\n[bold white]• Duplicate Rows Status:[/] {status_dup}"
        console.print(Panel(quality_text, title="[bold green]2. Data Quality Assessment[/bold green]", box=box.ROUNDED))
        
        # 3. Statistical Table
        print_stat_table(df)
        
        # 4. Target Distribution Chart
        render_terminal_bar_chart(df)
        
        # 5. Age vs Heart Rate Chart
        render_terminal_scatter_plot(df)
        
    else:
        print("\n=== EXPLORATORY DATA ANALYSIS REPORT ===")
        print(f"Records: {df.shape[0]}, Features: {df.shape[1]}")
        print_stat_table(df)
        render_terminal_bar_chart(df)
        render_terminal_scatter_plot(df)

    # Save PNG High-Res Charts
    try:
        # Plot 1: Target distribution
        plt.figure(figsize=(7, 5))
        sns.countplot(x='target', data=df, hue='target', palette=['#2ecc71', '#e74c3c'], legend=False)
        plt.title('Heart Disease Distribution (0: Healthy, 1: Disease Present)')
        plt.xlabel('Diagnosis Status')
        plt.ylabel('Patient Count')
        plt.xticks([0, 1], ['Healthy (0)', 'Disease Present (1)'])
        plt.tight_layout()
        plot1_path = os.path.join(save_dir, '1_target_distribution.png')
        plt.savefig(plot1_path)
        plt.close()
        
        # Plot 2: Age vs Heart Rate Scatter Plot
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='age', y='thalach', hue='target', data=df, palette=['#2ecc71', '#e74c3c'])
        plt.title('Age vs Maximum Heart Rate Distribution')
        plt.xlabel('Age (Years)')
        plt.ylabel('Max Heart Rate (thalach)')
        plt.tight_layout()
        plot2_path = os.path.join(save_dir, '2_age_vs_heart_rate.png')
        plt.savefig(plot2_path)
        plt.close()

        # Plot 3: Feature Correlation Heatmap
        plt.figure(figsize=(9, 7))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plot3_path = os.path.join(save_dir, '3_correlation_heatmap.png')
        plt.savefig(plot3_path)
        plt.close()
        
        if HAS_RICH:
            console.print(Panel(f"[bold green]✓ High-Resolution Plot Images Saved Successfully to folder:[/bold green] [cyan]{save_dir}/[/cyan]\n"
                                f"  • 1_target_distribution.png\n  • 2_age_vs_heart_rate.png\n  • 3_correlation_heatmap.png",
                                title="[bold cyan]Saved Visual Plots[/bold cyan]", box=box.ROUNDED))
        else:
            print(f"[+] All EDA plots saved to {save_dir}/")
    except Exception as e:
        print(f"[!] Plot generation note: {e}")

if __name__ == "__main__":
    if os.path.exists("heart_disease.csv"):
        df = pd.read_csv("heart_disease.csv")
        perform_eda(df)
    else:
        print("[!] heart_disease.csv not found. Run dataset_generator.py first.")
