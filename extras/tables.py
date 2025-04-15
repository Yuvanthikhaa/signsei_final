import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set global font settings
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12

# Tables data
tables = {
    "accuracy_comparison.png": {
        "title": "Accuracy Comparison (Our Model vs. Existing Models)",
        "columns": ["Model Name", "Accuracy (%)"],
        "data": [
            ["CNN-Based Model", 92.35],
            ["LSTM-Based Model", 93.12],
            ["MediaPipe + CNN Model", 94.87],
            ["SignSei (Proposed Model)", 96.58],
        ]
    },
    "precision_comparison.png": {
        "title": "Precision Comparison",
        "columns": ["Model Name", "Precision (%)"],
        "data": [
            ["CNN-Based Model", 91.78],
            ["LSTM-Based Model", 92.65],
            ["MediaPipe + CNN Model", 94.23],
            ["SignSei (Proposed Model)", 96.21],
        ]
    },
    "recall_comparison.png": {
        "title": "Recall Comparison",
        "columns": ["Model Name", "Recall (%)"],
        "data": [
            ["CNN-Based Model", 91.50],
            ["LSTM-Based Model", 92.34],
            ["MediaPipe + CNN Model", 94.56],
            ["SignSei (Proposed Model)", 96.47],
        ]
    },
    "f1score_comparison.png": {
        "title": "F1-Score Comparison",
        "columns": ["Model Name", "F1-Score (%)"],
        "data": [
            ["CNN-Based Model", 91.64],
            ["LSTM-Based Model", 92.49],
            ["MediaPipe + CNN Model", 94.39],
            ["SignSei (Proposed Model)", 96.34],
        ]
    },
    "overall_comparison.png": {
        "title": "Overall Evaluation",
        "columns": ["Model Name", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
        "data": [
            ["CNN-Based Model", 92.35, 91.78, 91.50, 91.64],
            ["LSTM-Based Model", 93.12, 92.65, 92.34, 92.49],
            ["MediaPipe + CNN Model", 94.87, 94.23, 94.56, 94.39],
            ["SignSei (Proposed Model)", 96.58, 96.21, 96.47, 96.34],
        ]
    }
}

def save_table(filename, title, columns, data):
    df = pd.DataFrame(data, columns=columns)

    fig, ax = plt.subplots(figsize=(6, len(data) * 0.8 + 1))  # Auto-adjust height
    ax.axis('off')

    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Header bold
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
        cell.set_text_props(ha='center', va='center')

        if row == 0:
            cell.set_text_props(weight='bold')

    plt.title(title, pad=20)
    plt.tight_layout()

    # Save with white background
    plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

# Generate all tables
for filename, table_info in tables.items():
    save_table(filename, table_info["title"], table_info["columns"], table_info["data"])

print("✅ Tables generated successfully as images!")
