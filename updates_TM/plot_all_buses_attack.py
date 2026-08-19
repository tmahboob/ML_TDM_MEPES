import json
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_ROOT / "bus_attack_results"
PLOTS_DIR = RESULTS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

BUSES = list(range(1, 10))

WINDOW_SIZE = 5
ATTACK_STRENGTH = 0.15
ATTACK_START = 200
ATTACK_END = 260
TOTAL_TIMESTEPS = 260
ATTACK_TIMESTEPS = 60
CLEAN_TIMESTEPS = 200

DETECTOR = "ocsvm"
REPRESENTATION = "state"


# ============================================================
# HELPER
# ============================================================

def safe_float(value):
    """
    Convert a value to float.
    Return NaN if conversion is impossible.
    """
    try:
        if value is None:
            return float("nan")

        value = float(value)

        if math.isnan(value):
            return float("nan")

        return value

    except (TypeError, ValueError):
        return float("nan")


def find_latest_run(bus_dir):
    """
    Find the newest run directory for a bus.
    """

    if not bus_dir.exists():
        return None

    run_dirs = [
        p for p in bus_dir.rglob("run_*")
        if p.is_dir()
    ]

    if not run_dirs:
        return None

    run_dirs.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return run_dirs[0]


def find_evaluation_json(run_dir):
    """
    Locate evaluation_metrics.json.
    """

    if run_dir is None:
        return None

    # Primary expected location
    direct_file = run_dir / "evaluation_metrics.json"

    if direct_file.exists():
        return direct_file

    # Fallback: search recursively
    matches = list(run_dir.rglob("evaluation_metrics.json"))

    if matches:
        return matches[0]

    # Also support metrics.json if present
    direct_file = run_dir / "metrics.json"

    if direct_file.exists():
        return direct_file

    matches = list(run_dir.rglob("metrics.json"))

    if matches:
        return matches[0]

    return None


def load_metrics(json_file):
    """
    Read metrics from evaluation_metrics.json.

    Supports several possible JSON structures.
    """

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --------------------------------------------------------
    # Case 1:
    # {
    #   "accuracy": ...,
    #   "precision": ...,
    #   ...
    # }
    # --------------------------------------------------------

    if isinstance(data, dict):

        # Some evaluation files may put everything directly
        keys_lower = {
            str(k).lower(): k
            for k in data.keys()
        }

        def get_key(name):
            if name in data:
                return data[name]

            if name.lower() in keys_lower:
                return data[keys_lower[name.lower()]]

            return None

        accuracy = get_key("accuracy")
        precision = get_key("precision")
        recall = get_key("recall")
        f1 = get_key("f1")
        fpr = get_key("fpr")

        tp = get_key("TP")
        fn = get_key("FN")
        fp = get_key("FP")
        tn = get_key("TN")

        total_timesteps = get_key("total_timesteps")
        attack_timesteps = get_key("attack_timesteps")
        clean_timesteps = get_key("clean_timesteps")
        total_alarms = get_key("total_alarms")

        # ----------------------------------------------------
        # Case 2: nested metrics dictionary
        # ----------------------------------------------------

        if accuracy is None and "metrics" in data:
            nested = data["metrics"]

            if isinstance(nested, dict):

                nested_lower = {
                    str(k).lower(): k
                    for k in nested.keys()
                }

                def nested_get(name):
                    if name in nested:
                        return nested[name]

                    if name.lower() in nested_lower:
                        return nested[nested_lower[name.lower()]]

                    return None

                accuracy = nested_get("accuracy")
                precision = nested_get("precision")
                recall = nested_get("recall")
                f1 = nested_get("f1")
                fpr = nested_get("fpr")

                tp = nested_get("TP")
                fn = nested_get("FN")
                fp = nested_get("FP")
                tn = nested_get("TN")

                total_timesteps = nested_get("total_timesteps")
                attack_timesteps = nested_get("attack_timesteps")
                clean_timesteps = nested_get("clean_timesteps")
                total_alarms = nested_get("total_alarms")

        # ----------------------------------------------------
        # Case 3: calculate metrics from confusion matrix
        # ----------------------------------------------------

        tp = safe_float(tp)
        fn = safe_float(fn)
        fp = safe_float(fp)
        tn = safe_float(tn)

        if math.isnan(safe_float(accuracy)):
            denominator = tp + tn + fp + fn

            if denominator > 0:
                accuracy = (tp + tn) / denominator

        if math.isnan(safe_float(precision)):
            denominator = tp + fp

            if denominator > 0:
                precision = tp / denominator

        if math.isnan(safe_float(recall)):
            denominator = tp + fn

            if denominator > 0:
                recall = tp / denominator

        if math.isnan(safe_float(f1)):
            p = safe_float(precision)
            r = safe_float(recall)

            if not math.isnan(p) and not math.isnan(r):
                if p + r > 0:
                    f1 = 2 * p * r / (p + r)

        if math.isnan(safe_float(fpr)):
            denominator = fp + tn

            if denominator > 0:
                fpr = fp / denominator

        return {
            "Total_Timesteps": total_timesteps,
            "Attack_Timesteps": attack_timesteps,
            "Clean_Timesteps": clean_timesteps,
            "Total_Alarms": total_alarms,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "Accuracy": safe_float(accuracy),
            "Precision": safe_float(precision),
            "Recall": safe_float(recall),
            "F1": safe_float(f1),
            "FPR": safe_float(fpr),
        }

    raise ValueError(
        f"Unsupported JSON structure: {json_file}"
    )


# ============================================================
# EXTRACT RESULTS
# ============================================================

print("=" * 80)
print("EXTRACTING IEEE-9 BUS ATTACK RESULTS")
print("=" * 80)

rows = []

for bus in BUSES:

    print()
    print("-" * 80)
    print(f"BUS {bus}")
    print("-" * 80)

    bus_dir = RESULTS_DIR / f"bus_{bus}"

    run_dir = find_latest_run(bus_dir)

    if run_dir is None:
        print("No run directory found.")
        continue

    print(f"Run directory:")
    print(run_dir)

    evaluation_file = find_evaluation_json(run_dir)

    if evaluation_file is None:
        print("evaluation_metrics.json NOT FOUND")
        continue

    print(f"Evaluation file:")
    print(evaluation_file)

    try:
        metrics = load_metrics(evaluation_file)

    except Exception as e:
        print(f"ERROR reading metrics: {e}")
        continue

    row = {
        "Bus": bus,
        "Window": WINDOW_SIZE,
        "Attack_Strength": ATTACK_STRENGTH,
        "Attack_Start": ATTACK_START,
        "Attack_End": ATTACK_END,

        "Total_Timesteps": (
            metrics["Total_Timesteps"]
            if metrics["Total_Timesteps"] is not None
            else TOTAL_TIMESTEPS
        ),

        "Attack_Timesteps": (
            metrics["Attack_Timesteps"]
            if metrics["Attack_Timesteps"] is not None
            else ATTACK_TIMESTEPS
        ),

        "Clean_Timesteps": (
            metrics["Clean_Timesteps"]
            if metrics["Clean_Timesteps"] is not None
            else CLEAN_TIMESTEPS
        ),

        "Total_Alarms": metrics["Total_Alarms"],

        "TP": metrics["TP"],
        "FN": metrics["FN"],
        "FP": metrics["FP"],
        "TN": metrics["TN"],

        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "FPR": metrics["FPR"],

        "Run_Directory": str(run_dir),
    }

    rows.append(row)

    print()
    print(f"Bus {bus} metrics:")
    print(f"  TP        : {row['TP']}")
    print(f"  FN        : {row['FN']}")
    print(f"  FP        : {row['FP']}")
    print(f"  TN        : {row['TN']}")
    print(f"  Accuracy  : {row['Accuracy']:.4f}")
    print(f"  Precision : {row['Precision']:.4f}")
    print(f"  Recall    : {row['Recall']:.4f}")
    print(f"  F1        : {row['F1']:.4f}")
    print(f"  FPR       : {row['FPR']:.4f}")


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError(
        "No evaluation results were found."
    )


# Sort by bus

df = df.sort_values("Bus").reset_index(drop=True)


# ============================================================
# SAVE CSV
# ============================================================

csv_file = RESULTS_DIR / "bus_attack_results.csv"

df.to_csv(
    csv_file,
    index=False,
    float_format="%.6f"
)

print()
print("=" * 80)
print("RESULTS CSV SAVED")
print("=" * 80)
print(csv_file)


# ============================================================
# PRINT TABLE
# ============================================================

display_columns = [
    "Bus",
    "Window",
    "Attack_Strength",
    "Total_Timesteps",
    "Attack_Timesteps",
    "Clean_Timesteps",
    "TP",
    "FN",
    "FP",
    "TN",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "FPR",
]

print()
print(df[display_columns].to_string(index=False))


# ============================================================
# PLOT 1: ALL METRICS
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "FPR",
]

plt.figure(figsize=(12, 7))

for metric in metrics:

    values = pd.to_numeric(
        df[metric],
        errors="coerce"
    )

    plt.plot(
        df["Bus"],
        values,
        marker="o",
        linewidth=2,
        label=metric
    )

plt.xlabel("Attack Bus")
plt.ylabel("Metric Value")
plt.title(
    "IEEE-9 Bus Attack Detection Performance"
)

plt.xticks(BUSES)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

png_file = PLOTS_DIR / "all_metrics_by_attack_bus.png"
pdf_file = PLOTS_DIR / "all_metrics_by_attack_bus.pdf"

plt.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf_file,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 2: CONFUSION MATRIX COMPONENTS
# ============================================================

plt.figure(figsize=(12, 7))

for metric in ["TP", "FN", "FP", "TN"]:

    values = pd.to_numeric(
        df[metric],
        errors="coerce"
    )

    plt.plot(
        df["Bus"],
        values,
        marker="o",
        linewidth=2,
        label=metric
    )

plt.xlabel("Attack Bus")
plt.ylabel("Number of Timesteps")
plt.title(
    "IEEE-9 Bus Attack Confusion-Matrix Components"
)

plt.xticks(BUSES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

png_cm = PLOTS_DIR / "confusion_components_by_bus.png"
pdf_cm = PLOTS_DIR / "confusion_components_by_bus.pdf"

plt.savefig(
    png_cm,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf_cm,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 3: F1 BY BUS
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df["Bus"],
    df["F1"],
)

plt.xlabel("Attack Bus")
plt.ylabel("F1 Score")
plt.title(
    "F1 Score by Attack Bus — IEEE-9"
)

plt.xticks(BUSES)
plt.ylim(0, 1.05)
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

f1_png = PLOTS_DIR / "f1_by_attack_bus.png"
f1_pdf = PLOTS_DIR / "f1_by_attack_bus.pdf"

plt.savefig(
    f1_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    f1_pdf,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 4: RECALL BY BUS
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df["Bus"],
    df["Recall"],
)

plt.xlabel("Attack Bus")
plt.ylabel("Recall")
plt.title(
    "Detection Recall by Attack Bus — IEEE-9"
)

plt.xticks(BUSES)
plt.ylim(0, 1.05)
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

recall_png = PLOTS_DIR / "recall_by_attack_bus.png"
recall_pdf = PLOTS_DIR / "recall_by_attack_bus.pdf"

plt.savefig(
    recall_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    recall_pdf,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# BEST / WORST BUS
# ============================================================

valid_f1 = df.dropna(subset=["F1"])

if not valid_f1.empty:

    best_row = valid_f1.loc[
        valid_f1["F1"].idxmax()
    ]

    worst_row = valid_f1.loc[
        valid_f1["F1"].idxmin()
    ]

    print()
    print("=" * 80)
    print("BUS PERFORMANCE SUMMARY")
    print("=" * 80)

    print(
        f"Best bus by F1  : Bus {int(best_row['Bus'])} "
        f"(F1 = {best_row['F1']:.4f})"
    )

    print(
        f"Worst bus by F1 : Bus {int(worst_row['Bus'])} "
        f"(F1 = {worst_row['F1']:.4f})"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("EXPERIMENT DATA EXTRACTION COMPLETE")
print("=" * 80)

print()
print("CSV:")
print(csv_file)

print()
print("PLOTS:")
print(PLOTS_DIR)

print()
print("Generated files:")

for file in sorted(PLOTS_DIR.iterdir()):

    if file.is_file():
        print(f"  {file.name}")

print()
print("=" * 80)
