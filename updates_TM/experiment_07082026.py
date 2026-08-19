from __future__ import annotations

from pathlib import Path
import sys
import subprocess
import json
import csv


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_live_ieee9.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_streaming_run.py"

DETECTOR_DIR = PROJECT_ROOT / "trained_detectors"

RESULT_ROOT = PROJECT_ROOT / "runs_ocsvm_experiments"


RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

REPRESENTATION = "state"

DETECTOR = "ocsvm"

ATTACK_BUS = 4

ATTACK_START = 200
ATTACK_END = 260

# Phase 1 uses this attack strength
BASE_ATTACK_STRENGTH = 0.15

# Run until this many streaming steps
STOP_AFTER_STEPS = 300


# ============================================================
# WINDOW SIZES TO COMPARE
# ============================================================

WINDOW_SIZES = [
    5,
    10,
    15
]


# ============================================================
# ATTACK STRENGTHS FOR BEST WINDOW
# ============================================================

ATTACK_STRENGTHS = [
    0.03,
    0.06,
    0.09,
    0.12,
    0.15,
    0.18,
    0.21
]


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not RUN_SCRIPT.exists():

    raise FileNotFoundError(
        f"\nCannot find run script:\n{RUN_SCRIPT}"
    )


if not EVAL_SCRIPT.exists():

    raise FileNotFoundError(
        f"\nCannot find evaluation script:\n{EVAL_SCRIPT}"
    )


if not DETECTOR_DIR.exists():

    raise FileNotFoundError(
        f"\nCannot find detector directory:\n{DETECTOR_DIR}"
    )


# ============================================================
# FIND METRICS FILE
# ============================================================

def find_metrics_file(run_dir: Path):

    candidates = [

        run_dir / "metrics.json",

        run_dir / "evaluation.json",

        run_dir / "evaluation_metrics.json",

        run_dir / "metrics" / "metrics.json",

    ]


    for path in candidates:

        if path.exists():

            return path


    # Recursive search

    matches = list(
        run_dir.rglob("metrics.json")
    )

    if matches:

        return matches[0]


    matches = list(
        run_dir.rglob("evaluation.json")
    )

    if matches:

        return matches[0]


    matches = list(
        run_dir.rglob("evaluation_metrics.json")
    )

    if matches:

        return matches[0]


    return None


# ============================================================
# EXTRACT METRIC FROM JSON
# ============================================================

def extract_metric(data, names):

    if isinstance(data, dict):

        # Direct keys

        for name in names:

            if name in data:

                value = data[name]

                if isinstance(
                    value,
                    (int, float)
                ):

                    return float(value)


        # Case-insensitive keys

        lowered = {

            str(k).lower(): v

            for k, v in data.items()

        }


        for name in names:

            key = name.lower()

            if key in lowered:

                value = lowered[key]

                if isinstance(
                    value,
                    (int, float)
                ):

                    return float(value)


        # Recursive search

        for value in data.values():

            result = extract_metric(
                value,
                names
            )

            if result is not None:

                return result


    elif isinstance(data, list):

        for value in data:

            result = extract_metric(
                value,
                names
            )

            if result is not None:

                return result


    return None


# ============================================================
# RUN ONE LIVE EXPERIMENT
# ============================================================

def run_live(
    run_dir: Path,
    window_size: int,
    attack_strength: float
):

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print("=" * 70)
    print("RUNNING LIVE EXPERIMENT")
    print("=" * 70)


    print(
        f"Window size     : {window_size}"
    )

    print(
        f"Attack strength : {attack_strength}"
    )

    print(
        f"Attack bus      : {ATTACK_BUS}"
    )

    print(
        f"Attack start    : {ATTACK_START}"
    )

    print(
        f"Attack end      : {ATTACK_END}"
    )

    print(
        f"Stop steps      : {STOP_AFTER_STEPS}"
    )

    print(
        f"Output wrapper  : {run_dir}"
    )


    # ========================================================
    # REQUIRED DETECTOR
    # ========================================================

    detector_file = (

        DETECTOR_DIR

        / f"ocsvm_ieee9_W{window_size}_{REPRESENTATION}.pkl"

    )


    scaler_file = (

        DETECTOR_DIR

        / f"scaler_ieee9_W{window_size}_{REPRESENTATION}.pkl"

    )


    if not detector_file.exists():

        raise FileNotFoundError(

            "\nMissing OCSVM detector:\n"

            f"{detector_file}\n\n"

            "Train this detector before running "
            f"window size {window_size}."

        )


    if not scaler_file.exists():

        raise FileNotFoundError(

            "\nMissing scaler:\n"

            f"{scaler_file}\n\n"

            "Train/create the scaler for "
            f"window size {window_size}."

        )


    # ========================================================
    # COMMAND
    # ========================================================

    cmd = [

        sys.executable,

        str(RUN_SCRIPT),


        "--scenario",
        "stealth",


        "--attack_schedule",
        "fixed",


        "--attack_start",
        str(ATTACK_START),


        "--attack_end",
        str(ATTACK_END),


        "--attack_strength",
        str(attack_strength),


        "--attack_buses",
        str(ATTACK_BUS),


        "--detector_type",
        DETECTOR,


        "--representation",
        REPRESENTATION,


        "--window_size",
        str(window_size),


        "--detector_dir",
        str(DETECTOR_DIR),


        "--enable_mitigation",


        "--enable_control",


        "--control_on_alarm",


        "--enable_recovery",


        "--stop_after_steps",
        str(STOP_AFTER_STEPS),


        "--seed",
        str(SEED),


        "--out_root",
        str(run_dir),

    ]


    print()
    print("COMMAND:")
    print(
        " ".join(
            map(str, cmd)
        )
    )


    # ========================================================
    # EXECUTE LIVE RUN
    # ========================================================

    completed = subprocess.run(

        cmd,

        cwd=PROJECT_ROOT,

        text=True,

        capture_output=True

    )


    # Print normal output

    if completed.stdout:

        print(
            completed.stdout
        )


    # Print errors/warnings

    if completed.stderr:

        print(
            completed.stderr
        )


    # ========================================================
    # CHECK RETURN CODE
    # ========================================================

    if completed.returncode != 0:

        raise RuntimeError(

            "\nLive experiment failed.\n"

            f"Window={window_size}\n"

            f"Attack strength={attack_strength}\n"

            f"Return code={completed.returncode}"

        )


    # ========================================================
    # FIND ACTUAL RUN DIRECTORY
    #
    # run_live_ieee9.py prints:
    #
    # [LIVE DONE] wrote:
    # <actual_run_directory>
    # ========================================================

    actual_run_dir = None


    for line in completed.stdout.splitlines():

        if "[LIVE DONE] wrote:" in line:

            actual_path = line.split(
                "[LIVE DONE] wrote:",
                1
            )[1].strip()


            actual_run_dir = Path(
                actual_path
            )


            break


    # ========================================================
    # FALLBACK:
    # SEARCH FOR run_* DIRECTORY
    # ========================================================

    if actual_run_dir is None:

        print()
        print(
            "Could not extract [LIVE DONE] path."
        )

        print(
            "Searching recursively..."
        )


        candidates = [

            p

            for p in run_dir.rglob("run_*")

            if p.is_dir()

        ]


        if candidates:

            candidates.sort(

                key=lambda p: p.stat().st_mtime,

                reverse=True

            )


            actual_run_dir = candidates[0]


    # ========================================================
    # VALIDATE ACTUAL RUN
    # ========================================================

    if actual_run_dir is None:

        raise RuntimeError(

            "\nLive experiment completed, "
            "but the actual run directory "
            "could not be located.\n"

            f"Expected somewhere under:\n{run_dir}"

        )


    actual_run_dir = actual_run_dir.resolve()


    print()
    print("=" * 70)
    print("ACTUAL RUN DIRECTORY")
    print("=" * 70)

    print(
        actual_run_dir
    )


    # ========================================================
    # CHECK REQUIRED EVALUATION FILE
    # ========================================================

    estimates_file = (

        actual_run_dir

        / "attacked_estimates.jsonl"

    )


    if not estimates_file.exists():

        raise FileNotFoundError(

            "\nLive run completed but "
            "attacked_estimates.jsonl "
            "was not found.\n\n"

            f"Expected:\n{estimates_file}\n\n"

            f"Actual run directory:\n"
            f"{actual_run_dir}"

        )


    print()
    print(
        "Found:"
    )

    print(
        estimates_file
    )


    return actual_run_dir


# ============================================================
# EVALUATE ONE RUN
# ============================================================

def evaluate_run(
    run_dir: Path
):

    print()
    print("=" * 70)
    print("EVALUATING")
    print("=" * 70)

    print(
        run_dir
    )


    # ========================================================
    # CHECK INPUT
    # ========================================================

    estimates_file = (

        run_dir

        / "attacked_estimates.jsonl"

    )


    if not estimates_file.exists():

        raise FileNotFoundError(

            "\nMissing attacked_estimates.jsonl:\n"

            f"{estimates_file}"

        )


    # ========================================================
    # EVALUATION COMMAND
    # ========================================================

    cmd = [

        sys.executable,

        str(EVAL_SCRIPT),

        str(run_dir),

        "--save",

    ]


    print()

    print(
        "COMMAND:"
    )

    print(
        " ".join(
            map(str, cmd)
        )
    )


    completed = subprocess.run(

        cmd,

        cwd=PROJECT_ROOT,

        text=True,

        capture_output=True

    )


    if completed.stdout:

        print(
            completed.stdout
        )


    if completed.stderr:

        print(
            completed.stderr
        )


    # ========================================================
    # CHECK EVALUATION RESULT
    # ========================================================

    if completed.returncode != 0:

        raise RuntimeError(

            "\nEvaluation failed for:\n"

            f"{run_dir}\n\n"

            f"Return code: "
            f"{completed.returncode}"

        )


    # ========================================================
    # FIND METRICS
    # ========================================================

    metrics_file = find_metrics_file(
        run_dir
    )


    if metrics_file is None:

        raise RuntimeError(

            "\nEvaluation succeeded, "
            "but no metrics JSON file "
            "was found.\n\n"

            f"Run directory:\n{run_dir}"

        )


    print()
    print(
        "Metrics file:"
    )

    print(
        metrics_file
    )


    # ========================================================
    # LOAD METRICS
    # ========================================================

    with open(

        metrics_file,

        "r",

        encoding="utf-8"

    ) as f:

        data = json.load(f)


    metrics = {

        "accuracy":
        extract_metric(
            data,
            [
                "accuracy",
                "Accuracy"
            ]
        ),


        "precision":
        extract_metric(
            data,
            [
                "precision",
                "Precision"
            ]
        ),


        "recall":
        extract_metric(
            data,
            [
                "recall",
                "Recall"
            ]
        ),


        "f1":
        extract_metric(
            data,
            [
                "f1",
                "F1",
                "f1_score",
                "F1_score"
            ]
        ),


        "fpr":
        extract_metric(
            data,
            [
                "fpr",
                "FPR",
                "false_positive_rate"
            ]
        ),


        "tp":
        extract_metric(
            data,
            [
                "tp",
                "TP",
                "true_positive"
            ]
        ),


        "fp":
        extract_metric(
            data,
            [
                "fp",
                "FP",
                "false_positive"
            ]
        ),


        "tn":
        extract_metric(
            data,
            [
                "tn",
                "TN",
                "true_negative"
            ]
        ),


        "fn":
        extract_metric(
            data,
            [
                "fn",
                "FN",
                "false_negative"
            ]
        ),

    }


    print()
    print(
        "EXTRACTED METRICS"
    )

    print(
        json.dumps(
            metrics,
            indent=4
        )
    )


    return metrics


# ============================================================
# SAVE RESULTS TO CSV
# ============================================================

def save_csv(
    path: Path,
    rows
):

    if not rows:

        print(
            "No rows to save."
        )

        return


    fields = [

        "window_size",

        "attack_strength",

        "accuracy",

        "precision",

        "recall",

        "f1",

        "fpr",

        "tp",

        "fp",

        "tn",

        "fn",

        "run_dir",

    ]


    with open(

        path,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=fields

        )


        writer.writeheader()


        for row in rows:

            writer.writerow({

                field:
                row.get(field)

                for field in fields

            })


    print()
    print(
        "CSV SAVED:"
    )

    print(
        path
    )


# ============================================================
# PHASE 1
#
# Compare W5, W10, W15
# Attack strength = 0.15
# ============================================================

def phase_1_window_comparison():

    print()
    print("#" * 80)
    print("# PHASE 1: OCSVM WINDOW SIZE COMPARISON")
    print("#" * 80)


    rows = []


    for window_size in WINDOW_SIZES:

        print()
        print(
            "=" * 80
        )

        print(
            f"PHASE 1 EXPERIMENT "
            f"{WINDOW_SIZES.index(window_size) + 1}"
            f"/{len(WINDOW_SIZES)}"
        )

        print(
            f"WINDOW SIZE = {window_size}"
        )

        print(
            "=" * 80
        )


        # ====================================================
        # Wrapper directory
        # ====================================================

        run_wrapper_dir = (

            RESULT_ROOT

            / f"window_{window_size}"

            / f"strength_{BASE_ATTACK_STRENGTH:.2f}"

        )


        # ====================================================
        # Run live experiment
        # ====================================================

        actual_run_dir = run_live(

            run_dir=run_wrapper_dir,

            window_size=window_size,

            attack_strength=BASE_ATTACK_STRENGTH

        )


        # ====================================================
        # Evaluate actual run
        # ====================================================

        metrics = evaluate_run(

            actual_run_dir

        )


        # ====================================================
        # Store row
        # ====================================================

        row = {

            "window_size":
            window_size,


            "attack_strength":
            BASE_ATTACK_STRENGTH,


            "accuracy":
            metrics["accuracy"],


            "precision":
            metrics["precision"],


            "recall":
            metrics["recall"],


            "f1":
            metrics["f1"],


            "fpr":
            metrics["fpr"],


            "tp":
            metrics["tp"],


            "fp":
            metrics["fp"],


            "tn":
            metrics["tn"],


            "fn":
            metrics["fn"],


            "run_dir":
            str(actual_run_dir),

        }


        rows.append(
            row
        )


        # ====================================================
        # Show result immediately
        # ====================================================

        print()
        print(
            "=" * 70
        )

        print(
            f"WINDOW {window_size} RESULT"
        )

        print(
            "=" * 70
        )

        print(
            f"Accuracy  : {metrics['accuracy']}"
        )

        print(
            f"Precision : {metrics['precision']}"
        )

        print(
            f"Recall    : {metrics['recall']}"
        )

        print(
            f"F1        : {metrics['f1']}"
        )

        print(
            f"FPR       : {metrics['fpr']}"
        )

        print(
            f"TP        : {metrics['tp']}"
        )

        print(
            f"FP        : {metrics['fp']}"
        )

        print(
            f"TN        : {metrics['tn']}"
        )

        print(
            f"FN        : {metrics['fn']}"
        )


    # ========================================================
    # Save CSV
    # ========================================================

    csv_path = (

        RESULT_ROOT

        / "ocsvm_window_results.csv"

    )


    save_csv(
        csv_path,
        rows
    )


    return rows


# ============================================================
# SELECT BEST WINDOW USING F1
# ============================================================

def select_best_window(
    rows
):

    valid_rows = [

        row

        for row in rows

        if row["f1"] is not None

    ]


    if not valid_rows:

        raise RuntimeError(

            "\nNo valid F1 scores were found.\n"

            "Cannot select the best window."

        )


    best = max(

        valid_rows,

        key=lambda row:
        row["f1"]

    )


    best_window = int(
        best["window_size"]
    )


    print()
    print("#" * 80)
    print("# BEST WINDOW SELECTED")
    print("#" * 80)


    print()
    print(
        f"Best window size : {best_window}"
    )


    print(
        f"Best F1          : {best['f1']}"
    )


    print(
        f"Precision        : {best['precision']}"
    )


    print(
        f"Recall           : {best['recall']}"
    )


    print(
        f"FPR              : {best['fpr']}"
    )


    print()


    return best_window


# ============================================================
# PHASE 2
#
# Use BEST WINDOW
# Test attack strengths:
# 0.05, 0.10, 0.20
# ============================================================

def phase_2_attack_strength(
    best_window
):

    print()
    print("#" * 80)
    print(
        f"# PHASE 2: ATTACK STRENGTH "
        f"COMPARISON"
    )
    print("#" * 80)


    print()
    print(
        f"Using best window = {best_window}"
    )


    rows = []


    for strength in ATTACK_STRENGTHS:

        print()
        print(
            "=" * 80
        )


        print(
            f"ATTACK STRENGTH = {strength}"
        )


        print(
            "=" * 80
        )


        # ====================================================
        # Wrapper directory
        # ====================================================

        run_wrapper_dir = (

            RESULT_ROOT

            / f"best_window_{best_window}"

            / f"strength_{strength:.2f}"

        )


        # ====================================================
        # Run live
        # ====================================================

        actual_run_dir = run_live(

            run_dir=run_wrapper_dir,

            window_size=best_window,

            attack_strength=strength

        )


        # ====================================================
        # Evaluate
        # ====================================================

        metrics = evaluate_run(

            actual_run_dir

        )


        # ====================================================
        # Save row
        # ====================================================

        row = {

            "window_size":
            best_window,


            "attack_strength":
            strength,


            "accuracy":
            metrics["accuracy"],


            "precision":
            metrics["precision"],


            "recall":
            metrics["recall"],


            "f1":
            metrics["f1"],


            "fpr":
            metrics["fpr"],


            "tp":
            metrics["tp"],


            "fp":
            metrics["fp"],


            "tn":
            metrics["tn"],


            "fn":
            metrics["fn"],


            "run_dir":
            str(actual_run_dir),

        }


        rows.append(
            row
        )


        # ====================================================
        # Print result
        # ====================================================

        print()
        print(
            "=" * 70
        )

        print(
            f"ATTACK STRENGTH {strength} RESULT"
        )

        print(
            "=" * 70
        )

        print(
            f"Accuracy  : {metrics['accuracy']}"
        )

        print(
            f"Precision : {metrics['precision']}"
        )

        print(
            f"Recall    : {metrics['recall']}"
        )

        print(
            f"F1        : {metrics['f1']}"
        )

        print(
            f"FPR       : {metrics['fpr']}"
        )

        print(
            f"TP        : {metrics['tp']}"
        )

        print(
            f"FP        : {metrics['fp']}"
        )

        print(
            f"TN        : {metrics['tn']}"
        )

        print(
            f"FN        : {metrics['fn']}"
        )


    # ========================================================
    # Save Phase 2 CSV
    # ========================================================

    csv_path = (

        RESULT_ROOT

        / "ocsvm_attack_strength_results.csv"

    )


    save_csv(

        csv_path,

        rows

    )


    return rows


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_summary(

    window_rows,

    strength_rows,

    best_window

):

    print()
    print()
    print("=" * 80)
    print("FINAL OCSVM EXPERIMENT SUMMARY")
    print("=" * 80)


    # ========================================================
    # PHASE 1
    # ========================================================

    print()
    print(
        "PHASE 1: WINDOW SIZE COMPARISON"
    )

    print(
        "-" * 80
    )


    print(

        f"{'Window':>10}"
        f"{'Accuracy':>14}"
        f"{'Precision':>14}"
        f"{'Recall':>14}"
        f"{'F1':>14}"
        f"{'FPR':>14}"

    )


    for row in window_rows:

        print(

            f"{row['window_size']:>10}"
            f"{str(row['accuracy']):>14}"
            f"{str(row['precision']):>14}"
            f"{str(row['recall']):>14}"
            f"{str(row['f1']):>14}"
            f"{str(row['fpr']):>14}"

        )


    # ========================================================
    # BEST WINDOW
    # ========================================================

    print()
    print(
        f"BEST WINDOW = {best_window}"
    )


    # ========================================================
    # PHASE 2
    # ========================================================

    print()
    print(
        "PHASE 2: ATTACK STRENGTH COMPARISON"
    )

    print(
        "-" * 80
    )


    print(

        f"{'Strength':>12}"
        f"{'Accuracy':>14}"
        f"{'Precision':>14}"
        f"{'Recall':>14}"
        f"{'F1':>14}"
        f"{'FPR':>14}"

    )


    for row in strength_rows:

        print(

            f"{row['attack_strength']:>12.2f}"
            f"{str(row['accuracy']):>14}"
            f"{str(row['precision']):>14}"
            f"{str(row['recall']):>14}"
            f"{str(row['f1']):>14}"
            f"{str(row['fpr']):>14}"

        )


    # ========================================================
    # BEST ATTACK STRENGTH
    # ========================================================

    valid_strength_rows = [

        row

        for row in strength_rows

        if row["f1"] is not None

    ]


    if valid_strength_rows:

        best_strength_row = max(

            valid_strength_rows,

            key=lambda row:
            row["f1"]

        )


        print()
        print(
            "BEST ATTACK STRENGTH RESULT"
        )

        print(
            "-" * 80
        )


        print(

            f"Window size     : "
            f"{best_strength_row['window_size']}"

        )


        print(

            f"Attack strength : "
            f"{best_strength_row['attack_strength']}"

        )


        print(

            f"F1              : "
            f"{best_strength_row['f1']}"

        )


        print(

            f"Precision       : "
            f"{best_strength_row['precision']}"

        )


        print(

            f"Recall          : "
            f"{best_strength_row['recall']}"

        )


    # ========================================================
    # FILES
    # ========================================================

    print()
    print(
        "=" * 80
    )


    print(
        "RESULT FILES"
    )


    print(
        f"Window results:\n"
        f"{RESULT_ROOT / 'ocsvm_window_results.csv'}"
    )


    print()


    print(
        f"Attack strength results:\n"
        f"{RESULT_ROOT / 'ocsvm_attack_strength_results.csv'}"
    )


    print()
    print(
        f"All experiment runs:\n"
        f"{RESULT_ROOT}"
    )


    print(
        "=" * 80
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("OCSVM STREAMING EXPERIMENT AUTOMATION")
    print("=" * 80)


    print(
        f"Project root: {PROJECT_ROOT}"
    )


    print(
        f"Detector dir: {DETECTOR_DIR}"
    )


    print(
        f"Results dir : {RESULT_ROOT}"
    )


    print()
    print(
        f"Window sizes: {WINDOW_SIZES}"
    )


    print(
        f"Phase 1 attack strength: "
        f"{BASE_ATTACK_STRENGTH}"
    )


    print(
        f"Phase 2 attack strengths: "
        f"{ATTACK_STRENGTHS}"
    )


    print(
        f"Attack bus: {ATTACK_BUS}"
    )


    print(
        f"Attack interval: "
        f"{ATTACK_START} - {ATTACK_END}"
    )


    print(
        f"Stop after steps: "
        f"{STOP_AFTER_STEPS}"
    )


    # ========================================================
    # PHASE 1
    # ========================================================

    window_rows = (
        phase_1_window_comparison()
    )


    # ========================================================
    # SELECT BEST WINDOW
    # ========================================================

    best_window = (
        select_best_window(
            window_rows
        )
    )


    # ========================================================
    # PHASE 2
    # ========================================================

    strength_rows = (
        phase_2_attack_strength(
            best_window
        )
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_summary(

        window_rows,

        strength_rows,

        best_window

    )


    print()
    print(
        "ALL EXPERIMENTS FINISHED."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()