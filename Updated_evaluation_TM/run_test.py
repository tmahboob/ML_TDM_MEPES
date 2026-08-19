from pathlib import Path
import subprocess
import sys


# ==================================================
# Project paths
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

RUN_SCRIPT = ROOT / "scripts" / "run_live_ieee9.py"

if not RUN_SCRIPT.exists():
    raise FileNotFoundError(
        f"Could not find:\n{RUN_SCRIPT}"
    )


# ==================================================
# Experiment configuration
# ==================================================

BEST_SEED = 42
WINDOW = 5

representations = [
    "state"
]

detectors = [
    "isolation_forest",
    "lof",
    "ocsvm"
]

scenarios = [
  #  "random",
  #  "standard",
    "stealth"
]


# ==================================================
# Output directory
# ==================================================

OUT_ROOT = ROOT / "representation_comparison"

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# Run one experiment
# ==================================================

def run_experiment(cmd):

    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80)


    subprocess.run(
        cmd,
        cwd=ROOT,
        check=True
    )


    print("\nFinished:")
    print(cmd)


# ==================================================
# Main experiment loop
# ==================================================

def main():

    jobs = []


    for representation in representations:

        for detector in detectors:

            for scenario in scenarios:


                out_dir = (
                    OUT_ROOT
                    / representation
                    / detector
                    / scenario
                )


                out_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )


                cmd = [

                    sys.executable,

                    str(RUN_SCRIPT),


                    "--representation",
                    representation,


                    "--scenario",
                    scenario,


                    "--detector_type",
                    detector,


                    "--window_size",
                    str(WINDOW),


                    "--seed",
                    str(BEST_SEED),


                    "--stop_after_steps",
                    "200",


                    "--out_root",
                    str(out_dir)

                ]


                jobs.append(cmd)



    print("\nTotal experiments:", len(jobs))


    # ==============================================
    # Sequential execution
    # ==============================================

    for i, job in enumerate(jobs, start=1):

        print(
            f"\n\nExperiment {i}/{len(jobs)}"
        )

        try:

            run_experiment(job)


        except subprocess.CalledProcessError as e:

            print(
                "\nFAILED EXPERIMENT:"
            )

            print(e)



    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 80)



if __name__ == "__main__":

    main()