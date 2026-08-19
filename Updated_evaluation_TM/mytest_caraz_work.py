import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

RUN_SCRIPT = ROOT / "scripts" / "run_live_ieee9.py"


BEST_SEED = 42
WINDOW = 5

representations = [
    "state",
]

detectors = [
    "isolation_forest",
    "lof",
    "ocsvm",
]

scenarios = [
   # "random",
    #"standard",
    "stealth"#,
]


MAX_WORKERS = 1


OUT_ROOT = ROOT / "representation_comparison"
OUT_ROOT.mkdir(parents=True, exist_ok=True)



def run_experiment(cmd):

    print("\n==============================")
    print("Running:")
    print(" ".join(cmd))
    print("==============================")

    subprocess.run(
        cmd,
        cwd=ROOT,
        check=True
    )

    return cmd



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

                    "--out_root",
                    str(out_dir),

                ]


                jobs.append(cmd)



    print("\nTotal experiments:", len(jobs))
    print("Parallel workers:", MAX_WORKERS)



    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:


        futures = [
            executor.submit(
                run_experiment,
                job
            )
            for job in jobs
        ]


        for future in as_completed(futures):

            try:

                result = future.result()

                print(
                    "\nCompleted:",
                    result
                )

            except Exception as e:

                print(
                    "\nFAILED:",
                    e
                )



    print("\n============================")
    print("All experiments finished.")
    print("============================")



if __name__ == "__main__":

    freeze_support()

    main()