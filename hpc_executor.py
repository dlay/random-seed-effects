import argparse
import json
import subprocess
from pathlib import Path
from static import *


def execute_clean_data(data_set_names, job_time, job_memory, job_cores, fail_email):
    for data_set_name in data_set_names:
        base_path = f"./{DATA_FOLDER}/{data_set_name}/{CLEAN_FOLDER}/{CLEAN_FILE}"
        if not Path(base_path).exists():
            script_name = f"_RSE_stage0_clean_{data_set_name}"
            script = "#!/bin/bash\n" \
                     "#SBATCH --nodes=1\n" \
                     f"#SBATCH --cpus-per-task={job_cores}\n" \
                     "#SBATCH --mail-type=FAIL\n" \
                     f"#SBATCH --mail-user={fail_email}\n" \
                     "#SBATCH --partition=short,medium,long\n" \
                     f"#SBATCH --time={job_time}\n" \
                     f"#SBATCH --mem={job_memory}\n" \
                     "#SBATCH --output=./omni_out/%x_%j.out\n" \
                     "module load singularity\n" \
                     "singularity exec --pwd /mnt --bind ./:/mnt ./rse.sif python -u " \
                     "./clean_data.py " \
                     f"--data_set_name {data_set_name}"
            with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                f.write(script)
            subprocess.run(["sbatch", f"./{script_name}.sh"])
            Path(f"./{script_name}.sh").unlink()


def execute_prune_data(data_set_names, prune_techniques, job_time, job_memory, job_cores, fail_email):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            base_path = f"./{DATA_FOLDER}/{data_set_name}/{PRUNE_FOLDER}/{prune_technique}_{PRUNE_FILE}"
            if not Path(base_path).exists():
                script_name = f"_RSE_stage1_prune_{data_set_name}_{prune_technique}"
                script = "#!/bin/bash\n" \
                         "#SBATCH --nodes=1\n" \
                         f"#SBATCH --cpus-per-task={job_cores}\n" \
                         "#SBATCH --mail-type=FAIL\n" \
                         f"#SBATCH --mail-user={fail_email}\n" \
                         "#SBATCH --partition=short,medium,long\n" \
                         f"#SBATCH --time={job_time}\n" \
                         f"#SBATCH --mem={job_memory}\n" \
                         "#SBATCH --output=./omni_out/%x_%j.out\n" \
                         "module load singularity\n" \
                         "singularity exec --pwd /mnt --bind ./:/mnt ./rse.sif python -u " \
                         "./prune_data.py " \
                         f"--data_set_name {data_set_name} " \
                         f"--prune_technique {prune_technique}"
                with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                    f.write(script)
                subprocess.run(["sbatch", f"./{script_name}.sh"])
                Path(f"./{script_name}.sh").unlink()


def execute_generate_splits(data_set_names, prune_techniques, split_techniques, num_folds, job_time, job_memory,
                            job_cores, fail_email, reproducibility_mode):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                def run_script(reproducibility_seed):
                    script_name = f"_RSE_stage2_split_{data_set_name}_{prune_technique}_{split_technique}_{num_folds}"
                    script = "#!/bin/bash\n" \
                             "#SBATCH --nodes=1\n" \
                             f"#SBATCH --cpus-per-task={job_cores}\n" \
                             "#SBATCH --mail-type=FAIL\n" \
                             f"#SBATCH --mail-user={fail_email}\n" \
                             "#SBATCH --partition=short,medium,long\n" \
                             f"#SBATCH --time={job_time}\n" \
                             f"#SBATCH --mem={job_memory}\n" \
                             "#SBATCH --output=./omni_out/%x_%j.out\n" \
                             "module load singularity\n" \
                             "singularity exec --pwd /mnt --bind ./:/mnt ./rse.sif python -u " \
                             "./generate_splits.py " \
                             f"--data_set_name {data_set_name} " \
                             f"--prune_technique {prune_technique} " \
                             f"--split_technique {split_technique} " \
                             f"--num_folds {num_folds} " \
                             f"--reproducibility_seed {reproducibility_seed}"
                    with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                        f.write(script)
                    subprocess.run(["sbatch", f"./{script_name}.sh"])
                    Path(f"./{script_name}.sh").unlink()

                if bool(reproducibility_mode):
                    seeds = json.loads(open(f"project_seeds.txt", "r").read())
                    for shuffle_seed in list(seeds[data_set_name].keys()):
                        run_script(shuffle_seed)
                else:
                    run_script(-1)


def execute_fit_recommender(data_set_names, prune_techniques, split_techniques, num_folds, recommenders,
                            recommender_seeding, job_time, job_memory, job_cores, fail_email, reproducibility_mode):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                shuffle_seeds = []
                for file in Path(f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}").iterdir():
                    _, file_seed, file_prune_technique, file_split_technique, _ = file.name.split(".")[0].split("_")
                    if file_prune_technique == prune_technique and file_split_technique == split_technique:
                        shuffle_seeds.append(file_seed)
                shuffle_seeds = list(set(shuffle_seeds))
                for recommender in recommenders:
                    for shuffle_seed in shuffle_seeds:
                        for recommender_seed in recommender_seeding:
                            for test_fold in range(num_folds):
                                def run_script(reproducibility_seed):
                                    base_path = f"./{DATA_FOLDER}/{data_set_name}/{RECOMMENDER_FOLDER}_{recommender}/" \
                                                f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_" \
                                                f"{recommender_seed}_{RECOMMENDER_FILE}"
                                    if not Path(base_path).exists() or Path(base_path).stat().st_size == 0:
                                        script_name = f"_RSE_stage3_fit_{data_set_name}_{prune_technique}_" \
                                                      f"{split_technique}_{shuffle_seed}_{num_folds}_{recommender}_" \
                                                      f"{recommender_seed}_{test_fold}"
                                        script = "#!/bin/bash\n" \
                                                 "#SBATCH --nodes=1\n" \
                                                 f"#SBATCH --cpus-per-task={job_cores}\n" \
                                                 "#SBATCH --mail-type=FAIL\n" \
                                                 f"#SBATCH --mail-user={fail_email}\n" \
                                                 "#SBATCH --partition=short,medium,long\n" \
                                                 f"#SBATCH --time={job_time}\n" \
                                                 f"#SBATCH --mem={job_memory}\n" \
                                                 "#SBATCH --output=./omni_out/%x_%j.out\n" \
                                                 "module load singularity\n" \
                                                 "singularity exec --pwd /mnt --bind ./:/mnt ./rse.sif python -u " \
                                                 "./fit_recommender.py " \
                                                 f"--data_set_name {data_set_name} " \
                                                 f"--prune_technique {prune_technique} " \
                                                 f"--split_technique {split_technique} " \
                                                 f"--num_folds {num_folds} " \
                                                 f"--test_fold {test_fold} " \
                                                 f"--shuffle_seed {shuffle_seed} " \
                                                 f"--recommender {recommender} " \
                                                 f"--recommender_seed {recommender_seed} " \
                                                 f"--reproducibility_seed {reproducibility_seed}"
                                        with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                                            f.write(script)
                                        subprocess.run(["sbatch", f"./{script_name}.sh"])
                                        Path(f"./{script_name}.sh").unlink()

                                if bool(reproducibility_mode):
                                    seeds = json.loads(open(f"project_seeds.txt", "r").read())
                                    run_script(seeds[data_set_name][shuffle_seed][recommender][str(test_fold)])
                                else:
                                    run_script(-1)


def execute_make_predictions(data_set_names, prune_techniques, split_techniques, num_folds, recommenders,
                             recommender_seeding, num_batches, job_time, job_memory, job_cores, fail_email):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                shuffle_seeds = []
                for file in Path(f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}").iterdir():
                    _, file_seed, file_prune_technique, file_split_technique, _ = file.name.split(".")[0].split("_")
                    if file_prune_technique == prune_technique and file_split_technique == split_technique:
                        shuffle_seeds.append(file_seed)
                shuffle_seeds = list(set(shuffle_seeds))
                for recommender in recommenders:
                    for shuffle_seed in shuffle_seeds:
                        for recommender_seed in recommender_seeding:
                            for test_fold in range(num_folds):
                                for run_batch in range(num_batches):
                                    base_path = f"./{DATA_FOLDER}/{data_set_name}/{PREDICTION_FOLDER}_{recommender}/" \
                                                f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_" \
                                                f"{recommender_seed}_{num_batches}_{run_batch}_{PREDICTION_FILE}"
                                    if not Path(base_path).exists() or Path(base_path).stat().st_size == 0:
                                        script_name = f"_RSE_stage4_predict_{data_set_name}_{prune_technique}_" \
                                                      f"{split_technique}_{shuffle_seed}_{num_folds}_{recommender}_" \
                                                      f"{recommender_seed}_{test_fold}_{num_batches}_{run_batch}"
                                        script = "#!/bin/bash\n" \
                                                 "#SBATCH --nodes=1\n" \
                                                 f"#SBATCH --cpus-per-task={job_cores}\n" \
                                                 "#SBATCH --mail-type=FAIL\n" \
                                                 f"#SBATCH --mail-user={fail_email}\n" \
                                                 "#SBATCH --partition=short,medium,long\n" \
                                                 f"#SBATCH --time={job_time}\n" \
                                                 f"#SBATCH --mem={job_memory}\n" \
                                                 "#SBATCH --output=./omni_out/%x_%j.out\n" \
                                                 "module load singularity\n" \
                                                 "singularity exec --pwd /mnt --bind ./:/mnt ./rse.sif python -u " \
                                                 "./make_predictions.py " \
                                                 f"--data_set_name {data_set_name} " \
                                                 f"--prune_technique {prune_technique} " \
                                                 f"--split_technique {split_technique} " \
                                                 f"--test_fold {test_fold} " \
                                                 f"--shuffle_seed {shuffle_seed} " \
                                                 f"--recommender {recommender} " \
                                                 f"--recommender_seed {recommender_seed} " \
                                                 f"--num_batches {num_batches} " \
                                                 f"--run_batch {run_batch}"
                                        with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                                            f.write(script)
                                        subprocess.run(["sbatch", f"./{script_name}.sh"])
                                        Path(f"./{script_name}.sh").unlink()


def execute_evaluate_predictions(data_set_names, prune_techniques, split_techniques, num_folds, recommenders,
                                 recommender_seeding, num_batches, topn_scores, job_time, job_memory, job_cores,
                                 fail_email):
    topn_scores_string = '-'.join([str(x) for x in topn_scores])
    topn_scores_script = ' '.join([str(x) for x in topn_scores])
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                shuffle_seeds = []
                for file in Path(f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}").iterdir():
                    _, file_seed, file_prune_technique, file_split_technique, _ = file.name.split(".")[0].split("_")
                    if file_prune_technique == prune_technique and file_split_technique == split_technique:
                        shuffle_seeds.append(file_seed)
                shuffle_seeds = list(set(shuffle_seeds))
                for recommender in recommenders:
                    for shuffle_seed in shuffle_seeds:
                        for recommender_seed in recommender_seeding:
                            for test_fold in range(num_folds):
                                base_path = f"./{DATA_FOLDER}/{data_set_name}/{EVALUATION_FOLDER}_" \
                                            f"{recommender}/{test_fold}_{shuffle_seed}_{prune_technique}_" \
                                            f"{split_technique}_{recommender_seed}_{num_batches}_" \
                                            f"{topn_scores_string}_{EVALUATION_FILE}"
                                if not Path(base_path).exists() or Path(base_path).stat().st_size == 0:
                                    script_name = \
                                        f"_RSE_stage5_evaluate_{data_set_name}_{prune_technique}_" \
                                        f"{split_technique}_{shuffle_seed}_{num_folds}_{recommender}_" \
                                        f"{recommender_seed}_{test_fold}_{num_batches}_{topn_scores_string}"
                                    script = "#!/bin/bash\n" \
                                             "#SBATCH --nodes=1\n" \
                                             f"#SBATCH --cpus-per-task={job_cores}\n" \
                                             "#SBATCH --mail-type=FAIL\n" \
                                             f"#SBATCH --mail-user={fail_email}\n" \
                                             "#SBATCH --partition=short,medium,long\n" \
                                             f"#SBATCH --time={job_time}\n" \
                                             f"#SBATCH --mem={job_memory}\n" \
                                             "#SBATCH --output=./omni_out/%x_%j.out\n" \
                                             "module load singularity\n" \
                                             "singularity exec --pwd /mnt --bind ./:/mnt ./rse.sif python -u " \
                                             "./evaluate_predictions.py " \
                                             f"--data_set_name {data_set_name} " \
                                             f"--prune_technique {prune_technique} " \
                                             f"--split_technique {split_technique} " \
                                             f"--test_fold {test_fold} " \
                                             f"--shuffle_seed {shuffle_seed} " \
                                             f"--recommender {recommender} " \
                                             f"--recommender_seed {recommender_seed} " \
                                             f"--num_batches {num_batches} " \
                                             f"--topn_scores {topn_scores_script}"
                                    with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                                        f.write(script)
                                    subprocess.run(["sbatch", f"./{script_name}.sh"])
                                    Path(f"./{script_name}.sh").unlink()


parser = argparse.ArgumentParser("HPC Executor Script for Random Seed Effects!")
parser.add_argument('--experiment', dest='experiment', type=str, default="template")
parser.add_argument('--stage', dest='stage', type=int, default=-1)
args = parser.parse_args()

experiment_settings = json.load(open(f"./experiment_{args.experiment}.json"))
if args.stage == 0:
    execute_clean_data(experiment_settings["DATA_SET_NAMES"], experiment_settings["STAGE0_CLEANING_TIME"],
                       experiment_settings["STAGE0_CLEANING_MEMORY"], experiment_settings["STAGE0_CLEANING_CORES"],
                       experiment_settings["JOB_FAIL_EMAIL"])
elif args.stage == 1:
    execute_prune_data(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                       experiment_settings["STAGE1_PRUNING_TIME"], experiment_settings["STAGE1_PRUNING_MEMORY"],
                       experiment_settings["STAGE1_PRUNING_CORES"], experiment_settings["JOB_FAIL_EMAIL"])
elif args.stage == 2:
    execute_generate_splits(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                            experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"],
                            experiment_settings["STAGE2_SPLITTING_TIME"],
                            experiment_settings["STAGE2_SPLITTING_MEMORY"],
                            experiment_settings["STAGE2_SPLITTING_CORES"], experiment_settings["JOB_FAIL_EMAIL"],
                            experiment_settings["REPRODUCIBILITY_MODE"])
elif args.stage == 3:
    execute_fit_recommender(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                            experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"],
                            experiment_settings["RECOMMENDERS"], experiment_settings["RECOMMENDER_SEEDING"],
                            experiment_settings["STAGE3_FITTING_TIME"], experiment_settings["STAGE3_FITTING_MEMORY"],
                            experiment_settings["STAGE3_FITTING_CORES"], experiment_settings["JOB_FAIL_EMAIL"],
                            experiment_settings["REPRODUCIBILITY_MODE"])
elif args.stage == 4:
    execute_make_predictions(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                             experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"],
                             experiment_settings["RECOMMENDERS"], experiment_settings["RECOMMENDER_SEEDING"],
                             experiment_settings["NUM_BATCHES"], experiment_settings["STAGE4_PREDICTING_TIME"],
                             experiment_settings["STAGE4_PREDICTING_MEMORY"],
                             experiment_settings["STAGE4_PREDICTING_CORES"], experiment_settings["JOB_FAIL_EMAIL"])
elif args.stage == 5:
    execute_evaluate_predictions(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                                 experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"],
                                 experiment_settings["RECOMMENDERS"], experiment_settings["RECOMMENDER_SEEDING"],
                                 experiment_settings["NUM_BATCHES"], experiment_settings["TOPN_SCORES"],
                                 experiment_settings["STAGE5_EVALUATING_TIME"],
                                 experiment_settings["STAGE5_EVALUATING_MEMORY"],
                                 experiment_settings["STAGE5_EVALUATING_CORES"], experiment_settings["JOB_FAIL_EMAIL"])
else:
    print("No valid stage selected!")
