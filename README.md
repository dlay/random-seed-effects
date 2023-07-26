# Random Seed Effects

___
This is the implementation for our study of the effects of data split random seeds on the accuracy of recommendation algorithms.

## Installation

___
This project was tested with Python 3.9 on Windows, Mac, and Linux.  
You can install the required packages using the `requirements.txt` file.  
You can also run this with singularity. You can build the image using the `scoring.def` file.

## Usage

---
This program has two main entry points.  
One for local execution and one for SLURM execution.  
The execution can be mixed between both entry points, e.g., you can start experiments locally and continue on SLURM.  
Both entries require you to set up an experiment configuration file.  
The file `experiment_full.json` serves as an example configuration.  
Make a copy of this file, configure your experiment, and replace `full` in the file name with your desired
experiment name.
The example configuration is already set up to run the full experiment as presented in our paper.
Note that you may omit configuration options if they are not required for your experiment, e.g. omit all SLURM options,
if you never run the experiment on SLURM.

The list below details all the configuration options inside the configuration file:

| Option                         | Description                                                                       |
|--------------------------------|-----------------------------------------------------------------------------------|
| `REPRODUCIBILITY_MODE`         | Whether the random seeds for the project should be generated or taken from a file |
| `DATA_SET_NAMES`               | Comma-separated list of data sets.                                                |
| `PRUNE_TECHNIQUES`             | The techniques to prune the data sets with.                                       |
| `SPLIT_TECHNIQUES`             | The techniques to split the data sets with.                                       |
| `NUM_FOLDS`                    | The number of folds for cross-validation.                                         |
| `RECOMMENDERS`                 | Comma-separated list of recommenders.                                             |
| `RECOMMENDER_SEEDING`          | The seeding techniques for the recommender fitting process.                       |
| `NUM_BATCHES`                  | Number of user batches that are predicted for. Increases parallelization.         |
| `TOPN_SCORE`                   | A list of cutoff values to evaluate for                                           |
| `JOB_FAIL_EMAIL`               | For SLURM: email to notify on job fail.                                           |
| `STAGE0_CLEANING_TIME`         | For SLURM: Time for data cleaning jobs.                                           |
| `STAGE0_CLEANING_MEMORY`       | For SLURM: Memory for data cleaning jobs.                                         |
| `STAGE0_CLEANING_CORES`        | For SLURM: Number of CPU cores for data cleaning jobs.                            |
| `STAGE1_PRUNING_TIME`          | For SLURM: Time for data pruning jobs.                                            |
| `STAGE1_PRUNING_MEMORY`        | For SLURM: Memory for data pruning jobs.                                          |
| `STAGE1_PRUNING_CORES`         | For SLURM: Number of CPU cores for data pruning jobs.                             |
| `STAGE2_SPLITTING_TIME`        | For SLURM: Time for data splitting jobs.                                          |
| `STAGE2_SPLITTING_MEMORY`      | For SLURM: Memory for data splitting jobs.                                        |
| `STAGE2_SPLITTING_CORES`       | For SLURM: Number of CPU cores for data splitting jobs.                           |
| `STAGE3_FITTING_TIME`          | For SLURM: Time for recommender fitting jobs.                                     |
| `STAGE3_FITTING_MEMORY`        | For SLURM: Memory for recommender fitting jobs.                                   |
| `STAGE3_FITTING_CORES`         | For SLURM: Number of CPU cores for recommender fitting jobs.                      |
| `STAGE4_PREDICTING_TIME`       | For SLURM: Time for recommender predicting jobs.                                  |
| `STAGE4_PREDICTING_MEMORY`     | For SLURM: Memory for recommender predicting jobs.                                |
| `STAGE4_PREDICTING_CORES`      | For SLURM: Number of CPU cores for recommender predicting jobs.                   |
| `STAGE5_EVALUATING_TIME`       | For SLURM: Time for evaluating predictions jobs.                                  |
| `STAGE5_EVALUATING_MEMORY`     | For SLURM: Memory for evaluating predictions jobs.                                |
| `STAGE5_EVALUATING_CORES`      | For SLURM: Number of CPU cores for evaluating predictions jobs.                   |

### Supported Data Sets

This framework natively supports nine data sets.  
Instructions to download the data sets can be found in the `data` folder.

### IMPORTANT: Execution order

Note that full execution of experiments is a five-stage process with two additional stages to plot results.  
The execution has to happen in sequential order, e.g., stage 2 cannot be executed before stage 1.  
The stages are as follows:
<ol start="0">
    <li>Data cleaning. The data is read from the original file(s), cleaned of duplicates, and saved in a homogeneous format.</li>
    <li>Data pruning. The data is pruned according to a pruning technique.</li>
    <li>Data splitting. The data is split according to a splitting technique. A random seed is generated in or passed to this function.</li>
    <li>Recommender fitting. The recommender is fitted. A random seed is generated in or passed to this function.</li>
    <li>Recommender predicting. The fitted recommender is used to predict ranked lists of recommendations.</li>
    <li>Evaluating predictions. Given a cutoff, the predictions are evaluated with ranking metrics.</li>
    <li>Reporting (local execution only). The evaluations are aggregated into a single report file.</li>
    <li>Plotting (local execution only). the report file is used to generate plots and print statistics.</li>
</ol>

### Execution Option 1: SLURM execution

SLURM execution requires Singularity and the required image.  
To schedule jobs with SLURM, run `hpc_executor.py` with commands `experiment` and `stage`.  
Example: `python hpc_executor.py --experiment full --stage 0`.

### Execution Option 2: Local execution

Local execution requires a Python environment with the required packages.
The entry point is `local_executor.py`.  
The configuration is controlled via `select_experiment.py`.  
Open `select_experiment.py`, make and save changes, then run `local_executor.py`.  
Example: `python local_executor.py`.

### Pre-Plotted results

We make a collection of plots from to the experiments available.
They are found in the folder `plots`.
They are generated with the report file `evaluation_report.pkl` that contains all results from our experiments.

### Reproducibility

Our experiments can be fully reproduced using the random seed file in this repository.
The random seeds used to split the data and fit recommenders are found in the `project_seeds.txt` file.
Running the experiments with our template file `experiment_full.json` that is in reproducibility mode by default, will use the seed file to reproduce our results.

## Notes

---
Running locally is advised only for small tests.  
It can take *extremely* long to run full experiments locally depending on your experiment configuration.
The execution heavily relies on massively parallel computing resources.  
Running on an HPC cluster with SLURM is strongly advised for full experiments.