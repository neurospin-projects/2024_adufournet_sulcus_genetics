Scripts Folder Overview
=============

This directory contains the necessary scripts and configuration files for running the pipeline. Below is a description of each file and its purpose.

File Structure
-------------------

```
scripts/
├── Cluster_params.json
├── Cycle.txt
├── Main_script.slurm
├── Path_tskcfg.yml
├── Pipeline.smk
├── README.txt
```

File Descriptions
-------------------

1. `Path_tskcfg.yml`
--------------------------
- **Purpose**: Defines paths for input directories, output directories, log directories, and temporary directories.
- **Instructions**: 
  - Edit this file to specify where you want to load your files.
  - If you want to use the parallel model, then, you have to write down in models (the list of models) the relative path (from a directory data) that lead to the infile. All the infiles must have the same name.
  - Besides, if you don't want to calculate the genomic information because you always work on the same cohort, then use the "tempodirgen" path, and comment the GWA rules. You also have to comment or uncomment the lines that were linked to the relative paths.
  - Create the directories according to your plan before running the code.

2. `Main_script.slurm`
--------------------------
- **Purpose**: The script used to submit your job to the cluster.
- **Instructions**:
  - Check and modify the following:
    - Job name
    - Standard and error output paths
    - `root_dir` definition
    - Paths to `.smk`, `.yml`, and `.json` files
    - Log entries
  - Ensure that the directories are correctly structured and exist.

3. `Cycle.txt`
--------------------------
- **Purpose**: Contains paths leading to the Snakemake `Pipeline.smk` file.
- **Instructions**:
  - Copy and paste the first three lines of `Cycle.txt` into your terminal to define the variables for a dry run.
  - Run the dry run to identify any missing files and to see which rules will be applied along with their commands.
  - If no errors occur, submit the jobs using the `ccc_msub` command.

4. `Pipeline.smk`
--------------------------
- **Purpose**: The main Snakemake pipeline file.
- **Instructions**:
  - No changes are needed if your files are in the correct format and located in the correct directories as specified in `Path_tskcfg.yml`.

5. `Cluster_params.json`
--------------------------
- **Purpose**: Defines the number of threads, memory, and time required for each rule in the `Pipeline.smk` file.
- **Instructions**:
  - Adjust these parameters as necessary to match your resource requirements for each pipeline rule.

Steps to Follow
-------------------

1. **Set Up Directories**:
   - Edit the `Path_tskcfg.yml` file to specify paths for inputs, outputs, logs, and temporary directories.
   - Create these directories according to your plan.

2. **Prepare Main Script**:
   - Open `Main_script.slurm` and verify or modify the job name, output paths, `root_dir`, and paths to the `.smk`, `.yml`, and `.json` files.
   - Ensure all mentioned directories exist and are correctly structured.

3. **Configure Cycle**:
   - Copy the first three lines of `Cycle.txt` into your terminal to set necessary variables.
   - Perform a dry run to check for missing files and review the rules and commands that will be applied.
   - If the dry run is successful and free of errors, submit the job using the `ccc_msub` command.
