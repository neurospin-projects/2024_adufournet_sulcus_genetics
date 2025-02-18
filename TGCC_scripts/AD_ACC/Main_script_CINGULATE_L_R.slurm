#!/bin/bash
#MSUB -r CINGULATE_L_R                                                                               # Name of the job submission
#MSUB -n 1
#MSUB -c 1
#MSUB -T 86400  #600 # 86400 #172800 #36000 #7200 #43200 
#MSUB -o /ccc/workflash/cont003/n4h00001/n4h00001/25irene_AD_CINGULATE_val/logs/slurm/Brain_%I.o    # Standard output. %I is the job id
#MSUB -e /ccc/workflash/cont003/n4h00001/n4h00001/25irene_AD_CINGULATE_val/logs/slurm/Brain_%I.e    # Error output. %I is the job id
#MSUB -q rome                     									  # Queue
#MSUB -Q normal                                                                                           # Priority normal/ long
#MSUB -m workflash,scratch                                                                                # Available filesystems
#MSUB -A gen13800

echo "${HOSTNAME}"

#  Envionments basic softwares
#
module switch dfldatadir/n4h00001
module load ig
module load snakemake/6.15.5

snakemake --version


# Root dir
ROOT=/ccc/workflash/cont003/n4h00001/n4h00001/25irene_AD_CINGULATE_val # Be careful here for the root dir !

# Snakemake files (smk, yml and json).
SMK=${ROOT}/scripts/Pipeline_CINGULATE_L_R.smk                # Be careful here !
TSKCFG=${ROOT}/scripts/Path_tskcfg_CINGULATE_L_R.yml    	# Be careful here !
CLUCFG=${ROOT}/scripts/Cluster_params.json	# Be careful here !

# logs entries
# LOGS=/ccc/scratch/cont003/n4h00001/dufourna/logs                # Be careful here ! just change it related to tempodirlog
LOGS=${ROOT}/logs
LOGS_CLU=${LOGS}/sm/1                                           # Be careful here !
# Worker number limit
JOBS=150

#
cat <<-____HH
Processing project
     SM files:          $SMK
     Task specific:     $TSKCFG
     Cluster specifics: $CLUCFG

     See logs in      : ${LOGS}
     Running on       : ${JOBS} jobs
____HH

# emplacement des logs des jobs slurm/bridge tel que defini 
# dans le fichier json $CLUCFG
mkdir -p ${LOGS_CLU}



snakemake \
  --jobs ${JOBS} \
  -s ${SMK} \
  --configfile ${TSKCFG} \
  --cluster-config ${CLUCFG} \
  --verbose --printshellcmds --reason \
  --cluster ${SM_SLURM_SUBM_NODEP} \
  --cluster-status ${SM_SLURM_STATUS}  \
  --latency-wait 10

# end
echo "Job gwas multi-task finished"
