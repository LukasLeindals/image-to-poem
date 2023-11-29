#!/bin/sh

#  assert correct run dir
run_dir="image-to-poem"
if ! [ "$(basename $PWD)" = $run_dir ];
then
    echo -e "\033[0;31mScript must be submitted from the directory: $run_dir\033[0m"
    exit 1
fi

# create dir for logs
mkdir -p "logs/hpc"

#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Similarity
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/hpc/%J.out
#BSUB -e logs/hpc/%J.err
# -- end of LSF options --


# activate env
source ../nlp_venv/bin/activate

# load additional modules
module load cuda/11.8

# run scripts
python image_to_poem/similarity/similarity_scoring.py