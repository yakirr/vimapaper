#!/bin/bash
mkdir -p logs

# for ratio in 0 0.25 0.5 0.75 1
for ratio in 0.013 0.38
do
    echo $ratio
    sbatch --export=RATIO=$ratio,CASEONLY=--caseonly run_sim.sh
done

# for ratio in 1 1.2 1.4 1.6 1.8 2.0 2.4 2.8
# do
#     echo $ratio
#     sbatch --export=RATIO=$ratio,CASEONLY= run_sim.sh
# done
