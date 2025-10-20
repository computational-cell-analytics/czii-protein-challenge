#! /bin/bash
#SBATCH -c 4 #4 #8
#SBATCH --mem 32G #120G #32G #64G #256G
#SBATCH -p grete:shared
#SBATCH -t 1:00:00 #6:00:00 #48:00:00 
#SBATCH -G A100:1 #V100:1 #2 #A100:1  #gtx1080:2 #v100:1 #H100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/detection/slurm/slurm-%j.out 
#SBATCH -A nim00007 #SBATCH --constraint 80gb #SBATCH --constraint=inet

#/scratch/usr/nimsmuth/mambaforge/envs/synapse-net
#/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/synapse-ne
/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/synapse-net/bin/python /user/muth9/u12095/czii-protein-challenge/detection/evaluation/run_evaluation.py \
 -p /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/detections/protein_detection_czii_v12/ \
 -l /scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/ground_truth_scaled_for_detection/ExperimentRuns -m detection_v12_with_adjustement