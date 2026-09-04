#! /bin/bash
#SBATCH -c 4 #4 #8
#SBATCH --mem 256G #120G #32G #64G #256G
#SBATCH -p grete:shared
#SBATCH -t 10:00:00 #6:00:00 #48:00:00 
#SBATCH -G A100:1 #V100:1 #2 #A100:1  #gtx1080:2 #v100:1 #H100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/detection/slurm/slurm-%j.out 
#SBATCH -A nim00007 #SBATCH --constraint 80gb #SBATCH --constraint=inet

#/scratch/usr/nimsmuth/mambaforge/envs/synapse-net
#/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/synapse-ne
#/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python
/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/pro-revelio/bin/python /user/muth9/u12095/czii-protein-challenge/detection_scripts/run_protein_detection.py \
 -i /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/tomograms_VoxelSpacing10/ \
 -o /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/detections/protein_detection_da_v41_ExperimentRuns/public_test_dataset \
 -m /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_detection_da_v41_ExperimentRuns \
 -j /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/training/protein_detection_czii_v41/split-ExperimentRuns_faket_dens1_5_distr_eqCl3.json