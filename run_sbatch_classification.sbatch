#! /bin/bash
#SBATCH -c 4 #4 #8
#SBATCH --mem 120G #120G #32G #64G #256G
#SBATCH -p grete:shared #grete-h100:shared
#SBATCH -t 3:00:00 #6:00:00 #48:00:00 
#SBATCH -G A100:1 #V100:1 #2 #A100:1  #gtx1080:2 #v100:1 #H100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out 
#SBATCH -A nim00007 #SBATCH --constraint 80gb #SBATCH --constraint=inet

#/scratch/usr/nimsmuth/mambaforge/envs/synapse-net
#/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/synapse-net
/mnt/vast-nhr/home/muth9/u12095/micromamba/envs/synapse-net/bin/python /user/muth9/u12095/czii-protein-challenge/classification/inference/run_protein_classification_with_predicted_detections.py \
 -i /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/tomograms_VoxelSpacing10 \
 -l /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/ground_truth \
 -o /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/classification/protein_classification_czii_v22/public_test_dataset_predictions_by_v13 \
 -m /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_classification_czii_v22/ \
 --max_extent 39 -sub_o /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/classification/protein_classification_czii_v22/public_test_dataset_predictions_by_v13/subtomograms \
 -p /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/detections/protein_detection_czii_v13/public_test_dataset 
 
 
 