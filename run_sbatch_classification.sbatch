#! /bin/bash
#SBATCH -c 4 #4 #8
#SBATCH --mem 64G #120G #32G #64G #256G
#SBATCH -p grete:shared #grete-h100:shared
#SBATCH -t 3:00:00 #6:00:00 #48:00:00 
#SBATCH -G A100:1 #V100:1 #2 #A100:1  #gtx1080:2 #v100:1 #H100:1
#SBATCH --output=/user/muth9/u12095/czii-protein-challenge/classification/slurm/slurm-%j.out 
#SBATCH -A nim00007 #SBATCH --constraint 80gb 


/scratch/usr/nimsmuth/mambaforge/envs/synapse-net/bin/python /user/muth9/u12095/czii-protein-challenge/classification/inference/run_protein_classification_with_predicted_detections.py \
 -p /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/detections/3d-adapted-model_faket/public_test_dataset/17682_protein_detections.json \
 -l /scratch-grete/projects/nim00007/cryo-et/challenge-data/public_test_dataset/ground_truth/17723/Picks \
 -m /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/models/checkpoints/protein_classification_czii_v16/ \
 -v /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/data/tomograms_VoxelSpacing10/17723/ \
 --max_extent 39 \
 -sub_o /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/classification/protein_classification_czii_v16/coord_from_detection_adapted_model_faket/subtomograms \
 -o /mnt/lustre-grete/usr/u12095/cryo-et/czii_challenge/classification/protein_classification_czii_v16/coord_from_detection_adapted_model_faket 