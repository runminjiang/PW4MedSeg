#!/bin/bash

# PW4MedSeg Complete Preprocessing Pipeline
# This script processes all organs for both BTCV and CHAOS datasets

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Configuration
POINTS=200
DELTA=10
THRESHOLD=0.5

# Create log directory
LOG_DIR="./preprocessing_logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/preprocessing_${TIMESTAMP}.log"

echo -e "${GREEN}Starting preprocessing pipeline at $(date)${NC}" | tee -a $LOG_FILE
echo -e "${GREEN}Log file: $LOG_FILE${NC}"

# Function to process one organ
process_organ() {
    local dataset=$1
    local organ=$2
    
    echo -e "\n${YELLOW}========================================${NC}" | tee -a $LOG_FILE
    echo -e "${YELLOW}Processing $dataset - $organ${NC}" | tee -a $LOG_FILE
    echo -e "${YELLOW}========================================${NC}" | tee -a $LOG_FILE
    
    # Step 2: Multi-label to single-label conversion
    echo -e "\n${GREEN}Step 2: Converting multi-label to single-label for $organ...${NC}" | tee -a $LOG_FILE
    python Data_preprocessing/multi2one.py --dataset $dataset --organ $organ 2>&1 | tee -a $LOG_FILE
    
    # Step 3: Generate pseudo-labels
    echo -e "\n${GREEN}Step 3: Generating pseudo-labels (points=$POINTS, delta=$DELTA)...${NC}" | tee -a $LOG_FILE
    python Data_preprocessing/point2gau.py --dataset $dataset --organ $organ --points $POINTS --delta $DELTA 2>&1 | tee -a $LOG_FILE
    
    # Step 4: Apply threshold
    echo -e "\n${GREEN}Step 4: Applying threshold ($THRESHOLD)...${NC}" | tee -a $LOG_FILE
    python Data_preprocessing/label_thr.py --dataset $dataset --organ $organ --points $POINTS --delta $DELTA --threshold $THRESHOLD 2>&1 | tee -a $LOG_FILE
    
    echo -e "${GREEN}Completed $dataset - $organ${NC}" | tee -a $LOG_FILE
}

# Process BTCV dataset
echo -e "\n${YELLOW}========================================${NC}" | tee -a $LOG_FILE
echo -e "${YELLOW}PROCESSING BTCV DATASET${NC}" | tee -a $LOG_FILE
echo -e "${YELLOW}========================================${NC}" | tee -a $LOG_FILE

# Step 1: Anisotropic to isotropic conversion for BTCV
echo -e "\n${GREEN}Step 1: Converting BTCV images to isotropic...${NC}" | tee -a $LOG_FILE
python Data_preprocessing/anisotropic2isotropic.py --dataset btcv 2>&1 | tee -a $LOG_FILE

# Process each BTCV organ
for organ in spleen liver right_kidney left_kidney; do
    process_organ btcv $organ
done

# Process CHAOS dataset
echo -e "\n${YELLOW}========================================${NC}" | tee -a $LOG_FILE
echo -e "${YELLOW}PROCESSING CHAOS DATASET${NC}" | tee -a $LOG_FILE
echo -e "${YELLOW}========================================${NC}" | tee -a $LOG_FILE

# Step 1: Anisotropic to isotropic conversion for CHAOS
echo -e "\n${GREEN}Step 1: Converting CHAOS images to isotropic...${NC}" | tee -a $LOG_FILE
python Data_preprocessing/anisotropic2isotropic.py --dataset chaos 2>&1 | tee -a $LOG_FILE

# Process each CHAOS organ
for organ in liver spleen right_kidney left_kidney; do
    process_organ chaos $organ
done

# Process MSCMRSEG dataset
echo -e "\n${YELLOW}========================================${NC}" | tee -a $LOG_FILE
echo -e "${YELLOW}PROCESSING MSCMRSEG DATASET${NC}" | tee -a $LOG_FILE
echo -e "${YELLOW}========================================${NC}" | tee -a $LOG_FILE

# Note: MSCMRSEG is already isotropic, so we skip Step 1
echo -e "\n${GREEN}Note: MSCMRSEG is already isotropic (1mm³), skipping Step 1${NC}" | tee -a $LOG_FILE

# Process each MSCMRSEG organ (heart structures)
for organ in lv_cavity lv_myocardium rv_cavity; do
    process_organ mscmrseg $organ
done

# Summary
echo -e "\n${GREEN}========================================${NC}" | tee -a $LOG_FILE
echo -e "${GREEN}PREPROCESSING COMPLETED at $(date)${NC}" | tee -a $LOG_FILE
echo -e "${GREEN}========================================${NC}" | tee -a $LOG_FILE

# Count generated files
echo -e "\n${GREEN}File count summary:${NC}" | tee -a $LOG_FILE
for dataset in btcv chaos mscmrseg; do
    echo -e "\n${YELLOW}$dataset:${NC}" | tee -a $LOG_FILE
    
    if [ -d "dataset/$dataset/image_iso" ]; then
        echo "  Isotropic images: $(ls dataset/$dataset/image_iso/*.nii.gz 2>/dev/null | wc -l)" | tee -a $LOG_FILE
    fi
    
    if [ -d "dataset/$dataset/label_iso" ]; then
        echo "  Isotropic labels: $(ls dataset/$dataset/label_iso/*.nii.gz 2>/dev/null | wc -l)" | tee -a $LOG_FILE
    fi
    
    # Determine organs based on dataset
    if [ "$dataset" = "mscmrseg" ]; then
        organs="lv_cavity lv_myocardium rv_cavity"
    else
        organs="spleen liver right_kidney left_kidney"
    fi
    
    for organ in $organs; do
        if [ -d "dataset/$dataset/$organ" ]; then
            echo -e "  ${organ}:" | tee -a $LOG_FILE
            
            if [ -d "dataset/$dataset/$organ/labelsTr_gt" ]; then
                echo "    Ground truth: $(ls dataset/$dataset/$organ/labelsTr_gt/*.nii.gz 2>/dev/null | wc -l)" | tee -a $LOG_FILE
            fi
            
            if [ -d "dataset/$dataset/$organ/p_${POINTS}_d_${DELTA}/point_label" ]; then
                echo "    Point labels: $(ls dataset/$dataset/$organ/p_${POINTS}_d_${DELTA}/point_label/*.nii.gz 2>/dev/null | wc -l)" | tee -a $LOG_FILE
            fi
            
            if [ -d "dataset/$dataset/$organ/p_${POINTS}_d_${DELTA}/labelsTr" ]; then
                echo "    Gaussian labels: $(ls dataset/$dataset/$organ/p_${POINTS}_d_${DELTA}/labelsTr/*.nii.gz 2>/dev/null | wc -l)" | tee -a $LOG_FILE
            fi
            
            if [ -d "dataset/$dataset/$organ/p_${POINTS}_d_${DELTA}/labelsTr_thres" ]; then
                echo "    Thresholded labels: $(ls dataset/$dataset/$organ/p_${POINTS}_d_${DELTA}/labelsTr_thres/*.nii.gz 2>/dev/null | wc -l)" | tee -a $LOG_FILE
            fi
        fi
    done
done

echo -e "\n${GREEN}Log saved to: $LOG_FILE${NC}"
echo -e "${GREEN}Done!${NC}"