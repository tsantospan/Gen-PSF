#!/bin/bash

# Download dataset from Zenodo



## Example of denoising (Section 5 of the article)

# The PSF to denoise
URL="https://zenodo.org/record/18329242/files/observations_PSF.zip"
DEST="obs.zip"

wget -O "$DEST" "$URL"
unzip "$DEST" -d data/
rm "$DEST"
echo "Observations of PSF downloaded and unpacked"

# The denoised PSF
URL="https://zenodo.org/record/18329242/files/results.zip"
DEST="res.zip"

wget -O "$DEST" "$URL"
unzip "$DEST"
rm "$DEST"
echo "Denoised PSF downloaded and unpacked"



## Training and test sets
URL="https://zenodo.org/record/18336389/files/PSF_training_sets.zip"
DEST="sets.zip"

wget -O "$DEST" "$URL"
unzip "$DEST" -d data/
rm "$DEST"
echo "Datasets of PSF downloaded and unpacked"



## Trained models
URL="https://zenodo.org/record/18417575/files/models.zip"
DEST="models.zip"

wget -O "$DEST" "$URL"
unzip "$DEST"
rm "$DEST"
echo "PSF generators downloaded and unpacked"
