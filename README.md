# Hybrid CNN–ConvFormer Model (Liver Steatosis in Histological Images)

## Description
This project develops a deep learning framework that combines Convolutional Neural Networks (CNNs) with ConvFormer architectures to improve accuracy in detecting liver steatosis from histological images. The approach leverages both local feature extraction and transformer-based global context to enhance medical image analysis.

## Data Source
- [Mendeley Dataset](https://data.mendeley.com/datasets/4mcc9rg4k5/1)

## Process
- Downloaded dataset from Mendeley.
- Wrote a Python script to validate images:
  - Check for valid size and format.
  - Detect duplicates.
  - Assess tissue coverage.
  - Identify fat vacuoles.
- Generated dataset report (valid, invalid, low tissue, no fat detected, duplicates).

## Requirements
See `requirements.txt` for dependencies.
