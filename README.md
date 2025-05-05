# Capstone Project

**Author:** Gurunishal Saravanan (Student ID: 801430631)
**Date:** May 1, 2025

## Overview

This repository contains the code and report for the Capstone Project on ECG signal classification using both deep learning models and traditional methods. We implement and compare three approaches on the ECG5000 dataset:

* **Method A:** Enhanced 1D CNN (three convolutional layers + global pooling).
* **Method B:** Ensemble of deep neural networks (FCN, ResNet, Encoder, MLP) with majority voting.
* **Method C:** K-Nearest Neighbors (KNN) as a traditional baseline.

## Repository Structure

```
├── data/                   # Processed ECG5000 dataset splits
├── notebooks/              # Jupyter notebooks for experiments
│   └── Capstone_Project.ipynb
├── src/                    # Source code modules
│   ├── data_preprocessing.py
│   ├── model_cnn.py
│   ├── model_ensemble.py
│   ├── model_knn.py
│   └── train.py
├── results/                # Metrics and plots output
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd ecg-classification-capstone
   ```
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\\Scripts\\activate   # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running Experiments

* **Data Preprocessing:**

  ```bash
  python src/data_preprocessing.py --input data/raw/ecg5000.csv --output data/processed
  ```
* **Train Models:**

  ```bash
  python src/train.py --method cnn    # Method A
  python src/train.py --method ensemble  # Method B
  python src/train.py --method knn    # Method C
  ```
* **View Results:**

  * Check `results/metrics.csv` for accuracy and F1-scores.
  * Plots in `results/plots/` show training curves and ensemble comparisons.

## Results Summary

| Method       | Accuracy (%) | F1-Score |
| ------------ | ------------ | -------- |
| CNN (A)      | 93.16        | 0.9263   |
| Ensemble (B) | 90.09        | 0.8780   |
| KNN (C)      | 93.49        | 0.9264   |

> The KNN baseline slightly outperformed the CNN, while the ensemble underperformed due to sensitivity to hyperparameters.

## References

1. Ansari, Y., Mourad, O., Qaraqe, K., & Serpedin, E. (2023). Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017–2023. *Frontiers in Physiology*, 14. DOI: 10.3389/fphys.2023.1246746
2. Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., & Muller, P.-A. (2019). Deep Neural Network Ensembles for Time Series Classification. *IJCNN*, Budapest. DOI: 10.1109/IJCNN.2019.8852018
3. Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., & Muller, P.-A. (2019). InceptionTime: Finding AlexNet for Time Series Classification. *arXiv preprint* arXiv:1909.04939

## Future Work

* Explore lightweight models for real-time deployment on wearable devices.
* Validate on additional ECG datasets to ensure generalization.

---
