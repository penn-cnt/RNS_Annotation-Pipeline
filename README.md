# RNS Active Learning Annotation Pipeline

[![DOI](https://img.shields.io/badge/DOI-10.1088%2F1741--2552%2Fade402-blue)](https://doi.org/10.1088/1741-2552/ade402)
[![Journal](https://img.shields.io/badge/Journal-J.%20Neural%20Eng.-green)](https://iopscience.iop.org/journal/1741-2552)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the official implementation for the paper:

**[Annotating Neurophysiologic Data at Scale with Optimized Human Input](https://doi.org/10.1088/1741-2552/ade402)**

*Journal of Neural Engineering*, Volume 22, Article 046003, 2025

## Authors

Zhongchuan Xu, Brittany H. Scheid, Erin C. Conrad, Kathryn A. Davis, Taneeta Ganguly, Michael A. Gelfand, James J. Gugger, Xiangyu Jiang, Joshua J. LaRocque, William K. S. Ojemann, Saurabh R. Sinha, Genna J. Waldman, Joost Wagenaar, Nishant Sinha, and Brian Litt

## Overview

This repository provides an active learning framework for efficiently annotating neurophysiologic data from Responsive Neurostimulation (RNS) devices. The pipeline combines self-supervised learning (SwAV) with various active learning query strategies to minimize annotation effort while maximizing model performance.

![Pipeline Overview](figure/fig2_full_bs_v2.svg)

### Key Features

- **Self-Supervised Pre-training**: SwAV-based representation learning on unlabeled RNS data
- **Multiple Active Learning Strategies**: Comprehensive implementation of 15+ query strategies
- **Scalable Annotation Pipeline**: Designed to reduce expert annotation burden by identifying the most informative samples

## Repository Structure

```
RNS_Annotation-Pipeline/
├── figure/                          # Paper figures and supplementary materials
├── scripts/
│   └── RNS_LITT_ANNOTATION_PIPELINE/
│       ├── rns_scripts/             # Main RNS active learning scripts
│       │   ├── models/              # Model architectures
│       │   │   ├── SwaV.py          # SwAV self-supervised model
│       │   │   ├── LSTMDownStream.py
│       │   │   ├── SupervisedDownstream.py
│       │   │   ├── WAAL_net.py
│       │   │   └── rns_dataloader.py
│       │   ├── rns_active_learning_LSTM.py
│       │   ├── rns_active_learning_lpl.py
│       │   └── rns_active_learning_waal.py
│       ├── kaggle_dog_scripts/      # Kaggle seizure detection experiments
│       └── tools/                   # Shared utilities
│           ├── query_strategies/    # Active learning query strategies
│           │   ├── random_sampling.py
│           │   ├── entropy_sampling.py
│           │   ├── margin_sampling.py
│           │   ├── least_confidence.py
│           │   ├── badge_sampling.py
│           │   ├── kcenter_greedy.py
│           │   ├── kmeans_sampling.py
│           │   ├── loss_prediction.py
│           │   ├── waal.py
│           │   ├── vaal.py
│           │   └── ...
│           ├── active_learning_data.py
│           ├── active_learning_net.py
│           └── active_learning_utility.py
├── user_data/                       # Private data directory (gitignored)
├── requirements.txt
└── README.md
```

## Implemented Query Strategies

| Strategy | Description |
|----------|-------------|
| **Random Sampling** | Baseline random selection |
| **Entropy Sampling** | Select samples with highest prediction entropy |
| **Margin Sampling** | Select samples with smallest margin between top predictions |
| **Least Confidence** | Select samples with lowest prediction confidence |
| **BADGE** | Batch Active learning by Diverse Gradient Embeddings |
| **K-Center Greedy** | Core-set selection using greedy k-center algorithm |
| **K-Means Sampling** | Cluster-based sampling using K-Means |
| **BALD** | Bayesian Active Learning by Disagreement |
| **Loss Prediction** | Learning Loss for Active Learning |
| **WAAL** | Wasserstein Adversarial Active Learning |
| **VAAL** | Variational Adversarial Active Learning |
| **Adversarial BIM/DeepFool** | Adversarial perturbation-based selection |

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/RNS_Annotation-Pipeline.git
cd RNS_Annotation-Pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

```
numpy~=1.23.1
tqdm~=4.64.0
matplotlib~=3.5.2
pandas~=1.4.3
nltk~=3.7
scipy~=1.8.1
scikit-learn
lightly~=1.3.1
h5py~=3.7.0
pytorch-lightning
torch
torchvision
```

## Usage

### 1. Data Preparation

Place your RNS data in the `user_data/` directory. The expected format is HDF5 or NumPy arrays with the appropriate structure.

### 2. Self-Supervised Pre-training (SwAV)

Train the SwAV model on unlabeled data:

```bash
cd scripts/RNS_LITT_ANNOTATION_PIPELINE/rns_scripts
python -c "from RNS_Swav_train import *"
```

Or use the Jupyter notebook:
```
scripts/RNS_LITT_ANNOTATION_PIPELINE/rns_scripts/RNS_Swav_train.ipynb
```

### 3. Active Learning

Run active learning with your chosen strategy:

```bash
cd scripts/RNS_LITT_ANNOTATION_PIPELINE/rns_scripts

# Using LSTM classifier
python rns_active_learning_LSTM.py

# Using Loss Prediction Learning
python rns_active_learning_lpl.py

# Using WAAL
python rns_active_learning_waal.py
```

### 4. Configuration

Key parameters in the active learning scripts:

```python
nStart = 1       # Initial labeled pool (% of total)
nEnd = 20        # Final labeled pool (% of total)
nQuery = 2       # Samples to query per round (% of total)
strategy_name = 'EntropySampling'  # Query strategy to use
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{Xu_2025_JNE,
  author = {Xu, Zhongchuan and Scheid, Brittany H. and Conrad, Erin C. and Davis, Kathryn A. and Ganguly, Taneeta and Gelfand, Michael A. and Gugger, James J. and Jiang, Xiangyu and LaRocque, Joshua J. and Ojemann, William K. S. and Sinha, Saurabh R. and Waldman, Genna J. and Wagenaar, Joost and Sinha, Nishant and Litt, Brian},
  title = {Annotating neurophysiologic data at scale with optimized human input},
  journal = {Journal of Neural Engineering},
  volume = {22},
  number = {4},
  pages = {046003},
  year = {2025},
  doi = {10.1088/1741-2552/ade402}
}
```

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Contact

For questions or collaboration inquiries, please contact:

- **Zhongchuan Xu** - Primary Author
- **Brian Litt** - Principal Investigator

Center for Neuroengineering and Therapeutics (CNT)  
University of Pennsylvania

