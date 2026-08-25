# Stacking Ensemble Forecaster for VNF Performance Metrics

**Multi-Step Time-Series Forecasting of CPU, Memory, Processing Latency, and Traffic Load in Virtual Network Functions.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Overview

This repository contains the official implementation of a robust **Stacking Ensemble framework** for forecasting key performance metrics of Virtual Network Functions (VNFs) within a 5G Service Function Chain (SFC). The framework forecast **four critical performance metrics**:

- ⚙️ **CPU Usage**
- 🧠 **Memory Usage**
- ⏱️ **Processing Latency**
- 📡 **Traffic Load**

By synergistically combining Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), and Transformers, the meta-learner effectively captures both **short-term fluctuations** and **long-term trends** in network resource consumption.

This work is published in the proceedings of IEEE CNSM 2025.

---

## 📊 Key Results

Our proposed ensemble framework significantly outperforms state-of-the-art single-model approaches:

- **75% reduction** in Mean Absolute Error (MAE) compared to the standalone Transformer model.
- **>84% reduction** in MAE compared to RNN, LSTM, and GRU models.
- Robust **multi-step forecasting** across 2, 15, 30, and 60-second horizons.

---

## 🧠 Methodology

The framework operates via a two-layer hierarchical architecture:

1. **Base Models (Layer 1)**:
   - Four deep learning architectures (RNN, LSTM, GRU, Transformer) are trained independently on the normalized time-series data.
   - Each model captures different aspects of the data (e.g., RNN for sequential dependencies, Transformers for global context).
2. **Stacking Ensemble (Layer 2)**:
   - **K-Fold Cross-Validation** generates Out-Of-Fold (OOF) predictions from the base models to prevent overfitting.
   - **Inverse-MAE Weighting** assigns higher importance to models with lower forecasting errors.
   - A final **Meta-Learner** (Ridge Regression / Linear Regressor) synthesizes the weighted base predictions into the ultimate accurate forecast.


---

## 📝 Citation

If you find this repository useful for your research or work, please cite our original paper:

```bibtex
@inproceedings{rafiee2025proactive,
  title={A Proactive Performance Prediction Framework for Virtual Network Functions in 5G Networks},
  author={Rafiee, Milad and Ocampo, Andres F and Taherkordi, Amir and Alay, Özgü},
  booktitle={2025 21st International Conference on Network and Service Management (CNSM)},
  pages={1--7},
  year={2025},
  organization={IEEE},
  address={Bologna, Italy}
}
```



