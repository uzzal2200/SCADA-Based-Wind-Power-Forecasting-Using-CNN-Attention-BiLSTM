"""
wtpf — Wind Turbine Power Forecasting
======================================

A modular, reproducible implementation of the CNN-Attention-BiLSTM hybrid
deep learning architecture for short-term wind turbine power forecasting
from multivariate SCADA data, as described in:

    Mia, M.U., Debnath, S., Biswas, A.K., Hosain, M.S., & Shimamura, T.
    "A Novel CNN-Attention-BiLSTM Hybrid Deep Learning Model for Wind
    Turbine Power Forecasting Using SCADA Data." IEEE Access, 2026.

Subpackages
-----------
data           : loading, cleaning, feature engineering, sequence/Dataset
models         : CNN, LSTM, BiLSTM, CNN-BiLSTM, Attention-BiLSTM,
                 CNN-Attention-BiLSTM (proposed) and ablation variants
training       : Trainer, early stopping, LR scheduling
evaluation     : metrics, statistical tests, seasonal & power-band analysis
visualization  : EDA and results plotting utilities
utils          : config loading, reproducibility, logging
"""

__version__ = "1.0.0"
__author__ = "Md. Uzzal Mia, Sajib Debnath, Arindam Kishor Biswas, Md. Sarwar Hosain, Tetsuya Shimamura"
