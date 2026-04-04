# ArcLight-AttGRU
Official implementation for **ArcLight-AttGRU: An Ultralightweight Attention-Gated GRU Framework for Low-Frequency Series Arc Fault Detection** (IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, 2026)

## Overview
ArcLight-AttGRU is a dedicated ultralightweight deep learning model for **low-frequency (60Hz) series arc fault (SAF) detection** in residential and industrial power systems. The model addresses the key pain points of high hardware cost, poor scalability and low deployability of traditional arc fault detection methods, and achieves high-precision detection based on standard power grid sampling rate signals, with excellent performance in parameter efficiency, inference speed and embedded deployment compatibility.

Due to confidentiality reasons, only part of the dataset can be made public.

## Core Model Design
ArcLight-AttGRU integrates lightweight convolution, recurrent neural network and attention mechanism, and combines structured pruning strategy to balance detection accuracy and model lightweight:
- **SeparableConv1D Block**: Two stacked depthwise separable convolution layers with batch normalization, efficiently extract local disturbance features of low-frequency current time series at low computational cost
- **Dual-layer GRU**: Gated Recurrent Unit with return sequence enabled, capture global temporal dependencies of arc fault signals
- **Feature-wise Attention Mechanism**: Adaptive reweighting of GRU output features, focus on key time steps and discriminative features of arc fault occurrence
- **Structured Pruning**: Polynomial decay-based magnitude pruning with 50% final sparsity, reduces redundant parameters by 48.28% while maintaining detection accuracy, significantly lowering memory footprint and inference latency

## Key Performance
Evaluated on the ArcSafe-60 low-frequency arc fault dataset (60Hz sampling), the model achieves state-of-the-art detection performance:
- Pre-pruning binary classification accuracy: 99.98%
- Post-pruning binary classification accuracy: 99.86% (only 0.12% accuracy drop)
- Ultra-low computational cost: 0.93 MFLOPs (batch size=1)
- Embedded deployment friendly: Sub-1ms inference latency, ~20mW power consumption on ARM Cortex-M4
- Strong generalization: 98.95% average accuracy across single-branch/multi-branch load scenarios

## Requirements
```bash
tensorflow>=2.8
tensorflow-model-optimization
numpy
pandas
```

## Citation
If you use this code or the ArcLight-AttGRU model in your research, please cite our paper:
```bibtex
@article{guo2026arclight,
  title={ArcLight-AttGRU: An Ultralightweight Attention-Gated GRU Framework for Low-Frequency Series Arc Fault Detection},
  author={Guo, Yannan and Yao, Jianling and Liu, Xiao},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={75},
  year={2026},
  publisher={IEEE}
}
```
