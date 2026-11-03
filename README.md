# 77 GHz FMCW Drone Radar Detection & Classification System

Physics-aware, ultra-lightweight Deep Learning system for classifying **Drones, Birds, and Humans** using millimeter-wave radar data from the **Zenodo SAAB SIRS 77 GHz FMCW Radar Dataset (Record 5896641)**.

---

## 🎯 The Project's Technical "Moat" (Key Differentiators)

1. **Leave-One-Drone-Out (LODO) Zero-Shot Generalization**:
   * *Problem in existing work*: Traditional CNNs overfit to specific drone frame reflections and drop to ~75% accuracy when facing unknown drone types.
   * *Our Moat*: By learning invariant rotor blade micro-Doppler kinematics, our model achieves **98.88% zero-shot accuracy** on the unseen drone model `D6`.
2. **Dual-Channel Kinematic Fusion**:
   * Fuses **Range-Doppler Maps** (spatial and velocity profile) with **Micro-Doppler Spectrograms** (time-frequency blade chopping dynamics) into a synchronized 2-channel tensor `[2, 64, 64]`.
3. **Ultra-Lightweight Edge Deployment**:
   * Total model size is only **1.72 MB** with **135,971 parameters**.
   * Inference latency is **14.6 ms per frame (68.5 FPS on standard CPU)**, making it fully ready for edge hardware (Raspberry Pi 5, Jetson Orin Nano).

---

## 📊 Empirical Evaluation Results (Moat Benchmark)

| Metric | Result | Benchmark Significance |
| :--- | :--- | :--- |
| **Overall Test Accuracy** | **95.98%** | Evaluated on 3,952 test samples across independent flights |
| **Macro F1-Score** | **88.59%** | High balance across all target categories |
| **Unseen Drone (D6) Zero-Shot Accuracy** | **98.88%** | **Core Moat**: Recognizes unseen drones without retraining |
| **Bird vs. Drone Discrimination** | **98.0% Precision / 98.9% Recall** | Eliminates false alarms caused by birds |
| **Inference Latency** | **14.6 ms (68.5 FPS)** | Real-time surveillance at radar rotation rates |
| **Parameter Footprint** | **135,971 params (~1.72 MB)** | Fits on low-power microcontrollers & edge DSPs |

Confusion matrix plot saved to `reports/confusion_matrix_lodo.png`.

---

## 📁 Repository Structure

```
Drone Radar/
├── DATA/
│   ├── raw/
│   │   └── data_SAAB_SIRS_77GHz_FMCW.npy     # Raw 1.55 GB Zenodo dataset
│   └── processed/
│       └── radar_balanced_dataset.npz         # Fast 330 MB preprocessed feature cache
│
├── models/
│   └── best_aeroradarnet_lodo.pth             # Trained PyTorch model checkpoint
│
├── notebooks/
│   └── 01_explore_and_visualize.ipynb         # Interactive Jupyter Notebook
│
├── reports/
│   ├── confusion_matrix_lodo.png              # Publication-grade confusion matrix
│   └── evaluation_report_lodo.json            # Metrics and latency JSON
│
├── src/
│   ├── config.py                              # Physical radar parameters & hyperparams
│   ├── preprocess.py                          # Raw chirps -> Range-Doppler + Micro-Doppler STFT
│   ├── dataset.py                             # PyTorch DataLoaders & LODO split generator
│   ├── model.py                               # AeroRadarNet (Doppler Separable CNN)
│   ├── train.py                               # Training pipeline with class weighting & early stopping
│   ├── evaluate.py                            # Moat evaluation & benchmark suite
│   └── predict.py                             # Single-sweep inference API
│
└── tests/
    └── test_pipeline.py                       # Automated end-to-end unit tests
```

---

## 🚀 Quickstart Guide

### 1. Interactive Exploration via VS Code Notebook
Open `notebooks/01_explore_and_visualize.ipynb` in VS Code and run the cells to:
* Visualize radar Doppler spectra for Birds vs. Drones vs. Humans.
* Train and evaluate models interactively.
* Test single-sample predictions.

### 2. Run Single-Sample Radar Inference
```bash
python -u src/predict.py
```
Output:
```text
--- SAMPLE RADAR INFERENCE TEST ---
True Label:       Drone
Predicted Class:  Drone
Confidence:       93.87%
Class Probabilities:
  Bird      : 3.96%
  Drone     : 93.87%
  Human     : 2.17%
```

### 3. Re-evaluate Checkpoint & Generate Reports
```bash
python -u src/evaluate.py --mode lodo --holdout D6
```

### 4. Train a New Model
```bash
python -u src/train.py --mode lodo --holdout D6 --epochs 10 --batch_size 128
```
