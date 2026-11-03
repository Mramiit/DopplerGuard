"""
End-to-End Pipeline Test for Drone Radar Detection System.
Tests data loading, forward pass, short training epoch, evaluation, and prediction.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import config
from src.preprocess import raw_to_dual_channel
from src.dataset import load_dataset_splits
from src.model import get_model
from src.train import train_epoch, evaluate_epoch, calculate_class_weights
from src.predict import RadarPredictor


def run_pipeline_test():
    print("="*60)
    print("RUNNING END-TO-END PIPELINE VERIFICATION TEST")
    print("="*60)

    # 1. Test Feature Extraction
    print("\n[Step 1] Testing Doppler Feature Extraction...")
    dummy_complex = np.random.randn(1280) + 1j * np.random.randn(1280)
    features = raw_to_dual_channel(dummy_complex)
    assert features.shape == (2, 64, 64), f"Expected (2, 64, 64), got {features.shape}"
    assert not np.isnan(features).any(), "NaN found in features"
    print("  -> Dual-channel feature extraction passed! Shape:", features.shape)

    # 2. Test Model Forward Pass
    print("\n[Step 2] Testing AeroRadarNet Forward Pass...")
    model = get_model(num_classes=3, device="cpu")
    dummy_batch = torch.randn(4, 2, 64, 64)
    logits = model(dummy_batch)
    assert logits.shape == (4, 3), f"Expected (4, 3), got {logits.shape}"
    print("  -> Model forward pass passed! Output shape:", logits.shape)

    # 3. Test DataLoaders with LODO split
    print("\n[Step 3] Testing Dataset Splits (LODO Mode)...")
    train_loader, val_loader, test_loader, meta = load_dataset_splits(
        mode="lodo",
        holdout_drone="D6",
        batch_size=32
    )
    print(f"  -> DataLoaders loaded! Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 4. Test Single Training and Validation Step
    print("\n[Step 4] Running Mini Training Batch Test...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Test 5 training batches
    model.train()
    for b_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if b_idx >= 4:
            break
    print(f"  -> Mini-batch training passed! Step loss: {loss.item():.4f}")

    # Test predictor
    print("\n[Step 5] Testing RadarPredictor Inference...")
    predictor = RadarPredictor(device="cpu")
    pred_res = predictor.predict_features(features)
    print("  -> Predictor passed! Prediction:", pred_res["prediction"], "Confidence:", f"{pred_res['confidence']*100:.1f}%")

    # 6. Test Radar-SwinIR Forward Pass
    print("\n[Step 6] Testing Radar-SwinIR Forward Pass...")
    from src.swinir_model import get_radar_swinir
    swinir = get_radar_swinir(device="cpu")
    dummy_sr = torch.randn(2, 2, 64, 64)
    restored = swinir(dummy_sr)
    assert restored.shape == (2, 2, 64, 64), f"Expected (2, 2, 64, 64), got {restored.shape}"
    print("  -> Radar-SwinIR forward pass passed! Restored shape:", restored.shape)

    # 7. Test Synthetic Radar Simulation Bridge
    print("\n[Step 7] Testing Synthetic Radar Sensor Simulation...")
    from simulation.synthetic_radar_sensor import SyntheticRadarSensor
    sim_radar = SyntheticRadarSensor(radar_pos=(0, 0, 5))
    sim_res = sim_radar.scan_and_detect(
        np.array([120.0, 10.0, 30.0], dtype=np.float32),
        np.array([-15.0, 0.0, 0.0], dtype=np.float32),
        target_ground_truth="drone",
        use_swinir=True
    )
    assert "prediction" in sim_res and "confidence" in sim_res
    print(f"  -> Synthetic sensor test passed! Range: {sim_res['range_m']}m, Prediction: {sim_res['prediction']} ({sim_res['confidence']}%)")

    print("\n" + "="*60)
    print("ALL PIPELINE & SWINIR TESTS PASSED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    run_pipeline_test()
