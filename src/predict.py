"""
Single-Sweep Radar Target Inference.
Allows predicting class and confidence on a raw radar sweep or test sample.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

try:
    from src import config
    from src.model import get_model
    from src.preprocess import raw_to_dual_channel
except ImportError:
    import config
    from model import get_model
    from preprocess import raw_to_dual_channel


class RadarPredictor:
    def __init__(self, model_path: Path = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model_path is None:
            model_path = config.MODEL_SAVE_DIR / "best_aeroradarnet_lodo.pth"
            if not model_path.exists():
                model_path = config.MODEL_SAVE_DIR / "best_aeroradarnet_standard.pth"

        if not model_path.exists():
            print(f"[RadarPredictor] Checkpoint not found at {model_path}. Using fresh AeroRadarNet model for demo inference.")
            self.model = get_model(num_classes=3, device=self.device)
        else:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = get_model(num_classes=3, device=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.class_names = [config.COARSE_CLASSES[i] for i in range(3)]

    def predict_raw_sweep(self, sweep_vector: np.ndarray):
        """
        Takes raw 1280 complex samples, computes dual-channel Doppler features,
        and returns prediction and class probabilities.
        """
        feat = raw_to_dual_channel(sweep_vector)
        return self.predict_features(feat)

    def predict_features(self, feat_array: np.ndarray):
        """
        Takes preprocessed (2, 64, 64) float array and returns prediction.
        """
        if feat_array.ndim == 3:
            feat_tensor = torch.from_numpy(feat_array).float().unsqueeze(0).to(self.device)
        else:
            feat_tensor = torch.from_numpy(feat_array).float().to(self.device)

        with torch.no_grad():
            logits = self.model(feat_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        probabilities = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": probabilities
        }


if __name__ == "__main__":
    # Test on a random preprocessed sample from processed dataset
    data_path = config.PROCESSED_DIR / "radar_balanced_dataset.npz"
    if data_path.exists():
        data = np.load(data_path, allow_pickle=True)
        idx = 42
        sample_x = data["X"][idx]
        true_y = data["y_coarse"][idx]
        true_label = config.COARSE_CLASSES[int(true_y)]

        try:
            predictor = RadarPredictor()
            res = predictor.predict_features(sample_x)
            print("\n--- SAMPLE RADAR INFERENCE TEST ---")
            print(f"True Label:       {true_label}")
            print(f"Predicted Class:  {res['prediction']}")
            print(f"Confidence:       {res['confidence']*100:.2f}%")
            print("Class Probabilities:")
            for c, p in res["probabilities"].items():
                print(f"  {c:10s}: {p*100:.2f}%")
        except Exception as e:
            print("Note: Train the model first before running inference CLI:", e)
