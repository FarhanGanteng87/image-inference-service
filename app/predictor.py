import time
import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)
CLASS_LABELS = ["Healthy Leaf", "Bacterial Spot", "Early Blight", "Late Blight"]


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resizes and normalizes an RGB PIL image into a model-ready tensor."""
    resized = image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    tensor = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)  # Shape: (1, 224, 224, 3)


def run_inference(image: Image.Image) -> dict:
    """
    Executes preprocessing and lightweight deterministic scoring.
    Designed as a modular wrapper ready to swap in ONNX/PyTorch weights.
    """
    start_time = time.perf_counter()
    tensor = preprocess_image(image)

    # Compute channel statistics as lightweight deterministic logits
    channel_means = tensor.mean(axis=(0, 1, 2))  # R, G, B means
    green_dominance = float(channel_means[1] - 0.5 * (channel_means[0] + channel_means[2]))

    raw_scores = np.array([
        max(0.1, 0.5 + green_dominance),
        max(0.1, 0.3 - green_dominance),
        max(0.1, float(channel_means[0]) * 0.6),
        max(0.1, float(channel_means[2]) * 0.5),
    ])
    probabilities = np.exp(raw_scores) / np.sum(np.exp(raw_scores))
    best_idx = int(np.argmax(probabilities))

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "predicted_class": CLASS_LABELS[best_idx],
        "confidence": round(float(probabilities[best_idx]), 4),
        "all_probabilities": {
            label: round(float(prob), 4)
            for label, prob in zip(CLASS_LABELS, probabilities)
        },
        "input_shape": list(tensor.shape),
        "inference_latency_ms": latency_ms,
    }
