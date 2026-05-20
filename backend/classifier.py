"""
EchoWarden — Audio Classifier
Wraps the PANNs (Pretrained Audio Neural Networks) model
for real-time environmental sound classification.

First run: downloads ~80MB model weights automatically.
Subsequent runs: loads from cache (~3 seconds).
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Danger sound classes ──────────────────────────────────────────────────────
# Sourced from AudioSet ontology. Expand this list as needed.
DANGER_CLASSES = {
    # Emergency
    "Siren", "Civil defense siren", "Ambulance (siren)", "Police car (siren)",
    "Fire engine, fire truck (siren)", "Emergency vehicle",
    # Human distress
    "Screaming", "Crying, sobbing", "Shout", "Yell", "Crowd",
    # Accidents
    "Glass breaking", "Breaking", "Car alarm",
    "Crash", "Explosion", "Gunshot, gunfire",
    # Traffic
    "Car", "Vehicle horn, car horn, honking", "Truck", "Motorcycle",
    "Reversing beeps", "Skidding",
    # Dogs (approaching threat)
    "Dog", "Bark",
    # Fire
    "Fire", "Smoke detector, smoke alarm", "Fire alarm",
    # Machinery
    "Chainsaw", "Power tool",
}

# ── Ambient (safe) classes — shown as info, not alert ──────────────────────
AMBIENT_CLASSES = {
    "Music", "Speech", "Bird", "Wind", "Rain", "Water",
    "Keyboard", "Mouse", "Typing", "Television",
    "Inside, small room", "Inside, large room or hall",
    "Outside, urban or manmade", "Outside, rural or natural",
}


class AudioClassifier:
    """
    Wraps panns_inference Cnn14 model.
    Falls back to a mock classifier if panns_inference is not installed,
    so you can test the API structure without the model.
    """

    def __init__(self):
        self.loaded = False
        self._model = None
        self._labels = None
        self._load()

    def _load(self):
        try:
            from panns_inference import AudioTagging
            self._model = AudioTagging(checkpoint_path=None, device="cpu")
            self._labels = self._load_labels()
            self.loaded = True
            logger.info("✅ PANNs Cnn14 model loaded successfully")
        except ImportError:
            logger.warning(
                "⚠️  panns_inference not installed. Using mock classifier.\n"
                "   Install with: pip install panns_inference"
            )
            self.loaded = False
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.loaded = False

    def _load_labels(self):
        """Load AudioSet class labels (527 classes)."""
        try:
            import panns_inference
            import os
            label_path = os.path.join(
                os.path.dirname(panns_inference.__file__),
                "config.py"
            )
            # panns_inference exposes labels via its config
            from panns_inference.config import labels
            return labels
        except Exception:
            # Fallback: return a minimal label set
            return [f"Class_{i}" for i in range(527)]

    def classify(self, audio: np.ndarray, sample_rate: int) -> list[tuple[str, float]]:
        """
        Classify audio array (mono float32).
        Returns list of (label, confidence) tuples sorted by confidence desc.
        """
        # Resample to 32kHz if needed (PANNs requirement)
        if sample_rate != 32000:
            audio = self._resample(audio, sample_rate, 32000)

        # Pad or trim to 1 second (32000 samples)
        target_len = 32000
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        if self.loaded and self._model is not None:
            try:
                # PANNs expects shape (batch, samples)
                audio_input = audio[np.newaxis, :]
                clipwise_output, _ = self._model.inference(audio_input)
                scores = clipwise_output[0]  # shape: (527,)

                results = sorted(
                    zip(self._labels, scores.tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )
                return results[:10]
            except Exception as e:
                logger.error(f"Inference error: {e}")

        # Mock fallback for development without the model
        return self._mock_classify(audio)

    def is_danger(self, label: str) -> bool:
        """Return True if this sound class is considered dangerous."""
        return any(danger.lower() in label.lower() for danger in DANGER_CLASSES)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback: naive resampling
            ratio = target_sr / orig_sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def _mock_classify(self, audio: np.ndarray) -> list[tuple[str, float]]:
        """Mock results when model is not installed — useful for UI development."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.01:
            return [("Silence", 0.95), ("Inside, small room", 0.03), ("White noise", 0.02)]
        return [
            ("Speech", 0.72),
            ("Inside, small room", 0.15),
            ("Music", 0.08),
            ("Keyboard", 0.03),
            ("Mouse", 0.02),
        ]
