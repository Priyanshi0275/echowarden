"""
EchoWarden — Quick Test
Run this first to verify your setup before launching the server.

Usage:
    python test_classifier.py                    # test with synthetic audio
    python test_classifier.py path/to/file.wav   # test with a real audio file
"""

import sys
import numpy as np
from classifier import AudioClassifier


def test_with_synthetic():
    print("Testing with synthetic audio (440Hz tone)...")
    sample_rate = 32000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


def test_with_file(path):
    import soundfile as sf
    print(f"Testing with file: {path}")
    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sample_rate


if __name__ == "__main__":
    print("\n🎧 EchoWarden classifier test\n" + "─" * 40)

    clf = AudioClassifier()
    print(f"Model loaded: {clf.loaded}")
    print()

    if len(sys.argv) > 1:
        audio, sr = test_with_file(sys.argv[1])
    else:
        audio, sr = test_with_synthetic()

    results = clf.classify(audio, sr)

    print("Top 5 predictions:")
    for i, (label, score) in enumerate(results[:5], 1):
        danger_marker = " ⚠ DANGER" if clf.is_danger(label) else ""
        bar = "█" * int(score * 30)
        print(f"  {i}. {label:<35} {score:.3f}  {bar}{danger_marker}")

    print("\n✅ Classifier test complete. Run `python main.py` to start the server.")
