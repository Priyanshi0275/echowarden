"""
EchoWarden — Backend Server
"""

import os
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import logging

from classifier import AudioClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EchoWarden API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = AudioClassifier()


class SoundEvent(BaseModel):
    label: str
    confidence: float
    is_danger: bool
    direction: str


class ClassificationResult(BaseModel):
    events: List[SoundEvent]
    top_label: str
    is_danger: bool
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier.loaded}


@app.post("/classify", response_model=ClassificationResult)
async def classify_audio(file: UploadFile = File(...)):
    import time
    import av
    start = time.time()

    contents = await file.read()
    tmp_path = None

    try:
        # Save to temp file — av handles ANY format Android sends
        with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False, dir=os.getcwd()) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Decode with PyAV — works with mp4, m4a, aac, wav, everything
        container = av.open(tmp_path, 'r')
        audio_stream = next(s for s in container.streams if s.type == 'audio')
        sample_rate = audio_stream.sample_rate

        samples = []
        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()
            samples.append(arr)
        container.close()
        del container

        if not samples:
            raise ValueError("No audio frames decoded")

        # Merge frames → mono float32
        data = np.concatenate(samples, axis=1)   # shape: (channels, samples)
        mono = data.mean(axis=0).astype(np.float32)

        # Normalize
        max_val = np.max(np.abs(mono))
        if max_val > 0:
            mono = mono / max_val

        direction = "center"

    except Exception as e:
        logger.error(f"Audio load error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Could not read audio: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    results = classifier.classify(mono, sample_rate)
    latency_ms = (time.time() - start) * 1000

    events = []
    for label, confidence in results[:5]:
        events.append(SoundEvent(
            label=label,
            confidence=round(float(confidence), 3),
            is_danger=classifier.is_danger(label) and float(confidence)> 0.5,
            direction=direction,
        ))

    top = events[0] if events else SoundEvent(
        label="Silence", confidence=1.0, is_danger=False, direction="center"
    )

    logger.info(f"✅ {top.label} ({top.confidence:.2f}) | danger={top.is_danger} | {latency_ms:.0f}ms")

    return ClassificationResult(
        events=events,
        top_label=top.label,
        is_danger=top.is_danger,
        latency_ms=round(latency_ms, 1),
    )


if __name__ == "__main__":
    print("\n🎧 EchoWarden backend starting on http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)