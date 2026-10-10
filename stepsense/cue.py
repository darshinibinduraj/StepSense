"""Generate rhythmic metronome cue audio for StepSense alerts."""

from __future__ import annotations

import io
import wave

import numpy as np


def generate_metronome_wav_bytes(
    bpm: float = 100.0,
    duration_seconds: float = 6.0,
    sample_rate: int = 44100,
) -> bytes:
    """Generate a short click-track WAV clip using numpy + wave."""

    if bpm <= 0:
        raise ValueError("bpm must be > 0")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    beat_interval_seconds = 60.0 / bpm
    total_samples = int(duration_seconds * sample_rate)
    waveform = np.zeros(total_samples, dtype=np.float32)

    click_length = int(0.03 * sample_rate)
    click_envelope = np.exp(-np.linspace(0.0, 6.0, click_length))
    click_carrier = np.sin(2.0 * np.pi * 1800.0 * np.arange(click_length) / sample_rate)
    click = 0.7 * click_carrier * click_envelope

    beat_starts = np.arange(0.0, duration_seconds, beat_interval_seconds)
    for beat_start in beat_starts:
        start_index = int(beat_start * sample_rate)
        end_index = min(total_samples, start_index + click_length)
        length = end_index - start_index
        if length > 0:
            waveform[start_index:end_index] += click[:length]

    waveform = np.clip(waveform, -1.0, 1.0)
    pcm_int16 = (waveform * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_int16.tobytes())

    return buffer.getvalue()

