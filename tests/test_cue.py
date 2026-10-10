"""Tests for rhythmic cue generation."""

from __future__ import annotations

import io
import unittest
import wave

from stepsense.cue import generate_metronome_wav_bytes


class CueTests(unittest.TestCase):
    def test_generate_metronome_wav_bytes(self) -> None:
        payload = generate_metronome_wav_bytes(bpm=100.0, duration_seconds=1.0, sample_rate=16000)
        self.assertGreater(len(payload), 44)

        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertGreater(wav_file.getnframes(), 0)


if __name__ == "__main__":
    unittest.main()

