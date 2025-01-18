#!/usr/bin/env python3
#testing
import os
import unittest
from pathlib import Path
from typing import List, Dict
import numpy as np
from tinygrad.tensor import Tensor
# Remove the extra.utils import since we don't need fetch for local files

class TestWhisperModelSizes(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Define test data path
    cls.test_dir = Path("LibriSpeech/test-clean/61/70968")
    cls.test_file = str(cls.test_dir / "61-70968-0000.flac")
    cls.test_transcript = str(cls.test_dir / "61-70968.trans.txt")
    
    # Load ground truth
    with open(cls.test_transcript, 'r') as f:
      cls.ground_truth = {}
      for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
          key, text = parts
          cls.ground_truth[key] = text

  def test_tiny_model_transcription(self):
    from examples.whisper import init_whisper, transcribe_file
    model, enc = init_whisper("tiny.en")
    transcription = transcribe_file(model, enc, self.test_file)
    self.assertIsNotNone(transcription)
    self.assertNotEqual(transcription.strip(), "")
    # Compare with ground truth
    key = Path(self.test_file).stem
    if key in self.ground_truth:
      self.assertLess(len(transcription.strip()) - len(self.ground_truth[key].strip()), 10)

  def test_small_model_transcription(self):
    from examples.whisper import init_whisper, transcribe_file
    os.environ["SMALL"] = "1"
    model, enc = init_whisper("small.en")
    transcription = transcribe_file(model, enc, self.test_file)
    self.assertIsNotNone(transcription)
    self.assertNotEqual(transcription.strip(), "")
    # Compare with ground truth
    key = Path(self.test_file).stem
    if key in self.ground_truth:
      self.assertLess(len(transcription.strip()) - len(self.ground_truth[key].strip()), 10)
    del os.environ["SMALL"]

  def test_medium_model_transcription(self):
    from examples.whisper import init_whisper, transcribe_file
    os.environ["MEDIUM"] = "1"
    model, enc = init_whisper("medium.en")
    transcription = transcribe_file(model, enc, self.test_file)
    self.assertIsNotNone(transcription)
    self.assertNotEqual(transcription.strip(), "")
    # Compare with ground truth
    key = Path(self.test_file).stem
    if key in self.ground_truth:
      self.assertLess(len(transcription.strip()) - len(self.ground_truth[key].strip()), 10)
    del os.environ["MEDIUM"]

  def test_model_parameter_differences(self):
    """Test that different model sizes have different parameter counts"""
    from examples.whisper import init_whisper
    
    def get_param_count(model):
      return sum(x.realize().numpy().size for x in model.parameters())

    tiny_model = init_whisper("tiny.en")[0]
    tiny_params = get_param_count(tiny_model)
    
    os.environ["SMALL"] = "1"
    small_model = init_whisper("small.en")[0]
    small_params = get_param_count(small_model)
    del os.environ["SMALL"]
    
    os.environ["MEDIUM"] = "1"
    medium_model = init_whisper("medium.en")[0]
    medium_params = get_param_count(medium_model)
    del os.environ["MEDIUM"]
    
    self.assertGreater(small_params, tiny_params)
    self.assertGreater(medium_params, small_params)

if __name__ == "__main__":
  unittest.main()
