"""Tests for pruning calibration data loading."""

import pytest


def test_unknown_calibration_dataset_raises_instead_of_silent_fallback(monkeypatch):
    """An explicitly requested calibration dataset that fails to load must fail loudly.

    Falling back to wikitext here would score neuron importance on the wrong
    corpus while logging the requested name, pruning the wrong neurons — damage
    discovered only after days of distillation on the pruned model.
    """
    import sys
    import types
    from unittest.mock import MagicMock

    from eullm_forge import pruning as pruning_module

    def _raise(*args, **kwargs):
        raise FileNotFoundError("Dataset 'no-such-calib' doesn't exist")

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = _raise
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="no-such-calib"):
        pruning_module._load_calibration_data("no-such-calib", tokenizer=MagicMock(), num_samples=8)
