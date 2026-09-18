"""Tests for identity dataset generation."""

import pytest

from eullm_forge.identity import IdentityConfig, generate_identity_dataset


def test_generate_identity_dataset_english():
    config = IdentityConfig(
        identity_name="TestAI",
        languages=["en"],
    )
    examples = generate_identity_dataset(config)
    assert len(examples) >= 6
    assert any("TestAI" in ex["output"] for ex in examples)
    assert any("Who are you" in ex["instruction"] for ex in examples)


def test_generate_identity_dataset_italian():
    config = IdentityConfig(
        identity_name="LegalAI di Studio Rossi",
        languages=["it", "en"],
    )
    examples = generate_identity_dataset(config)
    # Should have both English and Italian examples
    assert any("Chi sei" in ex["instruction"] for ex in examples)
    assert any("Who are you" in ex["instruction"] for ex in examples)
    assert any("LegalAI di Studio Rossi" in ex["output"] for ex in examples)


def test_generate_identity_dataset_german():
    config = IdentityConfig(
        identity_name="MedizinAI",
        languages=["de", "en"],
    )
    examples = generate_identity_dataset(config)
    assert any("Wer bist du" in ex["instruction"] for ex in examples)


def test_generate_identity_dataset_french():
    config = IdentityConfig(
        identity_name="FinanceAI",
        languages=["fr", "en"],
    )
    examples = generate_identity_dataset(config)
    assert any("Qui es-tu" in ex["instruction"] for ex in examples)


def test_generate_identity_dataset_default_name():
    config = IdentityConfig(languages=["en"])
    examples = generate_identity_dataset(config)
    assert any("EULLM Assistant" in ex["output"] for ex in examples)


def test_missing_custom_dataset_path_fails_closed(tmp_path):
    """A non-empty dataset_path that points nowhere must fail, not silently train synthetic.

    Without this, a typo or an unmounted volume falls into the synthetic branch,
    whose "Generated N examples" log is indistinguishable from an intentional run —
    burning 1-2h of GPU and baking generic branding into the merged weights.
    Placed before the heavy imports, so this needs no torch to verify.
    """
    from eullm_forge.identity import fine_tune_identity

    config = IdentityConfig(
        model_path="dummy",
        dataset_path=str(tmp_path / "missing-custom.json"),
    )
    with pytest.raises(FileNotFoundError, match="missing-custom.json"):
        fine_tune_identity(config)
