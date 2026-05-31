"""Tests for the correct-class construction (label-preserving variants).

The faulty class comes from killed mutants. The correct class comes from
clean base models with label-preserving perturbations: a variant that stays
statistically indistinguishable from the base model under the same kill test
is retained as a correct sample, and one that satisfies the killed criterion
is discarded. These tests pin that behavior with stub fine-tune callables so
they run without HuggingFace or a GPU.
"""
from __future__ import annotations


SEEDS = [42, 123, 456, 789, 101112]


def test_generate_clean_variants_produces_k_distinct_seeds() -> None:
    from defaultplusplus.deform import generate_clean_variants

    variants = generate_clean_variants("bert-base-uncased", "sst2", 5, base_seed=42)
    assert len(variants) == 5
    seeds = {v.variant_seed for v in variants}
    assert len(seeds) == 5                      # every variant re-seeds
    assert 42 not in seeds                      # never collides with the base seed
    assert all(v.model == "bert-base-uncased" and v.task == "sst2" for v in variants)
    assert all(v.variant_index == i for i, v in enumerate(variants))


def test_generate_is_deterministic() -> None:
    from defaultplusplus.deform import generate_clean_variants

    a = generate_clean_variants("gpt2", "wikitext2", 4, base_seed=42)
    b = generate_clean_variants("gpt2", "wikitext2", 4, base_seed=42)
    assert [v.config_id() for v in a] == [v.config_id() for v in b]


def test_seed_only_variants_carry_no_hyperparam_override() -> None:
    from defaultplusplus.deform import generate_clean_variants

    variants = generate_clean_variants(
        "bert-base-uncased", "sst2", 3, vary_hyperparams=False)
    assert all(v.hyperparams == {} for v in variants)


def test_label_preserving_variant_is_retained_as_correct() -> None:
    """A variant indistinguishable from the base model is a correct sample."""
    from defaultplusplus.deform import generate_clean_variants, run_one_clean_variant

    variant = generate_clean_variants("bert-base-uncased", "sst2", 1, base_seed=42)[0]

    def fine_tune(model, task, seed, hyperparams):
        # Base and variant draw from the same distribution -> not killed.
        return 0.90 + (seed % 7) * 0.001, {"acc": 0.90}

    result = run_one_clean_variant(
        variant, fine_tune, lambda b, v: {"feat": 1.0},
        higher_is_better=True, seeds=SEEDS)

    assert result.retained is True
    assert result.killed is False
    assert result.feature_vector is not None
    assert result.rejected_reason == ""


def test_behavior_shifting_variant_is_discarded() -> None:
    """A variant that satisfies the killed criterion is not a correct sample."""
    from defaultplusplus.deform import generate_clean_variants, run_one_clean_variant

    variant = generate_clean_variants("bert-base-uncased", "sst2", 1, base_seed=42)[0]

    def fine_tune(model, task, seed, hyperparams):
        # The variant carries a hyperparameter override (non-empty hyperparams),
        # and here it consistently lowers the metric on every seed -> killed.
        return (0.70 if hyperparams else 0.90), {"acc": 0.90}

    result = run_one_clean_variant(
        variant, fine_tune, lambda b, v: {"feat": 1.0},
        higher_is_better=True, seeds=SEEDS)

    assert result.killed is True
    assert result.retained is False
    assert result.feature_vector is None
    assert "killed criterion" in result.rejected_reason


def test_non_finite_metric_discards_variant() -> None:
    from defaultplusplus.deform import generate_clean_variants, run_one_clean_variant

    variant = generate_clean_variants("gpt2", "wikitext2", 1, base_seed=42)[0]

    def fine_tune(model, task, seed, hyperparams):
        return float("nan"), {}

    result = run_one_clean_variant(
        variant, fine_tune, lambda b, v: {"feat": 1.0},
        higher_is_better=False, seeds=SEEDS)

    assert result.retained is False
    assert "non-finite" in result.rejected_reason


def test_perplexity_direction_is_respected() -> None:
    """For perplexity (lower is better), a variant that lowers it is fine."""
    from defaultplusplus.deform import generate_clean_variants, run_one_clean_variant

    variant = generate_clean_variants("gpt2", "wikitext2", 1, base_seed=42)[0]

    def fine_tune(model, task, seed, hyperparams):
        # Base and variant indistinguishable -> retained.
        return 3.10 + (seed % 5) * 0.002, {"loss": 3.10}

    result = run_one_clean_variant(
        variant, fine_tune, lambda b, v: {"feat": 1.0},
        higher_is_better=False, seeds=SEEDS)

    assert result.retained is True
