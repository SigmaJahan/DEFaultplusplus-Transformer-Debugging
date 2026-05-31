"""Correct-class construction for DEFault-bench.

The faulty class of DEFault-bench comes from killed mutants. The *correct*
class comes from clean (unmutated) base models with label-preserving
perturbations, the only instances whose label is directly known rather
than inferred.

For each base model that produces ``k`` killed mutants, we generate ``k``
clean variants from the same base model. Each variant holds the base model
fixed and varies only factors that should not change task behavior:

  * the random seed, which also reshuffles the training-data order, and
  * hyperparameters within ranges that do not move task behavior past the
    kill threshold.

Each variant is then evaluated against the base model with the same
one-sided sign-flip permutation test used to identify killed mutants
(:func:`deform.validation.is_killed`). A variant that stays statistically
indistinguishable from the base model is retained as a *correct* sample. A
variant that satisfies the killed criterion is discarded. This rule is the
mirror image of the faulty path and keeps construction symmetric across the
two classes.

This module owns generation (:func:`generate_clean_variants`) and the
per-variant run (:func:`run_one_clean_variant`). The fine-tuning callable
and the feature builder are injected by the caller, exactly as for the
faulty path, so this module has no HuggingFace or GPU dependency and can be
unit-tested with stub callables.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from .fault_config import CleanVariant, CorrectSample
from .validation import is_killed


# Behavior-preserving hyperparameter ranges. A variant draws one override
# per perturbed key from the list assigned to that key. These ranges are
# deliberately narrow: they introduce optimization noise without moving
# task behavior past the kill threshold. The empty override (vary the seed
# only) is always available as the first option.
DEFAULT_HYPERPARAM_GRID: dict[str, Sequence[Any]] = {
    "learning_rate": (3e-5, 5e-5, 7e-5),
    "batch_size": (8, 16),
    "warmup_ratio": (0.0, 0.06, 0.1),
}


# A fine-tune callable for clean variants takes the base model name, the
# task, a seed, and a hyperparameter-override dict, and returns the test
# metric plus the training trace. It mirrors the faulty-path FineTuneFn but
# carries hyperparameter overrides instead of a fault injector.
CleanFineTuneFn = Callable[[str, str, int, dict[str, Any]], tuple[float, dict]]
FeatureBuilderFn = Callable[[list[dict], list[dict]], dict]


def generate_clean_variants(
        model: str,
        task: str,
        n_variants: int,
        *,
        base_seed: int = 42,
        hyperparam_grid: dict[str, Sequence[Any]] | None = None,
        vary_hyperparams: bool = True,
        seed_step: int = 1000,
) -> list[CleanVariant]:
    """Generate ``n_variants`` label-preserving variants of a base model.

    Each variant gets a distinct ``variant_seed`` (so re-running fine-tuning
    reshuffles the data order) and, when ``vary_hyperparams`` is set, one
    behavior-preserving hyperparameter override drawn deterministically from
    ``hyperparam_grid``. Generation is deterministic: the same arguments
    always produce the same variants, so a SLURM array index maps to a
    stable variant.

    Args:
        model:           clean base-model name.
        task:            downstream task name.
        n_variants:      number of variants to generate. The paper sets
                         this to ``k``, the number of killed mutants the
                         base model produced, so the faulty-to-correct
                         ratio stays balanced per base model.
        base_seed:       seed of the base reference run. Variant seeds are
                         offset from it so they never collide with it.
        hyperparam_grid: per-key candidate values. Defaults to
                         :data:`DEFAULT_HYPERPARAM_GRID`.
        vary_hyperparams: when False, only the seed varies (every variant
                         uses base hyperparameters).
        seed_step:       spacing between variant seeds.

    Returns:
        A list of ``n_variants`` :class:`CleanVariant` objects.
    """
    if n_variants < 0:
        raise ValueError(f"n_variants must be non-negative, got {n_variants}")
    grid = hyperparam_grid if hyperparam_grid is not None else DEFAULT_HYPERPARAM_GRID
    keys = list(grid)

    variants: list[CleanVariant] = []
    for i in range(n_variants):
        variant_seed = base_seed + seed_step * (i + 1)
        hyperparams: dict[str, Any] = {}
        if vary_hyperparams and keys:
            # Rotate through one hyperparameter per variant so the overrides
            # spread across keys deterministically rather than concentrating
            # on a single one.
            key = keys[i % len(keys)]
            choices = grid[key]
            if choices:
                hyperparams = {key: choices[i % len(choices)]}
        variants.append(
            CleanVariant(
                model=model,
                task=task,
                variant_seed=variant_seed,
                hyperparams=hyperparams,
                variant_index=i,
            )
        )
    return variants


def run_one_clean_variant(
        variant: CleanVariant,
        fine_tune: CleanFineTuneFn,
        feature_builder: FeatureBuilderFn,
        higher_is_better: bool,
        *,
        alpha: float = 0.05,
        seeds: Iterable[int] | None = None,
        base_hyperparams: dict[str, Any] | None = None,
) -> CorrectSample:
    """Run one clean variant against the base model and label it.

    The base model is fine-tuned with base hyperparameters, and the variant
    is fine-tuned with its label-preserving perturbation, both across the
    same matched seeds. The same sign-flip permutation test used for killed
    mutants then compares the two metric distributions.

    A variant that stays statistically indistinguishable from the base
    model (``killed`` is False) is retained as a *correct* sample. A variant
    that satisfies the killed criterion is discarded (``retained`` is
    False), exactly mirroring how the faulty path discards a surviving
    mutant.

    Args:
        variant:          the :class:`CleanVariant` to run.
        fine_tune:        callable ``(model, task, seed, hyperparams) ->
                          (metric, trace)``.
        feature_builder:  builds the labeled instance from paired base /
                          variant traces.
        higher_is_better: whether the task metric is accuracy-like or
                          perplexity-like.
        alpha:            kill threshold (default 0.05), the same value the
                          faulty path uses.
        seeds:            matched seeds. Defaults to the variant's single
                          seed; production callers pass the full five.
        base_hyperparams: hyperparameters for the base reference run. The
                          variant applies its own overrides on top of these.

    Returns:
        A :class:`CorrectSample`. ``retained`` is True only when the variant
        is statistically indistinguishable from the base model.
    """
    seed_list = list(seeds) if seeds is not None else [variant.variant_seed]
    base_hp = dict(base_hyperparams or {})
    variant_hp = {**base_hp, **variant.hyperparams}

    base_metrics: list[float] = []
    variant_metrics: list[float] = []
    base_traces: list[dict] = []
    variant_traces: list[dict] = []

    for seed in seed_list:
        b_metric, b_trace = fine_tune(variant.model, variant.task, seed, base_hp)
        if not _is_finite(b_metric):
            return CorrectSample(
                variant=variant,
                rejected_reason=(
                    f"base run produced non-finite metric (seed={seed}, value={b_metric!r})"
                ),
            )
        base_metrics.append(float(b_metric))
        base_traces.append(b_trace)

        v_metric, v_trace = fine_tune(variant.model, variant.task, seed, variant_hp)
        if not _is_finite(v_metric):
            return CorrectSample(
                variant=variant,
                rejected_reason=(
                    f"variant run produced non-finite metric (seed={seed}, value={v_metric!r})"
                ),
            )
        variant_metrics.append(float(v_metric))
        variant_traces.append(v_trace)

    killed, p = is_killed(base_metrics, variant_metrics,
                          higher_is_better=higher_is_better, alpha=alpha)
    retained = not killed

    feature_vector = None
    rejected_reason = ""
    if retained:
        feature_vector = feature_builder(base_traces, variant_traces)
    else:
        rejected_reason = (
            f"clean variant satisfied the killed criterion (p={p:.4f} < alpha={alpha})"
        )

    return CorrectSample(
        variant=variant,
        base_metrics=tuple(base_metrics),
        variant_metrics=tuple(variant_metrics),
        p_value=p,
        killed=killed,
        retained=retained,
        feature_vector=feature_vector,
        rejected_reason=rejected_reason,
    )


def _is_finite(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))
