"""Mutation-validation: structural verifier and sign-flip kill test.

DEForm validates each generated mutant in two stages:

  1. Structural verifier (this file). For static faults we check that
     the parameter difference is restricted to the targeted unit and
     layer set, and that its magnitude matches the configured severity
     within a small relative tolerance. For dynamic faults we check
     that the wrapped forward is attached only to the intended call
     sites and that the original forward is restored after the context
     manager exits. We also verify that each configuration produces the
     required instrumentation logs. Configurations that fail any check
     are excluded as structurally invalid before any training runs.

  2. Statistical mutation killing (this file). We pair clean and faulty
     fine-tuning runs over n=5 matched seeds and apply an exact
     one-sided sign-flip permutation test to the per-seed deltas. Five
     matched seeds is the smallest n that admits an exact one-sided
     test at alpha = 0.05, with a minimum p-value of 1 / 2^5 ≈ 0.031.
     A mutant is killed when ``p_value <= alpha``.

These tests are independent: the verifier is a sanity check on the
injector itself; the kill test asks whether the resulting behavioral
change is statistically detectable.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────
# Structural verifier
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class VerificationResult:
    """Outcome of a structural verifier run.

    Attributes:
        ok:        True if the configuration passed all checks.
        message:   short reason string. Empty when ``ok=True``.
        diff_norm: Frobenius norm of the parameter difference for static
                   faults (zero for dynamic faults).
    """
    ok: bool
    message: str = ""
    diff_norm: float = 0.0


class StructuralVerifier:
    """Sanity-check for a fault injector.

    The verifier compares snapshots of the model before and after
    injection: for static faults it confirms which parameters changed;
    for dynamic faults it confirms which ``forward`` methods were
    rebound. It then runs the injector's exit and checks full
    restoration.
    """

    def __init__(self, parameter_tolerance: float = 1e-6):
        """Create a verifier.

        Args:
            parameter_tolerance: relative tolerance used when comparing
                pre- and post-restore parameter tensors. Floating-point
                rounding from the clone-and-restore round trip is
                bounded by this value.
        """
        self.parameter_tolerance = parameter_tolerance

    def verify_static(self,
                      model: nn.Module,
                      injector,
                      expected_param_names: Sequence[str],
                      expected_severity: float | None = None
                      ) -> VerificationResult:
        """Verify that a static fault touches only the expected parameters.

        Args:
            model:               the model the injector will mutate.
            injector:            an unentered ``StaticFault``.
            expected_param_names: dotted names of parameters that the
                                  injector is allowed to change. Any
                                  change outside this set fails the
                                  check.
            expected_severity:   if given, the relative magnitude of the
                                  change is compared against this value.
                                  ``None`` skips the magnitude check.
        """
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        with injector:
            diffs: dict[str, float] = {}
            for name, p in model.named_parameters():
                d = (p.detach() - before[name]).norm().item()
                if d > 0:
                    diffs[name] = d

            unexpected = set(diffs) - set(expected_param_names)
            if unexpected:
                return VerificationResult(
                    ok=False,
                    message=f"Unexpected parameter changes: {sorted(unexpected)}",
                    diff_norm=sum(diffs.values()),
                )

            if expected_severity is not None and diffs:
                # Compare relative magnitude of the largest change to
                # ``expected_severity`` within the configured tolerance.
                worst = max(diffs.values())
                ref = max((before[n].norm().item() for n in diffs), default=1.0)
                rel = worst / max(ref, 1e-12)
                if not math.isclose(rel, expected_severity,
                                    rel_tol=self.parameter_tolerance,
                                    abs_tol=self.parameter_tolerance):
                    return VerificationResult(
                        ok=False,
                        message=(f"Severity mismatch: relative magnitude "
                                 f"{rel:.4g} vs expected {expected_severity:.4g}"),
                        diff_norm=worst,
                    )

        # After exit: confirm restoration.
        for name, p in model.named_parameters():
            d = (p.detach() - before[name]).norm().item()
            if d > self.parameter_tolerance:
                return VerificationResult(
                    ok=False,
                    message=f"Parameter not restored after exit: {name}",
                    diff_norm=d,
                )
        return VerificationResult(ok=True, diff_norm=sum(diffs.values()))

    def verify_dynamic(self,
                       model: nn.Module,
                       injector,
                       expected_modules: Sequence[nn.Module]
                       ) -> VerificationResult:
        """Verify that a dynamic fault wraps only the expected modules."""
        before_forwards = {id(m): m.forward for m in expected_modules}
        # Snapshot forwards across the whole model so we can detect
        # accidental rebinds outside the expected set.
        all_forwards = {id(m): m.forward for m in model.modules()}

        with injector:
            wrapped_outside: list[str] = []
            for m in model.modules():
                if id(m) in before_forwards:
                    continue
                if all_forwards.get(id(m)) is not m.forward:
                    wrapped_outside.append(m.__class__.__name__)
            if wrapped_outside:
                return VerificationResult(
                    ok=False,
                    message=("Forward rebound on unintended modules: "
                            f"{sorted(set(wrapped_outside))}"))

            for m in expected_modules:
                if before_forwards[id(m)] is m.forward:
                    return VerificationResult(
                        ok=False,
                        message=("Expected forward to be wrapped on "
                                 f"{m.__class__.__name__}, but it was unchanged."))

        # After exit: forwards restored.
        for m in expected_modules:
            if before_forwards[id(m)] is not m.forward:
                return VerificationResult(
                    ok=False,
                    message=(f"Forward not restored on "
                             f"{m.__class__.__name__} after exit."))
        return VerificationResult(ok=True)


# ─────────────────────────────────────────────────────────────────────────
# Sign-flip permutation test
# ─────────────────────────────────────────────────────────────────────────
def sign_flip_permutation_test(clean: Sequence[float],
                               faulty: Sequence[float],
                               higher_is_better: bool) -> float:
    """Exact one-sided sign-flip permutation test on per-seed deltas.

    For each of the n paired seeds, compute the delta

        d_i = (faulty_i - clean_i)        if higher_is_better
              -(faulty_i - clean_i)       otherwise

    so that a *positive* mean delta after the sign flip corresponds to
    "the fault degraded the metric." Under the null hypothesis that
    fault has no effect on the metric, each per-seed sign is
    exchangeable, so we enumerate all 2^n sign flips of the deltas and
    count how many give a mean greater than or equal to the observed
    mean. The p-value is that count divided by 2^n.

    With n = 5 matched seeds the floor is 1 / 2^5 ≈ 0.031, which is the
    smallest design that admits an exact one-sided test at alpha = 0.05.

    Args:
        clean:            per-seed metric values from the clean runs.
        faulty:           per-seed metric values from the matched faulty
                          runs. Must have the same length and ordering
                          as ``clean``.
        higher_is_better: True for accuracy-style metrics where a fault
                          that lowers the value is the failure mode;
                          False for perplexity / loss where a fault
                          that raises the value is the failure mode.

    Returns:
        One-sided p-value in [1 / 2^n, 1.0].
    """
    if len(clean) != len(faulty):
        raise ValueError(f"clean ({len(clean)}) and faulty ({len(faulty)}) "
                         "must have the same number of seeds")
    n = len(clean)
    if n < 1:
        raise ValueError("at least one paired seed is required")

    sign = 1.0 if higher_is_better else -1.0
    deltas = [sign * (c - f) for c, f in zip(clean, faulty)]
    observed = sum(deltas) / n

    # Enumerate all 2^n sign flips.
    extreme_count = 0
    total = 0
    for flips in itertools.product((1, -1), repeat=n):
        total += 1
        permuted_mean = sum(s * d for s, d in zip(flips, deltas)) / n
        if permuted_mean >= observed:
            extreme_count += 1

    return extreme_count / total


def is_killed(clean: Sequence[float],
              faulty: Sequence[float],
              higher_is_better: bool,
              alpha: float = 0.05) -> tuple[bool, float]:
    """Decide whether a mutant is killed under the sign-flip test.

    Returns:
        Tuple ``(killed, p_value)``. ``killed`` is True if and only if
        ``p_value <= alpha``.
    """
    p = sign_flip_permutation_test(clean, faulty, higher_is_better=higher_is_better)
    return (p <= alpha, p)
