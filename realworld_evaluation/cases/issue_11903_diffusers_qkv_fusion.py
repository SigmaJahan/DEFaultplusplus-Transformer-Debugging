from __future__ import annotations

import numpy as np

from realworld_evaluation.common import BenchmarkCase, CaseMetadata, CaseResult, seeded_rng


def build_projections(rng: np.random.Generator, dim: int) -> dict[str, np.ndarray]:
    return {
        "q_proj.weight": rng.normal(size=(dim, dim)),
        "k_proj.weight": rng.normal(size=(dim, dim)),
        "v_proj.weight": rng.normal(size=(dim, dim)),
    }


def fuse_qkv(projections: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate q/k/v into the fused ``to_qkv`` weight the forward uses."""
    return np.concatenate(
        [projections["q_proj.weight"], projections["k_proj.weight"], projections["v_proj.weight"]],
        axis=0,
    )


def forward(x: np.ndarray, to_qkv: np.ndarray) -> np.ndarray:
    """The fused forward path: only ``to_qkv`` is used to produce Q, K, V."""
    return x @ to_qkv.T


def lora_update(weight: np.ndarray, rng: np.random.Generator, scale: float = 0.1) -> np.ndarray:
    """A non-zero LoRA-style additive update applied to a target weight."""
    return weight + scale * rng.normal(size=weight.shape)


def buggy_path(projections, to_qkv, x, rng):
    """LoRA targets the stale q/k/v modules that the forward no longer uses.

    The split projections are updated, but the fused ``to_qkv`` weight that
    the forward path actually reads is left unchanged, so the model output
    does not move even though parameters changed.
    """
    updated = {name: lora_update(w, rng) for name, w in projections.items()}
    out_before = forward(x, to_qkv)
    out_after = forward(x, to_qkv)  # to_qkv untouched -> identical output
    # Update activity on the stale projections (non-zero) vs the active
    # fused path (zero), the signature DEFault++ reads.
    stale_update_ratio = float(
        np.mean([
            np.linalg.norm(updated[name] - projections[name]) / (np.linalg.norm(projections[name]) + 1e-12)
            for name in projections
        ])
    )
    active_update_ratio = 0.0
    return out_before, out_after, stale_update_ratio, active_update_ratio


def fixed_path(projections, to_qkv, x, rng):
    """LoRA targets the fused projection that the forward path uses.

    Updating ``to_qkv`` changes the model output, so the update is no
    longer stale.
    """
    updated_to_qkv = lora_update(to_qkv, rng)
    out_before = forward(x, to_qkv)
    out_after = forward(x, updated_to_qkv)
    stale_update_ratio = 0.0
    active_update_ratio = float(
        np.linalg.norm(updated_to_qkv - to_qkv) / (np.linalg.norm(to_qkv) + 1e-12)
    )
    return out_before, out_after, stale_update_ratio, active_update_ratio


def run() -> CaseResult:
    rng = seeded_rng(11903)
    dim = 4
    projections = build_projections(rng, dim)
    to_qkv = fuse_qkv(projections)
    x = rng.normal(size=(3, dim))

    _, buggy_after, buggy_stale, buggy_active = buggy_path(projections, to_qkv, x, seeded_rng(1))
    buggy_before, _ = forward(x, to_qkv), None
    fixed_before, fixed_after, fixed_stale, fixed_active = fixed_path(projections, to_qkv, x, seeded_rng(1))

    # Buggy: parameters change but the forward output does not.
    buggy_output_delta = float(np.max(np.abs(buggy_after - buggy_before)))
    # Fixed: updating the active fused path moves the output.
    fixed_output_delta = float(np.max(np.abs(fixed_after - fixed_before)))

    reproduced = (
        buggy_stale > 1e-3            # stale projections were updated
        and buggy_output_delta < 1e-10  # yet the output did not move
        and fixed_output_delta > 1e-3   # updating the active path does move it
    )
    return CaseResult(
        reproduced=reproduced,
        summary=f"stale update ratio {buggy_stale:.3f} with output delta {buggy_output_delta:.2e}",
        details={
            "buggy_stale_update_ratio": buggy_stale,
            "buggy_active_update_ratio": buggy_active,
            "buggy_output_delta": buggy_output_delta,
            "fixed_stale_update_ratio": fixed_stale,
            "fixed_active_update_ratio": fixed_active,
            "fixed_output_delta": fixed_output_delta,
        },
    )


CASE = BenchmarkCase(
    metadata=CaseMetadata(
        row_id=11903,
        slug="issue_11903_diffusers_qkv_fusion",
        title="LoRA updates stale q/k/v projections after qkv fusion",
        issue_url="https://github.com/huggingface/diffusers/issues/11903",
        source_repo="huggingface/diffusers",
        source_component="fuse_qkv_projections and LoRA adapter targeting",
        fix_url=None,
        symptom="Fine-tuning runs normally and the loss decreases, but model behavior does not change because LoRA updates projection modules that the fused forward path no longer uses.",
        dataset="synthetic projection tensors",
        fault_family="QKV Projection",
    ),
    run=run,
)
