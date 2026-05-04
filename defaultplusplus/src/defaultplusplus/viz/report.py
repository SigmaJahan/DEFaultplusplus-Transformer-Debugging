"""Self-contained HTML reports.

The output is one HTML file with embedded base64 PNGs — no external
assets, no JavaScript. Open it in a browser, attach to a Slack
message, or commit next to a bug report.

Two entry points:

    save_diagnosis_report(diagnosis, features, path)
        Full debug view. Headline summary + verdict plot + group
        importance + per-layer panels + QKV alignment.

    save_run_report(features, path)
        Sanity-check the feature dict without a diagnosis. Useful
        before pretrained weights exist or when validating a new
        extractor build.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Mapping

from ._deps import require_matplotlib
from . import plots as _plots


def _fig_to_b64_png(fig) -> str:
    """Encode a Figure as a base64 PNG suitable for ``<img src="data:...">``."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _img_tag(fig, alt: str = "") -> str:
    require_matplotlib()
    b64 = _fig_to_b64_png(fig)
    # Free the figure to keep memory bounded when many reports are made
    # in a loop.
    import matplotlib.pyplot as plt
    plt.close(fig)
    return (f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}" '
            f'style="max-width:100%;height:auto;margin:8px 0;" />')


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
       Roboto, Helvetica, Arial, sans-serif; margin: 24px;
       background:#fafafa; color:#222; }
h1 { font-size: 1.5rem; margin-bottom: 4px; }
h2 { font-size: 1.1rem; margin-top: 24px; border-bottom: 1px solid #ddd;
     padding-bottom: 4px; }
.summary { background: #fff; border-left: 4px solid #2980b9;
           padding: 12px 16px; margin: 12px 0; }
.summary.faulty { border-left-color: #c0392b; }
.summary.clean { border-left-color: #27ae60; }
.kv { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.85rem; color:#555; }
section { background:#fff; border:1px solid #eee; padding:8px 16px;
          margin-bottom:16px; border-radius:4px; }
"""


def _summary_paragraph(diagnosis) -> tuple[str, str]:
    """Return (css_class, prose) summarizing the verdict for humans."""
    is_faulty = bool(getattr(diagnosis, "is_faulty", False))
    det_p = float(getattr(diagnosis, "detection_prob", 0.0))
    cat = getattr(diagnosis, "category", None)
    cat_p = float(getattr(diagnosis, "category_prob", 0.0))
    rc = getattr(diagnosis, "root_cause", None)
    rc_p = float(getattr(diagnosis, "root_cause_prob", 0.0))
    importance = getattr(diagnosis, "group_importance", None) or {}

    if not is_faulty:
        return "clean", (
            f"This run looks <b>clean</b> with detection probability "
            f"<b>{det_p:.2f}</b>. No category- or root-cause-level "
            f"diagnosis was produced."
        )

    parts = [f"This run was flagged as <b>faulty</b> with detection "
             f"probability <b>{det_p:.2f}</b>"]
    if cat:
        parts.append(
            f", and the most likely cause is <b>{html.escape(str(rc) or '?')}</b> "
            f"in the <b>{html.escape(str(cat))}</b> subsystem "
            f"(category P={cat_p:.2f}, root-cause P={rc_p:.2f})"
        )
    if importance:
        top = sorted(importance.items(), key=lambda kv: -kv[1])[:3]
        names = ", ".join(f"<b>{html.escape(n)}</b>" for n, _ in top)
        parts.append(
            f". The three feature groups providing the strongest support are {names}."
        )
    else:
        parts.append(".")
    return "faulty", "".join(parts)


def _gather_per_layer_metrics(features: Mapping[str, float]) -> list[str]:
    """Return up to three metric-family substrings worth heatmapping."""
    seen = set()
    candidates: list[str] = []
    for k in features:
        kl = k.lower()
        for needle in ("attn_entropy", "attention_entropy", "ffn_norm",
                       "ffn_delta", "head_similarity", "qk_cos",
                       "res_cos", "ln_gamma", "score_mean"):
            if needle in kl and needle not in seen:
                seen.add(needle)
                candidates.append(needle)
    return candidates[:3]


def save_diagnosis_report(diagnosis,
                          features: Mapping[str, float],
                          path: str | Path) -> None:
    """Write a self-contained HTML diagnosis report.

    ``diagnosis`` is a ``defaultplusplus.diagnosis.Diagnosis``;
    ``features`` is the feature dict the predictor consumed.
    """
    require_matplotlib()
    css_class, summary = _summary_paragraph(diagnosis)

    sections: list[str] = []

    # Verdict + group importance
    sections.append("<section><h2>Verdict</h2>"
                    + _img_tag(_plots.plot_diagnosis(diagnosis),
                               alt="three-stage diagnosis")
                    + "</section>")
    sections.append("<section><h2>Group support</h2>"
                    + _img_tag(_plots.plot_group_importance(diagnosis),
                               alt="group importance bar chart")
                    + "</section>")

    # QKV alignment is a workhorse panel; cheap to always include
    sections.append("<section><h2>Q-K / Q-V / K-V alignment</h2>"
                    + _img_tag(_plots.plot_qkv_alignment(features),
                               alt="qkv alignment per layer")
                    + "</section>")

    # Per-layer heatmaps for whatever families this run carries
    metrics = _gather_per_layer_metrics(features)
    if metrics:
        sections.append("<section><h2>Per-layer heatmaps</h2>")
        for m in metrics:
            sections.append(_img_tag(
                _plots.plot_per_layer_heatmap(features, m),
                alt=f"per-layer heatmap for {m}",
            ))
        sections.append("</section>")

    rc = getattr(diagnosis, "root_cause", None)
    cat = getattr(diagnosis, "category", None)
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>DEFault++ diagnosis</title><style>{_CSS}</style></head>
<body>
<h1>DEFault++ diagnosis report</h1>
<div class="kv">arch={html.escape(str(getattr(diagnosis, '_arch', '') or ''))}
&nbsp;feature_dim={len(features)}</div>
<div class="summary {css_class}">{summary}</div>
{''.join(sections)}
<p class="kv">Predicted: faulty={getattr(diagnosis, 'is_faulty', False)} ·
   category={cat!r} · root_cause={rc!r}</p>
</body></html>
"""
    Path(path).write_text(body, encoding="utf-8")


def save_run_report(features: Mapping[str, float],
                    path: str | Path) -> None:
    """Write a sanity-check HTML report — no diagnosis required.

    Useful before pretrained weights exist, or when validating a new
    extractor build by eyeballing its outputs.
    """
    require_matplotlib()

    sections: list[str] = []
    sections.append("<section><h2>Q-K / Q-V / K-V alignment</h2>"
                    + _img_tag(_plots.plot_qkv_alignment(features),
                               alt="qkv alignment per layer")
                    + "</section>")

    metrics = _gather_per_layer_metrics(features)
    if metrics:
        sections.append("<section><h2>Per-layer heatmaps</h2>")
        for m in metrics:
            sections.append(_img_tag(
                _plots.plot_per_layer_heatmap(features, m),
                alt=f"per-layer heatmap for {m}",
            ))
        sections.append("</section>")

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>DEFault++ run report</title><style>{_CSS}</style></head>
<body>
<h1>DEFault++ run report</h1>
<div class="kv">feature_dim={len(features)}</div>
<div class="summary">Sanity-check view: this run has not been
diagnosed. Use <code>save_diagnosis_report</code> for the full
verdict.</div>
{''.join(sections)}
</body></html>
"""
    Path(path).write_text(body, encoding="utf-8")
