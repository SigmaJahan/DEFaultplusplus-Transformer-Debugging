# NDG Stage-3 Scoring — Mathematical Specification

## 1. Overview

Stage-3 performs **root cause diagnosis** using the Neural Diagnosis Graph (NDG).

Given:

* Stage-2 fault family probabilities ( p_\theta(f \mid x) )
* SHAP feature attributions
* Counterfactual feature changes
* (Optional) mutation-grounded subsystem impacts and differential signatures
* Per-run anomaly scores ( z_v ) for core features

NDG computes a ranked list of fault families using a structured, evidence-weighted scoring function.

This document defines the exact scoring equation and shows how it maps to the implementation in `ndg_stage3/inference.py`.

---

## 2. Notation

Let:

* ( f \in \mathcal{F} ) — fault family
* ( c \in \mathcal{C} ) — core feature
* ( s \in \mathcal{S} ) — subsystem
* ( x ) — observed mutant run
* ( z_c(x) ) — anomaly score (robust z-score) for core feature ( c )
* ( w^{\text{shap}}_{f,c} ) — SHAP importance weight for feature ( c ) with respect to family ( f )
* ( w^{\text{impact}}_{f,s} ) — mutation-grounded subsystem impact weight
* ( p_\theta(f \mid x) ) — Stage-2 model probability for family ( f )

We define:

[
\phi(z) = \min(|z|, z_{\max})
]

where ( z_{\max} = 6 ) (clipping to prevent dominance from extreme outliers).

---

## 3. Subsystem Aggregation

Subsystem activation is computed by pooling feature anomalies:

[
z_s(x) = \max_{c \in \mathcal{C}(s)} \phi(z_c(x))
]

This matches the implementation in:

```
aggregate_subsystem_anomalies()
```

in `inference.py`.

Max pooling is used (not sum) because:

* It preserves sharp fault localization.
* It avoids dilution from many small noisy features.
* It mirrors the thesis claim that subsystems activate when at least one strong indicator deviates.

---

## 4. NDG Family Scoring Equation

The NDG score for family ( f ) is:

[
\text{Score}(f \mid x)
======================

\log p_\theta(f \mid x)
+
\alpha \sum_{c \in \mathcal{C}} w^{\text{shap}}*{f,c} , \phi(z_c(x))
+
\beta \sum*{s \in \mathcal{S}} w^{\text{impact}}_{f,s} , z_s(x)
]

Where:

* ( \alpha ) controls feature-level evidence weight
* ( \beta ) controls subsystem-level mutation-grounded weight

Default configuration:

[
\alpha = 0.5, \quad \beta = 0.5
]

This exactly matches the implementation in:

```
score_families()
```

in `ndg_stage3/inference.py`.

---

## 5. Interpretation of Terms

### 5.1 Log Prior Term

[
\log p_\theta(f \mid x)
]

This term anchors diagnosis in Stage-2 classification output.

It prevents NDG from ignoring learned discriminative structure.

If no priors are provided, uniform priors are assumed.

---

### 5.2 Core Evidence Term

[
\alpha \sum_{c} w^{\text{shap}}_{f,c} \phi(z_c)
]

This captures:

* Feature anomaly magnitude
* Weighted by SHAP attribution strength

Only features that both:

* are anomalous
* and are important for classification

contribute meaningfully.

This term is entirely Stage-2 + XAI grounded.

---

### 5.3 Subsystem Impact Term

[
\beta \sum_{s} w^{\text{impact}}_{f,s} z_s
]

This is mutation-grounded structural reasoning:

* ( w^{\text{impact}}_{f,s} ) comes from differential signature JSON
* Encodes how strongly family ( f ) affects subsystem ( s )
* Subsystem activation is pooled from features

This is what makes NDG more than SHAP.

Without this term, NDG reduces to weighted SHAP scoring.

With it, NDG becomes architecture-aware.

---

## 6. Confusable Alternatives

From the confusion matrix:

[
\text{CONFUSABLE_WITH}(f, f') = \text{CM}_{f,f'}
]

Top-k off-diagonal normalized confusion rates are converted to graph edges.

These are not part of the main score equation, but are used to:

* Provide contrastive explanations
* Rank alternative hypotheses

---

## 7. Mutation-Grounded Signature Edges

When diagnosis JSONs are supplied, NDG adds:

[
\text{SIGNATURE}(f, c) = (\text{direction}, \text{effect_size})
]

These edges are not directly summed in the base scoring equation, but can be used to:

* Validate anomaly direction consistency
* Add directional gating if desired:

Optional directional gating:

[
\text{match}(c,f) =
\begin{cases}
1 & \text{if anomaly direction matches signature direction} \
0 & \text{otherwise}
\end{cases}
]

If enabled, the core term becomes:

[
\sum_c w^{\text{shap}}_{f,c} \cdot \text{match}(c,f) \cdot \phi(z_c)
]

This extension is consistent with your thesis formulation but is currently optional.

---

## 8. Final Ranking

After computing:

[
\text{Score}(f \mid x)
]

Families are ranked:

[
f^* = \arg\max_f \text{Score}(f \mid x)
]

Top-k are returned for diagnostic reporting.

---

## 9. Neural Stack Trace Extraction

For the top family ( f^* ):

1. Identify top activated subsystems by ( w^{\text{impact}}_{f^*,s} z_s )
2. Within each subsystem, rank features by:
   [
   w^{\text{shap}}_{f^*,c} \phi(z_c)
   ]
3. Extract minimal evidence chain covering 80% of score mass

This yields:

```
Family → Subsystem → CoreFeature → FeatureVariant → Remediation
```

This corresponds exactly to the thesis’s “AST-like neural stack trace.”

---

## 10. Guarantees

The NDG scoring:

* Is monotonic in anomaly magnitude.
* Is bounded (via clipping).
* Integrates both discriminative and structural evidence.
* Reduces to Stage-2 classification if α=β=0.
* Reduces to SHAP-only if β=0.
* Reduces to mutation-only structural reasoning if α=0.

This modularity is intentional and mathematically consistent.

---

## 11. Summary

The implemented NDG scoring exactly matches the thesis equation:

[
\boxed{
\text{Score}(f \mid x)
======================

\log p_\theta(f \mid x)
+
\alpha \sum_{c} w^{\text{shap}}*{f,c} \phi(z_c)
+
\beta \sum*{s} w^{\text{impact}}_{f,s} z_s
}
]

with:

* structured feature pooling,
* architecture-aware subsystem aggregation,
* mutation-grounded signature priors,
* and confusion-derived alternative hypotheses.

This is the precise mathematical backbone of Stage-3 NDG.

---

